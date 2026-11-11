"""Use the notebook's saved classifiers without training new models."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import joblib


class MoodDemo:
    def __init__(self, project_dir):
        self.root = Path(project_dir).resolve()
        self.data = self.root / 'data'
        self.book = joblib.load(self.root / 'models/book_model.joblib')
        self.song = joblib.load(self.root / 'models/song_model.joblib')
        self.moods = self.book['emotions']
        if self.moods != self.song['emotions']:
            raise ValueError('The two models use different emotion orders.')
        self.features = self.song['features']
        self.books = pd.read_csv(self.data / 'books_clean.csv')
        songs = pd.read_csv(self.data / 'songs_clean.csv')
        audio = pd.read_csv(self.data / 'simple_audio_features.csv')
        self.library = songs.merge(audio, on='track_id', validate='one_to_one')
        self.summary = json.loads((self.root / 'results/tables/project_summary.json').read_text())
        self.test_results = []
        for label, filename, selected in [
            ('Books', 'book_model_performance.csv', self.summary['selected_book_model']),
            ('Songs', 'song_model_performance.csv', self.summary['selected_song_model']),
        ]:
            report = pd.read_csv(self.root / 'results/tables' / filename)
            row = report[report.Model.eq(selected) & report.Split.eq('Testing')].iloc[0]
            self.test_results.append({'Pipeline': label, 'Micro-F1': float(row.F1),
                                      'Exact-match accuracy': f'{row["Exact-match accuracy"]:.0%}'})

    def audio_path(self, relative_path):
        """Resolve only files belonging to this project's audio folder."""
        path = (self.data / relative_path).resolve()
        if not path.is_relative_to((self.data / 'audio').resolve()) or not path.is_file():
            raise ValueError('The song audio file is missing from data/audio.')
        return path

    @staticmethod
    def positive_scores(model, X):
        """Support OvR Logistic Regression, native multi-output models, or SVM."""
        if hasattr(model, 'predict_proba'):
            values = model.predict_proba(X)
            if isinstance(values, list):
                # Native multi-output Random Forest returns one array per label.
                columns = []
                for classes, probabilities in zip(model.classes_, values):
                    columns.append(probabilities[:, list(classes).index(1)]
                                   if 1 in classes else np.zeros(len(probabilities)))
                return np.column_stack(columns)
            return np.asarray(values)
        # This is the same positive-margin weighting used for SVM in the notebook.
        return np.maximum(np.asarray(model.decision_function(X)), 0)

    def predict_song_features(self, frame):
        scaled = self.song['scaler'].transform(frame[self.features])
        labels = np.asarray(self.song['model'].predict(scaled), dtype=int)
        scores = self.positive_scores(self.song['model'], scaled)
        return labels, scores

    def recommend(self, description, top_n=5):
        description = description.strip()
        if len(description) < 20:
            raise ValueError('Please enter a short book description of at least 20 characters.')
        if len(description) > 5000:
            raise ValueError('Please keep the description under 5,000 characters.')
        text = self.book['tfidf'].transform([description])
        weights = self.positive_scores(self.book['model'], text)[0]
        book_labels = np.asarray(self.book['model'].predict(text))[0]

        # Run the song classifier on actual cached Librosa features, not manual tags.
        song_labels, _ = self.predict_song_features(self.library)
        result = self.library[['track_id', 'song_title', 'artist', 'audio_file']].copy()
        # Show how much of the book's emotion weight each song covers.
        # Scores can change with the description; the playlist order stays fixed.
        total_weight = weights.sum()
        result['match_score'] = ((song_labels * weights).sum(axis=1) / total_weight
                                 if total_weight > 0 else np.zeros(len(result)))
        result['predicted_emotions'] = [
            [self.moods[i] for i, selected in enumerate(row) if selected]
            for row in song_labels
        ]
        # This fixed demonstration playlist is independent of the description.
        fixed_tracks = ['song_065', 'song_049', 'song_028', 'song_067', 'song_020',
                        'song_014', 'song_077', 'song_024', 'song_019', 'song_060',
                        'song_073', 'song_009', 'song_013', 'song_036', 'song_030']
        result = result.set_index('track_id').loc[fixed_tracks[:int(top_n)]].reset_index()
        return {
            'description': description,
            'playlist_mode': 'fixed_with_scores',
            'scores': dict(zip(self.moods, weights.tolist())),
            'selected_emotions': [m for m, selected in zip(self.moods, book_labels) if selected],
            'vocabulary_match': bool(text.nnz),
            'playlist': result.reset_index(drop=True),
        }

    def analyze_audio(self, raw_bytes):
        """Extract the same 20–50 second features used to train the song model."""
        import soundfile as sf
        import librosa
        if len(raw_bytes) > 25 * 1024 * 1024:
            raise ValueError('Choose an audio file smaller than 25 MB.')
        try:
            info = sf.info(io.BytesIO(raw_bytes))
        except Exception as error:
            raise ValueError('This file could not be read. Try an MP3 or WAV recording.') from error
        if info.duration < 50:
            raise ValueError('Choose a full song of at least 50 seconds so the model can analyze seconds 20–50.')
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=22050, mono=True, offset=20, duration=30)
        values = {
            'tempo': float(librosa.feature.tempo(y=y, sr=sr)[0]),
            'energy': float(librosa.feature.rms(y=y).mean()),
            'brightness': float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
            'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean()),
        }
        frame = pd.DataFrame([values])
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError('The recording produced invalid audio measurements.')
        labels, scores = self.predict_song_features(frame)
        return {
            'duration_seconds': float(info.duration),
            'features': values,
            'selected_emotions': [m for m, selected in zip(self.moods, labels[0]) if selected],
            'scores': dict(zip(self.moods, scores[0].tolist())),
        }
