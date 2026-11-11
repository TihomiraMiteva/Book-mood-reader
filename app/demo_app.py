"""Local presentation interface: python -m streamlit run demo_app.py"""
from pathlib import Path
import html
import importlib
import pandas as pd
import streamlit as st
import demo_engine

# Reload changed matching code, including when a cached app is already running.
MoodDemo = importlib.reload(demo_engine).MoodDemo

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title='Book Mood Reader', page_icon='📖', layout='wide')

# Keep the presentation simple, with paper-like colors and readable spacing.
st.markdown('''<style>
.stApp {background: #f7f6f1; color: #233c37;}
.block-container {max-width: 1120px; padding-top: 2.3rem; padding-bottom: 3rem;}
h1 {letter-spacing: -1.5px; font-weight: 750;}
h2, h3 {letter-spacing: -.4px;}
div[data-testid="stMetric"] {background: #eeeee6; border-radius: 12px; padding: 12px 18px;}
.emotion-tag {display:inline-block; background:#e5eee5; color:#285444;
 padding:5px 11px; border-radius:30px; margin:3px 5px 3px 0; font-size:.88rem;}
.song-title {font-size:1.15rem;font-weight:650;margin:0;line-height:1.5;}
.match-pill {background:#eeeee6;border-radius:10px;padding:8px 10px;text-align:center;}
.match-pill span {font-size:.7rem;display:block;}
.match-pill strong {font-size:1.25rem;display:block;white-space:nowrap;}
.eyebrow {font-size:.76rem;letter-spacing:2px;text-transform:uppercase;color:#738479;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_demo(book_modified, song_modified, data_modified, engine_modified):
    # The arguments refresh the cache when the notebook saves new model/data files.
    return MoodDemo(ROOT)


try:
    demo = load_demo(
        (ROOT / 'models/book_model.joblib').stat().st_mtime,
        (ROOT / 'models/song_model.joblib').stat().st_mtime,
        (ROOT / 'data/simple_audio_features.csv').stat().st_mtime,
        Path(__file__).with_name('demo_engine.py').stat().st_mtime,
    )
except Exception as error:
    st.error(f'Could not load the project files: {error}')
    st.info('Keep this app inside the extracted project folder, alongside data/ and models/.')
    st.stop()


def tags(names):
    st.markdown(''.join(f'<span class="emotion-tag">{html.escape(name)}</span>' for name in names),
                unsafe_allow_html=True)


def emotion_scores(scores):
    # Show the five highest scores, without inventing additional predicted labels.
    for mood, score in sorted(scores.items(), key=lambda pair: -pair[1])[:5]:
        st.progress(min(max(float(score), 0), 1), text=f'{mood} · {score:.0%}')
    st.caption('Scores show model support. They are not 1–5 questionnaire ratings or measured accuracy.')


st.markdown('<div class="eyebrow">An AI semester project</div>', unsafe_allow_html=True)
st.title('Book Mood Reader')
st.write('Describe a story. Explore its emotions. Listen to a fixed demo playlist.')
a, b, c = st.columns(3)
a.metric('Books in the dataset', len(demo.books))
b.metric('Songs in the library', len(demo.library))
c.metric('Shared emotions', len(demo.moods))
st.caption('A small-data learning demo. Results are suggestions and may not reflect your interpretation.')

book_tab, song_tab, about_tab = st.tabs(['Book to playlist', 'Try the song model', 'How it works'])

with book_tab:
    st.subheader('1. Describe your book')
    st.caption('Describe the story, characters, themes and conflict in a few sentences.')
    examples = {
        'Mystery example': ('A young woman investigates a disappearance in an isolated town. '
                            'As dangerous secrets emerge, she must decide whom to trust '
                            'while protecting the people she loves.'),
        'Relationship example': ('Two friends reunite after years apart and discover that their feelings '
                                 'have never disappeared. They help one another through loss and change '
                                 'while deciding whether to rebuild their relationship.'),
    }
    e1, e2, _ = st.columns([1, 1.2, 2.4])
    for column, (label, description) in zip([e1, e2], examples.items()):
        if column.button(label, use_container_width=True):
            st.session_state['book_description'] = description
    description = st.text_area('Book description', key='book_description', height=145,
                               placeholder='A young woman discovers a dangerous secret…', max_chars=5000)
    left, right = st.columns([3, 1])
    count = right.selectbox('Number of songs', [5, 10, 15], index=0)
    if left.button('Generate my song list', type='primary', use_container_width=True):
        try:
            with st.spinner('Running the book and song classifiers…'):
                st.session_state['book_result'] = demo.recommend(description, count)
        except ValueError as error:
            st.warning(str(error))

    # Clear results generated before the fixed-playlist update.
    if st.session_state.get('book_result', {}).get('playlist_mode') != 'fixed_with_scores':
        st.session_state.pop('book_result', None)
    result = st.session_state.get('book_result')
    if result:
        st.divider()
        if (description.strip() != result['description'] or len(result['playlist']) != count
):
            st.info('These results are from the previous request. Generate again to apply your changes.')
        if not result['vocabulary_match']:
            st.warning('The description has no words in the small training vocabulary. These scores mainly reflect the model baseline.')
        scores_column, playlist_column = st.columns([1, 2.1], gap='large')
        with scores_column:
            st.subheader('2. Book emotion scores')
            emotion_scores(result['scores'])
            st.write('Labels selected by the book classifier')
            if result['selected_emotions']:
                tags(result['selected_emotions'])
            else:
                st.caption('No strong emotion label was selected at the classifier threshold.')
        with playlist_column:
            st.subheader('3. Fixed demo playlist')
            st.caption('The playlist stays fixed. Match scores use your book description and each song’s predicted emotions; they do not measure how much you will like a song.')
            for rank, row in enumerate(result['playlist'].itertuples(), 1):
                with st.container(border=True):
                    info, score = st.columns([3, 1])
                    info.markdown(f'<div class="song-title">{rank}. {html.escape(row.song_title)}</div>', unsafe_allow_html=True)
                    info.caption(row.artist)
                    score.markdown(f'<div class="match-pill"><span>Match score</span><strong>{row.match_score:.1%}</strong></div>', unsafe_allow_html=True)
                    if row.predicted_emotions:
                        tags(row.predicted_emotions)
                    else:
                        st.caption('The song model selected no emotion labels.')
                    # Cached predictions work even when recordings are not distributed.
                    try:
                        local_audio = demo.audio_path(row.audio_file)
                    except ValueError:
                        st.caption('Audio playback is unavailable. Add your local recording to data/audio/.')
                    else:
                        st.audio(str(local_audio), format='audio/mpeg')
            export = result['playlist'].copy()
            export['predicted_emotions'] = export['predicted_emotions'].apply(lambda names: '; '.join(names))
            st.download_button('Save song list as CSV', export.to_csv(index=False).encode('utf-8-sig'),
                               file_name='book_mood_playlist.csv', mime='text/csv')

with song_tab:
    st.subheader('Listen to a song, then predict its emotions')
    st.write('This runs Librosa on the recording and passes the four measurements to the saved song classifier.')
    source = st.radio('Choose an audio source', ['A song from my library', 'Upload a song'], horizontal=True)
    audio_bytes = None
    audio_format = 'audio/mpeg'
    selected_name = ''
    if source == 'A song from my library':
        ordered = demo.library.sort_values(['song_title', 'artist'])
        choices = ordered.track_id.tolist()
        names = {r.track_id: f'{r.song_title} — {r.artist}' for r in ordered.itertuples()}
        selected = st.selectbox('Song', choices, format_func=lambda value: names[value])
        track = ordered[ordered.track_id.eq(selected)].iloc[0]
        selected_name = names[selected]
        try:
            audio_bytes = demo.audio_path(track.audio_file).read_bytes()
        except ValueError:
            st.info('This recording is not included in the repository. Add it to data/audio/ or choose Upload a song.')
    else:
        uploaded = st.file_uploader('Choose an MP3, WAV, FLAC or OGG file', type=['mp3', 'wav', 'flac', 'ogg'])
        st.caption('Use a recording between 50 seconds and 25 MB. The app analyzes seconds 20–50, just like the notebook.')
        if uploaded is not None:
            selected_name = uploaded.name
            audio_bytes = uploaded.getvalue()
            audio_format = {'.wav': 'audio/wav', '.flac': 'audio/flac', '.ogg': 'audio/ogg'}.get(
                Path(uploaded.name).suffix.lower(), 'audio/mpeg')
    if audio_bytes is not None:
        st.audio(audio_bytes, format=audio_format)
    if st.button('Predict this song’s emotions', type='primary', disabled=audio_bytes is None):
        try:
            with st.spinner('Extracting audio features and running the song classifier…'):
                st.session_state['song_result'] = (selected_name, demo.analyze_audio(audio_bytes))
        except ValueError as error:
            st.warning(str(error))
        except Exception:
            st.error('This audio file could not be analyzed. Try another readable MP3 or WAV recording.')
    previous = st.session_state.get('song_result')
    if previous:
        name, prediction = previous
        st.divider()
        st.subheader(f'Prediction: {name}')
        if prediction['selected_emotions']:
            tags(prediction['selected_emotions'])
        else:
            st.info('The model selected no labels above its decision threshold. The scores below show its strongest remaining suggestions.')
        emotion_scores(prediction['scores'])
        with st.expander('See the four audio measurements'):
            st.dataframe(pd.Series(prediction['features'], name='Value').to_frame(), use_container_width=True)

with about_tab:
    st.subheader('Two models, connected through emotions')
    st.markdown('''
**Book pipeline:** description → TF-IDF → One-vs-Rest Logistic Regression → emotion scores.

**Song pipeline:** audio → Librosa features → One-vs-Rest Logistic Regression → emotion labels.

**Playlist:** this interface shows a fixed demonstration playlist for every description. Match scores sum the book emotion weights covered by each song’s predicted labels, divided by the total book emotion weight. Scores do not change the fixed playlist order.

For the library, audio features are already cached. The app still runs the song classifier; it does not copy the manually assigned mood tags. The separate song tab extracts features again so you can demonstrate the complete audio step.

The book target is a questionnaire rating of **4 or 5 = strongly present**. Scores shown for a new description are classifier scores, not predicted 1–5 ratings.
''')
    st.write('Saved model results on the small held-out test sets')
    st.dataframe(pd.DataFrame(demo.test_results), hide_index=True, use_container_width=True)
    st.caption('Exact-match accuracy requires every emotion label for an example to agree. These small, subjective datasets limit the results; playlist quality has not been independently evaluated.')
