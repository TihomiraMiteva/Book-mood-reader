# Data provenance and local audio

## Included files

- `raw/book_moods_clean_ratings.csv`: 15 questionnaire book records with title, author, genre, description and 25 original 1–5 ratings.
- `raw/songs.xlsx`: the final five-column song table through `emotion_labels`.
- `books_clean.csv`: cleaned book records, preserving original ratings.
- `songs_clean.csv`: 80 song records with `track_id`, `song_title`, `artist`, `audio_file` and `emotion_labels`.
- `simple_audio_features.csv`: `track_id`, `tempo`, `energy`, `brightness` and `zero_crossing_rate`, extracted from the original recordings.

Book ratings 4–5 become positive strong-emotion targets; 1–3 become negative strong-emotion targets. Song targets indicate whether a label was selected. These are related but different annotation methods. Some cleaning and song annotation suggestions used AI assistance. Song labels have not undergone independent review.

## Local recordings

Recordings are omitted from Git and the repository ZIP. To enable library playback, place your local copies under `data/audio/` using the exact names referenced in the `audio_file` column. Values such as `audio/001_example.mp3` are relative to `data/`.

The supplied feature cache supports the notebook's modeling experiment and the demo's cached song predictions without recordings. Missing recordings prevent checking their quality or repeating feature extraction. The notebook reports this distinction. The app shows an unavailable-playback message instead of failing.

Do not replace a recording under the same track ID and assume its old features remain valid. If changing the recordings or extraction settings, restore all required audio and regenerate `simple_audio_features.csv` with the notebook.

## Audio features

The analysis uses seconds 20–50, mono audio at 22,050 Hz:

| Column | Meaning |
| --- | --- |
| `tempo` | Estimated beat rate in beats per minute |
| `energy` | Mean RMS amplitude, a measure of signal energy |
| `brightness` | Mean spectral centroid in Hz |
| `zero_crossing_rate` | Mean fraction of sign changes in the waveform |

These measurements describe sound properties, not direct measurements of emotions. They do not represent lyrics or the full recording.

The source PDFs in `docs/reports/` document the original collection plan. The final implementation preserves all book ratings and derives strong-emotion targets rather than selecting only one highest-rated emotion.
