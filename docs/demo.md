# Local demo

## Start

From the repository root, create a Python 3.12 virtual environment and install `requirements.txt`. Then run:

```powershell
.venv\Scripts\python.exe -m streamlit run app/demo_app.py
```

Or double-click `RUN_DEMO.bat`. Keep the terminal open and visit http://127.0.0.1:8501. Stop the server with Ctrl+C. The app loads the saved models; it does not train them.

## Book to playlist

1. Choose an example or enter a description of at least 20 characters.
2. Select 5, 10 or 15 songs and click **Generate my song list**.
3. Inspect the book emotion predictions and each song's predicted emotions and match score.
4. Download the displayed list as CSV if needed.

This presentation version intentionally uses a **fixed playlist**. The same requested number of songs appears in the same order for every description. Scores still use the current description, but they do not select or reorder the playlist. This differs from the notebook's mood-ranked matching experiment.

## Song prediction

Choose a local library recording or upload an MP3, WAV, FLAC or OGG. A recording must be at least 50 seconds long and no larger than 25 MB. The app analyzes seconds 20–50 and applies the saved song classifier.

Original recordings are not distributed. Without them, cached library predictions still work, playback is unavailable, and the song tab prompts you to upload a recording. Restore local audio using `data/README.md` to demonstrate library playback.

## Interpretation

Book scores are classifier support, not questionnaire ratings. Match scores are not probabilities that a listener will like a song. The reported F1 and accuracy concern held-out emotion classification, not the quality of the fixed playlist.
