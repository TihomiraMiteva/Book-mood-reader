# Reflection

Book Mood Reader has potential as a way to help readers explore music that fits a story's atmosphere. The project demonstrates the complete process: collecting and cleaning data, exploring patterns, preparing text and audio features, comparing classifiers, evaluating predictions and building a small interface.

The main limitation was the data. A short collection period restricted the amount and variety I could gather. There are only 15 books and 80 songs across 25 emotion labels, with very uneven coverage. Emotion is also subjective, and some song annotations received AI assistance rather than independent review. These limitations make reliable learning and matching difficult.

The results show why training performance alone is not enough. Linear SVM achieved a book training micro F1 of 1.00 but only 0.36 on testing. Random Forest fell from approximately 0.83 to 0.41 for songs. These gaps are consistent with overfitting. Logistic Regression was selected for both pipelines using validation micro F1, but its test results of 0.37 for books and 0.45 for songs remain modest. The book test set contains only three examples, so its score is particularly uncertain.

I learned how to turn a project idea into a classification workflow, why clean and consistent labels matter, how TF-IDF and audio features represent different inputs, and how validation, testing and confusion matrices reveal weaknesses that training scores can hide. I also learned that successful software integration does not prove that recommendations feel right to users.

The next iteration needs substantially more diverse books and songs, especially examples of underrepresented emotions. Labels should be reviewed by multiple people where possible. Further experiments should compare against a simple baseline, evaluate on more unseen examples, and consider separating authors or artists across splits. More data is an important next step, but better results are not guaranteed without good annotations, suitable features and careful evaluation.

Finally, model evaluation and playlist evaluation are different tasks. Listener feedback is needed to determine whether a suggested playlist fits a book. The current presentation app uses a fixed playlist with scores, so it does not demonstrate personalized playlist selection. The notebook retains the mood-based ranking experiment. This project is a promising prototype and a valuable learning experience, rather than evidence of reliable emotional matching.
