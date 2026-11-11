# Methodology

## Classification targets

The task is multi-label classification across 25 shared mood names. Books use a threshold of 4 on original 1–5 ratings. Song labels are encoded as selected/unselected. Raw ratings remain available for EDA and a known book's weighting profile.

## Features and models

- **Books:** TF-IDF with at most 500 features. Compare One-vs-Rest Logistic Regression with One-vs-Rest Linear SVM.
- **Songs:** four Librosa features. Standardize using training-set statistics and compare One-vs-Rest Logistic Regression with a multi-output Random Forest of 100 trees, maximum depth 5.

TF-IDF and standardization are fitted only on training examples. Random splits use seed 42. There are 10/2/3 books and 56/8/16 songs in train/validation/test. Validation micro F1 selects the classifier. The book candidates tie, so Logistic Regression is retained as the baseline. Song validation selects Logistic Regression.

The split is not grouped by author or artist. Repeated styles or artists may therefore influence estimates. Rare labels may have no positive training examples. A larger dataset is needed for stronger evaluation.

## Evaluation

Precision measures the correctness of predicted labels; recall measures how many target labels are found. Micro F1 combines all label decisions and balances precision and recall. Frequent labels influence it more than rare labels. Exact-match accuracy requires the whole set of 25 predictions to agree with the target for one item.

Training/testing comparisons reveal performance gaps. Confusion matrices for Dark explain errors for one emotion; they do not summarize all emotions. Results are exploratory because the test sets and annotation coverage are small.

## Matching in the notebook

For a new description, the selected book classifier produces emotion support scores. Each song receives predicted binary emotion labels from its audio features. The matching calculation is:

```text
match score = sum of book weights covered by predicted song labels
              / sum of all book emotion weights
```

For a known book, original ratings can be used as weights. Model scores for new descriptions are not predicted 1–5 questionnaire ratings or proven measures of liking. The rule may favor songs with more predicted labels.

The song library includes training songs. This is an inference demonstration, not an independent recommendation-quality test. The presentation app currently keeps a fixed song selection and order while displaying calculated scores. Listener evaluation remains future work.
