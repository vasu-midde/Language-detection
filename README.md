# Language Detection with NLP

Classifies a sentence into one of **17 languages** using a bag-of-words model. Multinomial Naive Bayes reaches **97.7 % accuracy** on a held-out test set.

## Results

| Model | Test accuracy |
|---|---|
| **Multinomial Naive Bayes** | **0.977** |
| Random Forest | 0.931 |
| K-Nearest Neighbours | 0.526 |

70 / 30 train–test split (`random_state=0`). Full classification reports and confusion matrices are in the notebook.

## Data

`Language Detection.csv`: 10,337 text samples across 17 languages, including English, French, Spanish, Portuguese, Italian, Russian, Swedish, Malayalam, Dutch, Arabic, Turkish, German, Tamil, Danish, Kannada, Greek and Hindi. The notebook drops 66 duplicate rows, leaving 10,271 samples.

## Approach

1. **Cleaning:** strip symbols and digits, then lower-case the text (regex).
2. **Features:** `CountVectorizer` bag-of-words, which gives 39,370 features.
3. **Labels:** `LabelEncoder`.
4. **Models:** compare KNN, Random Forest and Multinomial Naive Bayes.
5. **Inference:** a `prediction(text)` helper that returns the detected language.

```python
prediction("Η μνήμη σας βελτιώνεται καθώς μαθαίνετε μια γλώσσα.")
# This word/sentence contains Greek word(s).
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook languageDetection.ipynb
```

## Tech

Python · scikit-learn · pandas · NumPy · Matplotlib · seaborn
