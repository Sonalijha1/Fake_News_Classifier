# 📰 Fake News Classifier – ML Project with Streamlit Web App

A machine learning project that classifies news articles as **Real** or **Fake** using a TF-IDF vectorizer and a PassiveAggressiveClassifier. This includes a clean and interactive **Streamlit web app** for real-time predictions.

---

## ✅ Features
- Preprocessed real and fake news articles from the [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Applied TF-IDF with bigrams
- Trained a Passive Aggressive Classifier (92.3% accuracy)
- Streamlit web app for easy input and prediction
- Model saved and loaded using `pickle`

---

## 📊 Dataset
Used the **Fake and Real News Dataset** from Kaggle:
- `Fake.csv` — fake articles from unreliable sources
- `True.csv` — real news articles from trusted publishers

Combined and cleaned, then split into training/testing sets.

---

## 🧠 Model Overview
- **Text Vectorization**: `TfidfVectorizer` with 10,000 features, bigrams
- **Classifier**: `PassiveAggressiveClassifier`
- **Accuracy**: 92.3% on test set
- **Limitation**: May misclassify longer, politically charged real news due to TF-IDF limitations and dataset bias.

---

## 🖥️ Streamlit App

To run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
