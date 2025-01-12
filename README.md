# Hate Speech Detection using NLP and Machine Learning

This project focuses on detecting hate speech and offensive language in tweets using natural language processing (NLP) and machine learning techniques. The model achieves **88% accuracy** on the test dataset.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Preprocessing Steps](#preprocessing-steps)
- [Model and Evaluation](#model-and-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)

---

## Introduction
Social media platforms generate massive amounts of user-generated content daily. Among this content, hate speech and offensive language pose significant challenges for content moderation. This project utilizes **machine learning** to classify tweets into:
- **Hate Speech**
- **Offensive Language**
- **Neither**

---

## Dataset
The dataset used for training and testing is `twitter.csv`, containing labeled tweets with categories for:
- **Hate Speech**
- **Offensive Language**
- **No Hate or Offensive Language**

### Dataset Overview:
- **Columns**:
  - `tweet`: The text of the tweet.
  - `class`: The label indicating the tweet's classification:
    - `0`: Hate Speech
    - `1`: Offensive Language
    - `2`: No Hate or Offensive Language

---

## Technologies Used
The project leverages the following Python libraries:
- **Data Manipulation**: `pandas`, `numpy`
- **NLP**: `nltk`
- **Feature Extraction**: `scikit-learn`'s `CountVectorizer`
- **Modeling**: `scikit-learn`'s `DecisionTreeClassifier`
- **Visualization**: `seaborn`

---

## Preprocessing Steps
1. **Text Cleaning**:
   - Converted text to lowercase.
   - Removed URLs, HTML tags, punctuation, and numbers.
2. **Stopword Removal**:
   - Used `nltk.corpus.stopwords` to remove common English stopwords.
3. **Stemming**:
   - Applied stemming using `nltk.SnowballStemmer` to reduce words to their root forms.
4. **Feature Extraction**:
   - Used `CountVectorizer` to convert text into numerical features with a vocabulary size of 10,000.

---

## Model and Evaluation
### Model:
- **Algorithm**: Decision Tree Classifier
- **Hyperparameters**:
  - `max_depth=25`
  - `min_samples_split=8`
  - `random_state=42`

### Results:
- **Accuracy**: **88%**
- **Evaluation Metrics**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

---

## How to Run
Follow these steps to set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hate-speech-detection.git
   cd hate-speech-detection
