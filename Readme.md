# 🧠 Hate Speech Classification using RNN

## 📌 Objective

The goal of this project is to build a **binary text classification model** that detects whether a tweet contains hate speech or not. The model is implemented using a **Recurrent Neural Network (RNN)** to capture sequential patterns in text data.

---

## 📂 Dataset

* The dataset consists of tweets labeled for hate speech.
* Only the following columns are used:

  * `tweet` → input text
  * `label` → target (0 = non-hate, 1 = hate speech)

---

## 🧹 Text Preprocessing

To ensure clean and meaningful input for the model, the following preprocessing steps are applied:

* Convert all text to lowercase
* Remove:

  * User mentions (`@username`)
  * URLs
  * Special characters and numbers
* Remove extra whitespaces

---

## 🔤 Tokenization & Vocabulary

* Tweets are tokenized into individual words
* A vocabulary is built from the dataset
* Vocabulary size is limited to the **top 5000 most frequent words**
* Special tokens:

  * `<PAD>` → index `0` (used for padding)
  * `<OOV>` → used for unknown/out-of-vocabulary words

---

## 🔢 Text to Sequences

* Each tweet is converted into a sequence of integers
* Words are mapped to their corresponding indices
* Sequences are padded to a fixed length:

```python
maxlen = 25  # based on 95th percentile of token lengths
```

* Padding and truncation are applied at the end of sequences

---

## 🔀 Train-Test Split

* Dataset is split into:

  * **80% training**
  * **20% testing**
* Stratified sampling is used to preserve class distribution

---

## 🏗️ Model Architecture

The model is built using a simple RNN-based architecture:

* **Embedding Layer**

  * Converts word indices into dense vector representations

* **Simple RNN Layer**

  * Captures sequential dependencies in text

* **Dense Output Layer**

  * Single neuron with sigmoid activation for binary classification

---

## ⚙️ Training Configuration

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Epochs:** 10
* **Batch Size:** 32

---

## 📊 Evaluation Results

### ✅ Overall Performance

* **Accuracy:** **94.79%**

### 📈 Classification Report

```
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      5942
           1       0.66      0.54      0.60       451

    accuracy                           0.95      6393
   macro avg       0.81      0.76      0.78      6393
weighted avg       0.94      0.95      0.95      6393
```

---

## 🧠 Interpretation of Results

* The model performs **very well on the majority class (non-hate speech)**
* Performance on **hate speech (class 1)** is weaker:

  * Lower **recall (0.54)** → model misses many hate tweets
  * Moderate **precision (0.66)** → some false positives

👉 This indicates a **class imbalance problem**, where the model is biased toward the majority class.

---

## 🚀 How to Run

1. Load and preprocess the dataset
2. Tokenize and convert text to padded sequences
3. Split data into training and testing sets
4. Build and compile the RNN model
5. Train the model
6. Evaluate performance on test data

---

## 📌 Future Improvements

* Replace SimpleRNN with **LSTM/GRU** for better sequence learning
* Handle class imbalance using:

  * Class weights
  * Oversampling / undersampling
* Use pre-trained embeddings (e.g., GloVe)
* Tune decision threshold instead of fixed 0.5
* Hyperparameter tuning for improved recall on hate speech

---

## ✨ Conclusion

This project demonstrates a complete NLP pipeline—from preprocessing to evaluation—using an RNN for hate speech detection. While overall accuracy is high, improving detection of minority class (hate speech) remains the key next step for making the model more reliable in real-world scenarios.

---
