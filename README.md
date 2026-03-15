# Bilingual Fake Review Detection with Explainable AI (XAI)

A machine learning and deep learning framework for detecting **fake reviews in bilingual text data** using traditional ML models, neural networks, and transformer-based language models. The system integrates **Explainable AI using SHAP** to interpret model predictions and understand feature contributions.

---

## 📌 Project Overview

Online platforms are increasingly affected by **fake or deceptive reviews**, which mislead users and damage trust. This project focuses on detecting such reviews in **bilingual textual data** using multiple modeling approaches and interpreting predictions with **Explainable AI (XAI)** techniques.

The system evaluates three categories of models:

* **Traditional Machine Learning Models**
* **Deep Neural Network Models**
* **Transformer-based Language Models**

To improve transparency and interpretability, **SHAP (SHapley Additive exPlanations)** is used to explain predictions made by the models.

---

## 🚀 Key Features

* Bilingual Fake Review Detection
* Comparison of **ML, Deep Learning, and Transformer models**
* Model interpretability using **SHAP**
* Ensemble learning for improved performance
* Modular training pipeline
* Explainable prediction outputs

---

## 🧠 Models Used

### Traditional Machine Learning Models

* Logistic Regression
* Linear SVM
* Random Forest
* XGBoost
* Voting Ensemble Classifier

These models rely on **vectorized textual features** such as TF-IDF.

---

### Deep Learning Models

* LSTM
* BiLSTM
* CNN-BiLSTM Hybrid

These models capture **contextual and sequential patterns** in text.

---

### Transformer Models

* mBERT (Multilingual BERT)
* XLM-RoBERTa

These pretrained transformer architectures are fine-tuned for the **fake review classification task**.

---

## 🧾 Explainable AI (XAI)

To interpret model predictions, **SHAP (SHapley Additive Explanations)** is used.

SHAP helps to:

* Identify **important words contributing to predictions**
* Understand **why a review is classified as fake or genuine**
* Provide **model transparency**

Example insights provided by SHAP:

* Feature importance plots
* Token contribution visualizations
* Local explanation for individual predictions

---

## 📂 Project Structure

```
FRD-XAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   ├── evaluation.ipynb
│
├── models/
│   ├── traditional/
│   ├── deep_learning/
│   ├── transformers/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_ml_models.py
│   ├── train_dl_models.py
│   ├── train_transformers.py
│   ├── explainability.py
│
├── results/
│   ├── metrics/
│   ├── shap_plots/
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains **bilingual textual reviews** labeled as:

* **Fake Review**
* **Genuine Review**

Typical dataset fields include:

| Column      | Description         |
| ----------- | ------------------- |
| review_text | Review content      |
| label       | Fake or Genuine     |
| language    | Language identifier |

The dataset undergoes the following preprocessing steps:

* Text cleaning
* Tokenization
* Stopword removal
* Vectorization (TF-IDF for ML models)
* Token encoding (for transformers)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/frd-xai.git
cd frd-xai
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

### Train Traditional ML Models

```bash
python src/train_ml_models.py
```

### Train Deep Learning Models

```bash
python src/train_dl_models.py
```

### Train Transformer Models

```bash
python src/train_transformers.py
```

---

## 🔍 Model Explainability with SHAP

To generate SHAP explanations:

```bash
python src/explainability.py
```

This will generate:

* SHAP summary plots
* Feature importance visualizations
* Individual prediction explanations

---

## 📈 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example evaluation output:

```
Accuracy  : 0.91
Precision : 0.89
Recall    : 0.90
F1 Score  : 0.89
```

---

## 🧪 Experimental Comparison

The project compares:

| Model Type    | Models                                                  |
| ------------- | ------------------------------------------------------- |
| ML            | Logistic Regression, Linear SVM, Random Forest, XGBoost |
| Ensemble      | Voting Classifier                                       |
| Deep Learning | LSTM, BiLSTM, CNN-BiLSTM                                |
| Transformers  | mBERT, XLM-RoBERTa                                      |

---

## 📌 Applications

* E-commerce platforms
* Online review platforms
* Social media monitoring
* Trust and credibility analysis

---

## 🔮 Future Improvements

* Larger multilingual datasets
* Attention visualization
* Deployment via API
* Real-time fake review detection system

---

## 👨‍💻 Author

**Rojin Dhami**

AI/ML Enthusiast working on Natural Language Processing and Explainable AI.

---

## 📜 License

This project is licensed under the **MIT License**.
