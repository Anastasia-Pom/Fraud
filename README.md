# Credit Card Fraud Detection

## 1. Dataset
- **Source:** [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **File used:** `creditcard_2023.csv` (~150 MB), uploaded manually into Colab.
- **Description:** Anonymised credit card transactions by European cardholders
  in 2023. Contains 28 PCA-transformed features (V1–V28), plus `Amount` and a
  binary `Class` label (1 = fraud, 0 = legitimate).
- **Balance:** Synthetically balanced (~50 % fraud, ~50 % legitimate), so no
  resampling is required.

## 2. Objective
Build and evaluate machine-learning models to detect fraudulent transactions,
following the workflow of Lab 8: data loading → EDA → preprocessing → model
training → evaluation.

## 3. Methodology

### 3.1 Data loading
The CSV is uploaded directly into Colab via `google.colab.files.upload()`,
the same approach used in Lab 8. This avoids external dependencies
(`opendatasets`, `kagglehub`) and works on Colab's Python 3.13 runtime.

### 3.2 Exploratory Data Analysis (EDA)
- Checked dataset shape, data types, and missing values.
- Visualised class distribution (count plot) to confirm balance.
- Plotted a correlation heatmap and a transaction-amount histogram.

### 3.3 Preprocessing
- Removed the `id` column (not predictive).
- Scaled `Amount` with `StandardScaler` (V1–V28 are already standardised).
- 80 / 20 stratified train–test split.

### 3.4 Models
- **Logistic Regression** – interpretable baseline.
- **Decision Tree** – non-linear, easy to visualise.
- **Random Forest** – ensemble, strong on tabular data.

### 3.5 Evaluation
- Classification report (precision, recall, F1).
- Confusion matrix (false positives vs. false negatives).
- ROC AUC (ranking quality).

## 4. Results
The models were evaluated on a test set of 113,726 transactions. The obtained metrics are as follows:

| Model                | Accuracy | ROC AUC |
|----------------------|----------|---------|
| Logistic Regression  | 0.9649   | 0.9935  |
| Decision Tree        | 0.9978   | 0.9978  |
| Random Forest        | 0.9998   | 1.0000  |

**Random Forest** achieves the best performance, with near-perfect accuracy and a perfect ROC AUC of 1.0000.

### 4.1 Feature Importance (Random Forest)
The top 15 most important features are visualised in the bar chart. The five most discriminative features are:
1. **V10** (highest importance, ~0.155)
2. **V14** (~0.150)
3. **V4** (~0.140)
4. **V12** (~0.110)
5. **V11** (~0.085)

These PCA components capture the underlying patterns that most strongly separate fraudulent transactions from legitimate ones.

### 4.2 Detailed Analysis of Confusion Matrices
- **Logistic Regression:** 55,591 True Negatives (Legit), 1,272 False Positives, 2,715 False Negatives (missed frauds), 54,148 True Positives. It performs well but misses a significant number of fraud cases.
- **Decision Tree:** 56,683 True Negatives, 180 False Positives, 73 False Negatives, 56,790 True Positives. A massive improvement over the baseline, with very few missed frauds.
- **Random Forest:** 56,844 True Negatives, 19 False Positives, **0 False Negatives**, 56,863 True Positives. This is the ideal outcome for fraud detection – the model caught every single fraudulent transaction in the test set while maintaining an extremely low false-alarm rate.

## 5. Why these choices?
- **Manual upload** — matches the Lab 8 workflow and avoids Python-version
  issues (`cgi` removed in Python 3.13, which broke `opendatasets`).
- **Logistic Regression** as a fast, interpretable baseline.
- **Random Forest** for robustness to noise and non-linear relationships.
- **Stratified split** preserves the 50/50 class ratio in both sets.

## 6. Conclusion
The 2023 dataset is balanced and clean, making it ideal for benchmarking
fraud-detection models. The Random Forest classifier delivers outstanding
performance, achieving 0 false negatives and an ROC AUC of 1.0000. This
demonstrates that ensemble tree-based methods are highly effective for this
type of tabular data. However, in real-world imbalanced settings, additional
techniques (SMOTE, cost-sensitive learning) would be necessary to achieve
similar results without overfitting.

## 7. How to run
1. Download the dataset from Kaggle (`creditcard_2023.csv` or `archive.zip`).
2. Open the notebook in Google Colab.
3. Run the first cell — a file-picker dialog will appear; select the CSV (or
   ZIP and unzip it).
4. Run the remaining cells — the notebook trains the models and prints the
   metrics automatically.
