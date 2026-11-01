# IEEE-CIS Fraud Detection (Kaggle Competition) Report

## Project Overview
This repository contains a comprehensive Machine Learning pipeline for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) competition hosted on Kaggle. 

The goal of the competition is to predict the probability of a transaction being fraudulent (`isFraud`). The task is challenging due to:
*   **High Class Imbalance:** Fraudulent transactions represent only ~3.5% of the dataset.
*   **High Cardinality:** Many categorical features (e.g., `card1`, `addr1`) with thousands of unique values.
*   **Time-Series Nature:** Transaction data requires careful validation strategies to avoid data leakage.

## Dataset & Preprocessing
The dataset consists of two main tables merged by `TransactionID`:
1.  `transaction`: Contains transactional data (amount, product code, card details, etc.).
2.  `identity`: Contains device and network information.

### Key Data Challenge: Column Naming Mismatch
A known issue in this competition is that the identity columns are named differently in the training and testing sets:
*   **Train:** `id_01`, `id_02`, ..., `id_38` (with underscore)
*   **Test:** `id-01`, `id-02`, ..., `id-38` (with hyphen)

**Solution Implemented:** The pipeline automatically detects and renames the columns in the test set to match the training set using `test_id.rename(columns=lambda c: c.replace('id-', 'id_'))`. This prevents the silent dropping of 38 crucial identity features during the column alignment step.

## Feature Engineering
To maximize the predictive power of the model, the following features were engineered:
*   **Time-based Features:** Extracted `hour`, `day_of_week`, and `day` from `TransactionDT`.
*   **Client UID:** Constructed a unique identifier (`uid`) by combining `card1`, `addr1`, `P_emaildomain`, and the transaction `day`. This helps the model group transactions by the "profile" of a user.
*   **UID Aggregations:** Calculated mean, standard deviation, and count of transaction amounts and D-columns grouped by `uid`. *(Note: Calculated strictly on train to prevent data leakage)*.
*   **Frequency Encoding:** Applied to high-cardinality categorical columns (e.g., `card1`, `addr1`) to help the model identify rare, anomalous values.
*   **Categorical Handling:** Used `pd.Categorical` to align categorical codes between train and test natively supported by LightGBM.

## Modeling & Validation
*   **Model:** **LightGBM** (Gradient Boosting Decision Trees), chosen for its efficiency and native handling of categorical features.
*   **Validation Strategy:** **GroupKFold** (5 folds) using `card1` as the group. This prevents data leakage by ensuring that transactions from the same credit card do not appear in both the training and validation sets simultaneously.

## Results
The baseline model achieved the following results on the Kaggle Leaderboard:
*   **Public Score (ROC-AUC):** `0.933271`
*   **Private Score (ROC-AUC):** `0.899871`
  <img width="1493" height="113" alt="image" src="https://github.com/user-attachments/assets/d26a32eb-18d2-41e7-8051-1bf0facd3a6e" />


### Future Improvements (Roadmap)
To push the score beyond 0.94+ (Top 10% level), the following strategies are planned:
1.  **Post-processing (UID Magic):** If a specific `uid` is confirmed fraudulent in the training set, boost the prediction probability for the same `uid` in the test set.
2.  **Advanced Aggregations:** Add aggregations for time-deltas (e.g., time since the last transaction for a specific card).
3.  **Ensemble Modeling:** Blend LightGBM with **CatBoost** and **XGBoost** using a weighted average or a stacking meta-classifier.
