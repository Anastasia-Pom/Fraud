# Financial Fraud Detection: Custom Loss Function Optimization

## Overview
This project simulates a financial transaction environment to address a critical business problem: **optimizing the balance between catching fraud and avoiding false positives (blocking legitimate users).** 

Standard Machine Learning metrics like Accuracy or ROC-AUC often fail to capture the real-world financial impact of fraud detection. This project introduces a **Custom Business Loss Function** that penalizes the model based on actual financial losses:
*   **False Negative (Missed Fraud):** High cost (e.g., refunding the stolen money).
*   **False Positive (Blocked Legitimate User):** Low cost (e.g., customer support and compensation).

The script compares three approaches: a random baseline, a primitive rule-based algorithm, and a Machine Learning model with optimized thresholding.

## Key Features
*   **Synthetic Data Generation:** Simulates transaction amounts, times, and international flags to create a realistic fraud distribution.
*   **Custom Loss Function:** Implements a cost-sensitive metric (`COST_FN = $500`, `COST_FP = $20`) instead of traditional accuracy.
*   **Algorithm Comparison:**
    1.  *Random Trigger:* Baseline benchmark (blocking 5% of transactions randomly).
    2.  *Primitive Rules:* A simple "if-else" logic (e.g., high amount + nighttime).
    3.  *Machine Learning:* Decision Tree classifier with dynamic threshold optimization to minimize total financial loss.
*   **Business-Oriented Conclusion:** Demonstrates how choosing the optimal probability threshold saves thousands of dollars compared to random or default approaches.

## Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn
