# Store Sales - Time Series Forecasting (Kaggle)

This repository contains my solution for the [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. 

### Results
* **Public Leaderboard Score (RMSLE):** `0.41816`
* **Leaderboard Position:** **188** (Top 1.5% globally)

## Project Overview
The goal of this competition is to predict daily sales for thousands of product families across 54 stores of Corporación Favorita (a large Ecuadorian grocery retailer). The evaluation metric is the **Root Mean Squared Logarithmic Error (RMSLE)**. 

The dataset is highly complex: it contains heavy seasonality (weekly, yearly), a massive number of zero-sales days (intermittent demand), holiday anomalies, a major earthquake in 2016, and a 16-day forecasting horizon.

## The Winning Architecture
After extensive experimentation, the final solution is an ensemble of **LightGBM, XGBoost, and CatBoost**, combined via a recursive forecasting strategy. 

Here is exactly what makes this architecture work and why these methods were chosen:

### 1. Robust Recursive Forecasting
Instead of training a single "direct" model for the 16-day horizon, we use a **recursive approach**: we predict day 1, append that prediction to the history, and use it to predict day 2, and so on. 
* **Why:** Retail sales rely heavily on the most recent days. A recursive approach allows the model to use the *latest* lags and moving averages dynamically, rather than relying on stale data from 16 days ago. We implemented this with vectorized operations and `np.fromiter` to ensure high performance (avoiding slow `iterrows` loops).

### 2. Solving the "Stale Lags" & Data Leakage Problems
In time-series forecasting, test features can easily become "frozen" or misaligned, causing massive distribution shifts. We implemented three critical fixes:
* **Transaction Lag Shift:** Transaction data is unavailable for the test set. Instead of filling it with a constant median, we shifted the historical transaction data by exactly 16 days. This provides the model with *actual* transaction levels corresponding to each specific forecast day.
* **Historical Promo Prefix:** Rolling promo features (`promo_roll_7`) on the test set were corrupted because the test set only contains 16 days. We prepended the last 14 days of historical data to the test set before calculating the rolling windows, ensuring the feature distribution matches the training data perfectly.
* **Smoothed Target Encoding:** We calculate the mean sales by Day-of-Week strictly using historical data *prior* to the validation window. This prevents data leakage while giving the model a strong baseline for weekly seasonality.

### 3. Powerful Feature Engineering
The models are fed a mix of:
* **Lags & Rolling Means:** Lags (1, 7, 14, 21, 28, 364) capture short-term momentum and annual seasonality. (The 364-day lag is chosen over 365 to align the day of the week).
* **Calendar & Fourier Transforms:** To capture the complex cyclical nature of retail, we added Fourier Sine/Cosine pairs for both weekly and yearly cycles.
* **Holiday Anomalies:** Features indicating paydays (15th and month-end), national holidays, and the "earthquake recovery" period.
* **Oil Prices:** Ecuador's economy is heavily tied to oil, making it a crucial macroeconomic indicator.

### 4. Log-Space Ensembling
The target variable is transformed using `log1p(sales)` before training, which directly optimizes for the RMSLE metric and stabilizes the variance.
* We train **LightGBM**, **XGBoost**, and **CatBoost** simultaneously. 
* The predictions of the three models are blended in the **log-space** using optimized weights (e.g., `0.5*LGB + 0.3*CB + 0.2*XGB`). 
* **Why:** Tree-based models handle categorical features and missing data well, but each uses a different boosting algorithm (Leaf-wise vs. Level-wise vs. Ordered). Blending them in log-space cancels out individual model biases, resulting in a highly stable and generalized prediction.


## Rejected Experiments (What did NOT work)
To achieve this result, several popular techniques were tested and discarded due to poor performance on this specific dataset:
* **Tweedie Loss:** Performed significantly worse than `log1p` + RMSE due to the extreme sparsity of sales.
* **Direct Multi-Horizon (16 separate models):** While it avoids error accumulation, it fails to capture short-term momentum and performed worse (LB ~0.45).
* **Two-Stage Models (Classifier + Regressor):** Overcomplicating the prediction of zeros resulted in higher variance and worse RMSLE.
* **Heavy Damping/Blending in Recursion:** Mixing predictions with historical means (e.g., 0.7*Pred + 0.3*Mean) suppressed vital weekend/holiday spikes, artificially lowering the overall prediction mean.

## How to Run
1. Clone the repository.
2. Download the dataset from Kaggle and place it in the `../input/store-sales-time-series-forecasting/` directory.
3. Install the dependencies: `pip install pandas numpy scikit-learn lightgbm xgboost catboost tqdm`.
4. Run `solution.py`. The final submission will be generated as `submission_optimized.csv`.
