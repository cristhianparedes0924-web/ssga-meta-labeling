# SSGA Meta-Labeling for Tactical Asset Allocation

This repository contains the implementation of a **Meta-Labeling** framework for Tactical Asset Allocation (TAA), developed for the Brandeis Field Project with State Street Global Advisors (SSGA). 

The core objective of this project is to implement the "modeling the model" concept—using a secondary Machine Learning layer to filter out "false positive" signals from a primary trend-following strategy to improve risk-adjusted returns.

## 🧠 Project Architecture
The system operates as a two-stage decision engine:

1.  **Primary Model (The Strategy):** A 1-month price momentum signal on the S&P 500. It issues a "Buy" signal if the previous month's return is positive.
2.  **Secondary Model (The Filter):** A **Random Forest Classifier** trained on historical market features. This model predicts the *probability* that the primary signal will be successful.

## 🛠️ Key Features
* **Meta-Labeling Logic:** Rather than predicting market direction, the model learns to predict the **validity** of the primary strategy's signal.
* **Feature Engineering:** Includes cross-asset signals such as 10Y Treasury Carry, S&P 500 Volatility (Z-scores), and fundamental Value indicators.
* **Backtesting Engine:** A custom framework that calculates cumulative returns, maximum drawdowns, and risk-adjusted metrics for both Naive and Optimized strategies.
* **Sensitivity Analysis:** Tools to calibrate the **Confidence Threshold** (e.g., finding the 0.40–0.44 "Goldilocks" zone) to balance profit against risk protection.

## 📊 Summary of Findings
* **Risk Mitigation:** The secondary model successfully identified high-volatility regimes (like 2008 and early 2020) where the primary momentum signal was likely to fail.
* **Performance:** In the out-of-sample test period (2020–2025), the Optimized strategy achieved a final value of **$1.95**, matching the Naive strategy while providing a superior framework for dynamic risk management.
* **Feature Importance:** **Volatility (`Z5_Vol`)** was identified as the most critical predictor of strategy failure, allowing the model to act as a "Risk-Off" switch during market panics.

## 🔄 Project Workflow
To understand the pipeline, the project follows these steps:
1.  **Data Ingestion:** Loading and merging multi-asset indices (S&P 500, BCOM, Treasury, IG Corp).
2.  **Feature Generation:** Calculating Z-scores for Momentum, Carry, Value, and Volatility to use as "features" for the secondary model.
3.  **Meta-Labeling:** Creating "binary labels" (1 if the primary signal was profitable, 0 if it failed) to train the Random Forest.
4.  **Optimization:** Running a backtest that only executes trades when the Random Forest confidence exceeds a specific threshold (e.g., 0.44).

## 💻 Usage
To run the analysis and reproduce the results:
1.  **Data Preparation:** Ensure the raw Excel files are placed in the `data/` directory.
2.  **Model Training:** Open `meta_labeling_final.ipynb` and run the "Model Training" cells to fit the Random Forest on the historical training period.
3.  **Backtesting:** Execute the "Backtest Engine" cells to compare the Optimized strategy against the Naive strategy.
4.  **Sensitivity Check:** Use the "Threshold Loop" cell to see how different confidence levels (0.40–0.50) impact your final return and drawdown.

## 📋 Collaboration Rules
* Do not commit directly to `main`.
* Create a personal branch for your work.
* Always pull the latest changes before starting.
* Commit frequently with clear messages.
* Open a Pull Request to merge into `main`.

## 🚀 Setup & Installation

### 1. Clone the repository
git clone git@github.com:cristhianparedes0924-web/ssga-meta-labeling.git
cd ssga-meta-labeling

### 2. Install required packages
pip install -r requirements.txt

### 3. Launch the Analysis
Launch the main Jupyter Notebook to view the training pipeline and backtest results:
jupyter lab meta_labeling_final.ipynb
