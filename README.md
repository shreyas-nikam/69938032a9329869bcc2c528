This is a comprehensive `README.md` file for your Streamlit application lab project, designed for developers and users, with proper markdown formatting.

---

# QuLab: Lab 49: ML-Enhanced Factor Model

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Description

This Streamlit application, "QuLab: Lab 49: ML-Enhanced Factor Model," guides quantitative investment professionals, particularly CFA Charterholders, through a robust workflow to compare traditional linear factor models (Ordinary Least Squares - OLS) with machine learning models (XGBoost) for predicting stock returns in a long-short equity strategy.

The project addresses the critical question of whether more complex ML models can uncover non-linear alpha signals that traditional linear approaches might miss, leading to enhanced portfolio performance. It simulates a comprehensive backtesting environment, analyzes performance using key financial metrics, interprets ML model behavior with SHAP, and assesses deployment readiness from a practical governance perspective.

Our goal is to explore a more adaptive, data-driven approach to portfolio management, potentially enhancing returns and mitigating drawdowns by leveraging advanced analytics.

### Learning Outcomes:

*   Implement OLS and XGBoost models for cross-sectional return prediction.
*   Execute a robust walk-forward backtest with monthly rebalancing.
*   Form long-short quintile portfolios based on model predictions.
*   Compare model performance using Sharpe ratio, maximum drawdown, return, volatility, hit rate, and turnover.
*   Conduct SHAP analysis to interpret the ML model's feature importance.
*   Assess the governance implications of deploying a more complex ML model based on its performance lift.

---

## Features

The application provides a guided, interactive experience through several key stages of a quantitative investment research workflow:

1.  **Interactive Data Generation**:
    *   Generate synthetic panel data for a configurable number of stocks and months.
    *   Embeds both linear and pre-defined non-linear alpha signals to create a controlled testing environment.
    *   Adjust parameters like `Number of Stocks` and `Number of Months`.

2.  **Walk-Forward Backtesting**:
    *   Execute a robust walk-forward backtest for both OLS and XGBoost models.
    *   Simulate a realistic trading environment, preventing look-ahead bias.
    *   Configurable `Minimum Training Months` and `Rebalance Frequency`.

3.  **Long-Short Portfolio Construction**:
    *   Form long-short quintile portfolios based on model predictions (long top 20%, short bottom 20%).
    *   Calculates long-short returns for each rebalancing period.

4.  **Comprehensive Performance Analysis**:
    *   Calculate key financial metrics: Annualized Return, Annualized Volatility, Sharpe Ratio, Maximum Drawdown, Hit Rate, and Total Return.
    *   Visual comparison of cumulative returns between OLS and XGBoost strategies.

5.  **ML Model Interpretability (SHAP)**:
    *   Utilize SHAP (SHapley Additive exPlanations) to interpret the XGBoost model's feature importance.
    *   Provides insights into which features drive the model's predictions.

6.  **Deployment Governance Assessment**:
    *   A structured framework to assess the justification for deploying the more complex ML model based on its "Sharpe Lift" (the performance difference relative to the linear model).
    *   Provides a clear verdict (Substantial, Moderate, or Marginal Alpha Lift) and associated governance recommendations.

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have the following installed:

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/quolab-lab49-ml-factor-model.git
    cd quolab-lab49-ml-factor-model
    ```
    *(Note: Replace `your-username` with the actual repository owner's username.)*

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *(The `requirements.txt` file should contain the following based on the application code):*
    ```
    streamlit>=1.0.0
    numpy>=1.20.0
    pandas>=1.3.0
    matplotlib>=3.4.0
    shap>=0.40.0
    xgboost>=1.6.0
    scikit-learn>=1.0.0 # Likely used in source.py for OLS or utilities
    statsmodels>=0.13.0 # Potentially used in source.py for OLS
    ```

---

## Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  Your web browser should automatically open to the application (usually `http://localhost:8501`).

### Basic Usage Instructions:

*   **Navigate** through the sections using the sidebar on the left.
*   **1. Data Generation**: Use the sliders to configure the number of stocks and months, then click "Generate Synthetic Data".
*   **2. Backtesting & Portfolio Construction**: Once data is generated, adjust backtest parameters and click "Run Walk-Forward Backtest". This will run both OLS and XGBoost models.
*   **3. Performance Analysis**: View the strategy comparison table and cumulative returns plot to analyze performance metrics.
*   **4. Interpretability & Governance**: Examine the SHAP feature importance for the ML model and review the governance assessment based on the Sharpe Lift.

---

## Project Structure

```
.
├── app.py                     # Main Streamlit application file
├── source.py                  # Contains core logic: data generation, prediction models, backtesting, metric calculation
├── requirements.txt           # List of Python dependencies
├── README.md                  # This README file
└── assets/                    # Directory for static assets (e.g., logos)
    └── logo5.jpg              # QuantUniversity logo
```

---

## Technology Stack

*   **Frontend/UI**: [Streamlit](https://streamlit.io/)
*   **Backend/Logic**: [Python 3.x](https://www.python.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Machine Learning**: [XGBoost](https://xgboost.ai/)
*   **Interpretability**: [SHAP](https://shap.readthedocs.io/en/latest/)
*   **Plotting**: [Matplotlib](https://matplotlib.org/)
*   **Statistical Models**: (Likely `scikit-learn` for OLS or `statsmodels` in `source.py`)

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and ensure the code passes any tests.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Contact

For questions or inquiries, please refer to [QuantUniversity](https://www.quantuniversity.com/) for more information on similar educational projects and labs.