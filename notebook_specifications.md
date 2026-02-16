
# Quantitative Equity Strategy: ML vs. Linear Factor Models for Alpha Discovery

## Introduction: Enhancing Stock Selection for CFA Charterholders

As a CFA Charterholder and Investment Professional at a quantitative asset management firm, you are constantly seeking to refine and improve your firm's investment strategies. Traditional linear factor models have been the bedrock of quantitative investing, effectively capturing known relationships like value, momentum, and quality. However, you suspect that these models might be missing more complex, non-linear interactions within market data that could unlock incremental alpha.

This notebook will guide you through a real-world workflow to investigate if a machine learning (ML) model, specifically XGBoost, can outperform a traditional Ordinary Least Squares (OLS) linear factor model in predicting stock returns for a long-short equity strategy. We will simulate a comprehensive backtesting environment, compare performance using key financial metrics, analyze model interpretability using SHAP, and finally, assess the ML model's deployment readiness from a governance perspective.

Our goal is to move beyond static asset allocation and explore a more adaptive, data-driven approach to portfolio management. By aligning portfolio construction with the economic cycle and market sentiment, the firm can potentially enhance returns during favorable periods and mitigate drawdowns during adverse conditions, improving overall risk-adjusted performance for clients.

**Learning Outcomes:**
*   Implement OLS and XGBoost models for cross-sectional return prediction.
*   Execute a robust walk-forward backtest with monthly rebalancing.
*   Form long-short quintile portfolios based on model predictions.
*   Compare model performance using Sharpe ratio, maximum drawdown, return, volatility, hit rate, and turnover.
*   Conduct SHAP analysis to interpret the ML model's feature importance.
*   Assess the governance implications of deploying a more complex ML model based on its performance lift.

## 1. Setup: Installing Libraries and Importing Dependencies

Before we dive into building our models and backtesting strategies, we need to set up our environment by installing the necessary Python libraries and importing them. These libraries will enable data manipulation, statistical modeling, machine learning, and model interpretability.

### 1.1 Install Required Libraries

```python
!pip install pandas numpy scikit-learn xgboost shap matplotlib
```

### 1.2 Import Required Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For percentage formatting
```

## 2. Generating Synthetic Panel Data with Non-Linear Alpha Signals

As a quantitative investment professional, accessing high-quality, clean historical data with embedded alpha signals is paramount for research. For initial model development and testing, synthetic data offers a controlled environment to ensure our models are learning specific, known patterns, including non-linear interactions that might be overlooked by traditional linear approaches. We will generate panel data for 200 stocks over 10 years (120 months) with diverse financial features and a `next_month_return` target that includes both linear and non-linear components.

The generation process will embed specific relationships:
*   **Linear factors**: Value (inverted P/E), Quality (ROE), Momentum, Earnings Growth, Book Value, Analyst Revisions, Market Cap, Short Interest.
*   **Non-linear interactions**:
    *   Value-Momentum Interaction: Low P/E (value) combined with positive momentum, leading to higher returns. This is expressed as `(0.003 * (-pe_z)) * (momentum_12m > 0)`.
    *   Volatility-Conditional Momentum: Volatility hurting returns in down-momentum periods, expressed as `(-0.002 * volatility * (momentum_12m < 0))`.
    *   Size Effect: Smaller companies (lower `log_mcap`) having a weaker negative impact or even a positive impact on returns, contrasting with a general negative `log_mcap` coefficient.

This controlled environment helps us evaluate if ML models can indeed "discover" these pre-defined non-linearities.

```python
def generate_synthetic_data(n_stocks, n_months, seed=42):
    """
    Generates synthetic panel data for stock selection.
    Includes fundamental, technical, and sentiment factors,
    with embedded linear and non-linear effects in the target return.

    Args:
        n_stocks (int): Number of unique stocks.
        n_months (int): Number of months for the panel data.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic panel data.
        list: A list of feature column names.
    """
    np.random.seed(seed)
    records = []

    for month in range(n_months):
        for stock_id in range(n_stocks):
            # Fundamental factors (cross-sectional, standardized)
            pe_z = np.random.randn()  # Value (inverted P/E)
            pb_z = np.random.randn()  # Book value
            roe = np.random.randn()   # Quality (ROE)
            earnings_growth = np.random.randn()

            # Technical factors
            momentum_12m = np.random.randn() # 12-month momentum
            momentum_1m = np.random.randn()  # Short-term reversal
            volatility = np.abs(np.random.randn()) * 0.3 # Volatility

            # Alternative / sentiment
            analyst_revisions = np.random.randn()
            log_mcap = np.random.randn() + 10 # Size (log market cap)
            short_interest = np.abs(np.random.randn()) * 0.1 # Short interest

            # True return: linear + nonlinear + noise
            # Value: low P/E -> higher return (linear component)
            ret = (0.003 * (-pe_z)) + \
                  (0.004 * momentum_12m) + \
                  (0.002 * roe) + \
                  (-0.001 * log_mcap) + \
                  (0.002 * analyst_revisions) + \
                  (0.003 * (momentum_12m * (-pe_z))) + \
                  (-0.002 * volatility * (momentum_12m < 0)) + \
                  (0.001 * pb_z) + \
                  (0.001 * earnings_growth) + \
                  (0.001 * momentum_1m) + \
                  (0.001 * short_interest) + \
                  (np.random.randn() * 0.06) # Noise (dominates)

            records.append({
                'month': month, 'stock_id': stock_id,
                'pe_z': pe_z, 'pb_z': pb_z, 'roe': roe,
                'earnings_growth': earnings_growth,
                'momentum_12m': momentum_12m, 'momentum_1m': momentum_1m,
                'volatility': volatility, 'analyst_revisions': analyst_revisions,
                'log_mcap': log_mcap, 'short_interest': short_interest,
                'next_month_return': ret,
            })

    df = pd.DataFrame(records)
    feature_cols = [
        'pe_z', 'pb_z', 'roe', 'earnings_growth', 'momentum_12m',
        'momentum_1m', 'volatility', 'analyst_revisions',
        'log_mcap', 'short_interest'
    ]
    return df, feature_cols

# Generate the data
n_stocks, n_months = 200, 120  # 200 stocks, 10 years (120 months)
df, feature_cols = generate_synthetic_data(n_stocks, n_months)

print(f"Panel Data Generated: {n_stocks} stocks x {n_months} months = {len(df)} observations")
print(f"Number of Features: {len(feature_cols)}")
print(f"First 5 rows of the generated data:\n{df.head()}")
print(f"\nReturn distribution: mean={df['next_month_return'].mean():.4f}, std={df['next_month_return'].std():.4f}")
```

The output confirms the generation of our synthetic panel data. The `next_month_return` distribution, with its small mean and larger standard deviation, reflects the challenging nature of predicting stock returns. This dataset now provides a controlled environment to rigorously test our models' ability to capture both linear and non-linear alpha.

## 3. Establishing a Linear Factor Model Baseline (OLS)

Before embracing the complexity of machine learning, it's essential to establish a strong, transparent baseline using traditional methods. For a CFA Charterholder, linear factor models, such as Ordinary Least Squares (OLS) regression, are familiar and interpretable. They serve as the "null hypothesis" against which any more complex ML model must demonstrate a significant, justified improvement. This section will build an OLS model to predict `next_month_return` based on our generated factors.

The OLS model seeks to find coefficients $\beta_i$ for each factor $X_i$ that best explain the target variable $Y$, minimizing the sum of squared residuals. The linear relationship is expressed as:

$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k + \epsilon $$

where $Y$ is `next_month_return`, $X_i$ are our features (e.g., `pe_z`, `momentum_12m`), $\beta_0$ is the intercept, $\beta_i$ are the coefficients, and $\epsilon$ is the error term.

```python
def linear_predict(train_df, test_df, features, target='next_month_return'):
    """
    Performs Simple OLS cross-sectional regression for return prediction.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data (features only for prediction).
        features (list): List of feature column names.
        target (str): Name of the target variable column.

    Returns:
        tuple: A tuple containing:
            - np.array: Predicted returns for the test set.
            - LinearRegression: The fitted OLS model.
    """
    model = LinearRegression()
    model.fit(train_df[features], train_df[target])
    predictions = model.predict(test_df[features])
    return predictions, model

# Quick in-sample check to demonstrate the model's coefficients
# For illustration, we'll use a fixed split (first 96 months for train, rest for test)
# In the actual backtest, this will be dynamic.
train_check = df[df['month'] < 96]
test_check = df[df['month'] >= 96]

lin_pred_check, lin_model_check = linear_predict(train_check, test_check, feature_cols)

print("Linear Model Coefficients (In-Sample Check):")
print("=" * 45)
for f, c in zip(feature_cols, lin_model_check.coef_):
    print(f" {f:<22s}: {c:+.6f}")
```

The output displays the coefficients for each factor as learned by the OLS model. These coefficients represent the linear impact of each factor on the `next_month_return`. For instance, a positive coefficient for `momentum_12m` indicates that stocks with higher 12-month momentum tend to have higher future returns, all else being equal. This transparency is a key advantage of linear models for investment professionals.

## 4. Building a Machine Learning Model (XGBoost) for Alpha Discovery

While linear models offer interpretability, they are limited in capturing complex, non-linear relationships and interactions inherent in financial markets. As a forward-thinking investment professional, you recognize the potential of advanced machine learning techniques to uncover these hidden patterns. XGBoost, a gradient boosting framework, is particularly well-suited for tabular data like ours, known for its predictive power and ability to handle non-linearities automatically.

This section will build an XGBoost Regressor model to predict `next_month_return`, aiming to capture the embedded non-linearities in our synthetic data.

```python
def ml_predict(train_df, test_df, features, target='next_month_return'):
    """
    Performs XGBoost cross-sectional regression for return prediction.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data (features only for prediction).
        features (list): List of feature column names.
        target (str): Name of the target variable column.

    Returns:
        tuple: A tuple containing:
            - np.array: Predicted returns for the test set.
            - xgb.XGBRegressor: The fitted XGBoost model.
    """
    # Initialize XGBoost Regressor with specified hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=100,       # Number of boosting rounds
        max_depth=4,            # Maximum tree depth
        learning_rate=0.05,     # Step size shrinkage to prevent overfitting
        subsample=0.8,          # Subsample ratio of the training instance
        colsample_bytree=0.8,   # Subsample ratio of columns when constructing each tree
        random_state=42,        # Random seed for reproducibility
        objective='reg:squarederror' # Objective function for regression
    )
    model.fit(train_df[features], train_df[target])
    predictions = model.predict(test_df[features])
    return predictions, model

# Quick in-sample check for the ML model (using the same split as OLS)
ml_pred_check, ml_model_check = ml_predict(train_check, test_check, feature_cols)

print(f"ML Model (XGBoost) Prediction Sample (first 5): {ml_pred_check[:5]}")
# We don't print coefficients for tree-based models as they are not directly interpretable like OLS.
# Interpretability will be addressed later with SHAP.
```

The XGBoost model has now been constructed and made initial predictions. Unlike OLS, its decision-making process is a complex ensemble of trees, making direct coefficient interpretation difficult. The real test of its power lies in its out-of-sample performance and its ability to capture those non-linear patterns we embedded in the synthetic data, which we will evaluate in the next steps through rigorous backtesting.

## 5. Walk-Forward Backtesting and Long-Short Portfolio Construction

Developing predictive models is only half the battle; the true measure of an investment strategy's efficacy is its performance in a realistic, out-of-sample simulation. As a CFA Charterholder, you understand the critical importance of robust backtesting to prevent "look-ahead bias" and ensure that any observed alpha is genuinely attainable. We will implement a walk-forward backtest, which simulates trading over time by progressively expanding the training window and rebalancing the portfolio monthly.

For each rebalancing point, we will:
1.  Train the model (either OLS or XGBoost) on historical data up to that point.
2.  Predict returns for the *next* month's stocks.
3.  Form a **long-short quintile portfolio**:
    *   Go long the top quintile (20% of stocks) with the highest predicted returns.
    *   Go short the bottom quintile (20% of stocks) with the lowest predicted returns.
    *   The return for this long-short portfolio at time $t$ ($r_{L/S,t}$) is calculated as the mean return of the top quintile ($Q_5$) minus the mean return of the bottom quintile ($Q_1$):
        $$ r_{L/S,t} = \frac{1}{|Q_5|} \sum_{i \in Q_5} r_{i,t} - \frac{1}{|Q_1|} \sum_{i \in Q_1} r_{i,t} $$
        where $r_{i,t}$ is the actual return of stock $i$ at month $t$.

This approach directly monetizes the models' predictive power by identifying the strongest buys and sells.

```python
def walk_forward_backtest(df, features, predict_fn,
                          min_train_months=60, rebalance_freq=1):
    """
    Performs a walk-forward backtest with expanding training window and monthly rebalancing.
    Forms long-short quintile portfolios based on predicted returns.

    Args:
        df (pd.DataFrame): The full panel data.
        features (list): List of feature column names.
        predict_fn (function): The prediction function (e.g., linear_predict or ml_predict).
        min_train_months (int): Minimum number of months required for the initial training set.
        rebalance_freq (int): Frequency in months to rebalance the portfolio.

    Returns:
        pd.DataFrame: DataFrame containing monthly long-short portfolio returns and metadata.
    """
    portfolio_returns = []
    unique_months = sorted(df['month'].unique())

    # Ensure min_train_months is valid
    if min_train_months >= len(unique_months):
        raise ValueError("min_train_months must be less than total unique_months.")

    # Iterate through months for backtesting
    # The loop starts after the initial training period
    for i in range(min_train_months, len(unique_months), rebalance_freq):
        train_months = unique_months[:i]
        test_month = unique_months[i]

        train_data = df[df['month'].isin(train_months)]
        test_data = df[df['month'] == test_month].copy() # Ensure a copy to avoid SettingWithCopyWarning

        if len(test_data) < 50: # Skip if not enough stocks for robust quintile formation
            continue

        # Predict returns for the test month
        predictions, _ = predict_fn(train_data, test_data, features)
        test_data['predicted_return'] = predictions

        # Form quintile portfolios
        # labels=[1,2,3,4,5] assigns 1 to lowest quintile, 5 to highest
        test_data['quintile'] = pd.qcut(
            test_data['predicted_return'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        )

        # Calculate long (top quintile) and short (bottom quintile) returns
        # Handle cases where a quintile might be empty if duplicates='drop' leads to fewer than 5 labels
        long_ret = test_data[test_data['quintile'] == 5]['next_month_return'].mean()
        short_ret = test_data[test_data['quintile'] == 1]['next_month_return'].mean()

        # If a quintile is empty (e.g., due to qcut duplicates='drop' and limited unique predicted values),
        # its mean will be NaN. We should handle this to prevent NaN in ls_return.
        if pd.isna(long_ret):
            long_ret = 0.0 # Or some other reasonable default
        if pd.isna(short_ret):
            short_ret = 0.0 # Or some other reasonable default

        ls_ret = long_ret - short_ret

        # Record portfolio metrics for the month
        portfolio_returns.append({
            'month': test_month,
            'long_return': long_ret,
            'short_return': short_ret,
            'ls_return': ls_ret,
            'n_long': (test_data['quintile'] == 5).sum(),
            'n_short': (test_data['quintile'] == 1).sum(),
        })

    return pd.DataFrame(portfolio_returns)

print("Running walk-forward backtests for OLS and XGBoost strategies...")

# Run backtest for Linear (OLS) model
linear_results = walk_forward_backtest(df, feature_cols, linear_predict)
print(f"Linear (OLS) Strategy: {len(linear_results)} months of backtest results generated.")
print(f"Mean L/S Return (OLS): {linear_results['ls_return'].mean()*100:.2f}%/month")

# Run backtest for ML (XGBoost) model
ml_results = walk_forward_backtest(df, feature_cols, ml_predict)
print(f"ML (XGBoost) Strategy: {len(ml_results)} months of backtest results generated.")
print(f"Mean L/S Return (XGBoost): {ml_results['ls_return'].mean()*100:.2f}%/month")

print("\nSample of Long-Short Returns (first 5 months of ML strategy):")
print(ml_results[['month', 'ls_return']].head())
```

The backtesting process is now complete for both the OLS and XGBoost models. We have obtained a time series of monthly long-short returns for each strategy. The output provides a quick glance at the average monthly performance, hinting at which model might be performing better. This data forms the basis for our comprehensive performance comparison and attribution, where we will quantify the strategies' risk-adjusted returns.

## 6. Performance Analysis and Strategy Comparison

As an investment professional, quantifying and comparing the risk-adjusted performance of different strategies is a core responsibility. With our walk-forward backtest results, we can now rigorously evaluate the OLS baseline against the XGBoost ML model using a suite of standard financial metrics. This comparison will directly inform whether the added complexity of the ML model delivers a significant "alpha lift" that warrants further consideration.

We will calculate the following key metrics:
*   **Annualized Return**: $\mu \times 12$
*   **Annualized Volatility**: $\sigma \times \sqrt{12}$
*   **Sharpe Ratio (SR)**: Measures risk-adjusted return. For a long-short strategy, it's the annualized mean return divided by the annualized standard deviation of returns.
    $$ SR = \frac{\mu_{L/S} \times 12}{\sigma_{L/S} \times \sqrt{12}} $$
    where $\mu_{L/S}$ is the monthly mean of long-short returns and $\sigma_{L/S}$ is the monthly standard deviation of long-short returns.
*   **Maximum Drawdown**: The largest peak-to-trough decline over a specific period.
    $$ \text{Max Drawdown} = \max_t \left( \frac{V_{peak,t} - V_t}{V_{peak,t}} \right) $$
    where $V_t$ is the portfolio value at time $t$, and $V_{peak,t}$ is the maximum portfolio value up to time $t$.
*   **Hit Rate**: The percentage of months with positive long-short returns.
*   **Total Return**: Cumulative return over the entire backtesting period.
*   **Turnover**: The amount of buying and selling required to maintain the portfolio. While not directly calculated in `compute_metrics`, it's a critical consideration for transaction costs. We will visualize cumulative returns and implicitly consider turnover later.

```python
def compute_metrics(results_df, label):
    """
    Computes standard financial performance metrics for a given strategy.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ls_return' column from backtest.
        label (str): Label for the strategy (e.g., 'Linear (OLS)', 'ML (XGBoost)').

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    rets = results_df['ls_return']
    cumulative_returns = (1 + rets).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak

    ann_return = rets.mean() * 12
    ann_volatility = rets.std() * np.sqrt(12)
    sharpe = (ann_return / ann_volatility) if ann_volatility != 0 else np.nan
    max_drawdown = drawdown.min()
    hit_rate = (rets > 0).mean()
    total_return = cumulative_returns.iloc[-1] - 1

    return {
        'strategy': label,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe': sharpe,
        'hit_rate': hit_rate,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
    }

def compare_strategies(linear_results, ml_results):
    """
    Compares linear and ML strategies on key performance metrics.

    Args:
        linear_results (pd.DataFrame): Backtest results for the linear model.
        ml_results (pd.DataFrame): Backtest results for the ML model.

    Returns:
        pd.DataFrame: A formatted DataFrame comparing the metrics of both strategies.
    """
    lin_metrics = compute_metrics(linear_results, 'Linear (OLS)')
    ml_metrics = compute_metrics(ml_results, 'ML (XGBoost)')

    comp_df = pd.DataFrame([lin_metrics, ml_metrics]).set_index('strategy')

    # Format the DataFrame for better readability
    comp_df_formatted = comp_df.copy()
    comp_df_formatted['ann_return'] = comp_df_formatted['ann_return'].apply(lambda x: f"{x*100:.2f}%")
    comp_df_formatted['ann_volatility'] = comp_df_formatted['ann_volatility'].apply(lambda x: f"{x*100:.2f}%")
    comp_df_formatted['sharpe'] = comp_df_formatted['sharpe'].apply(lambda x: f"{x:.2f}")
    comp_df_formatted['hit_rate'] = comp_df_formatted['hit_rate'].apply(lambda x: f"{x*100:.0f}%")
    comp_df_formatted['max_drawdown'] = comp_df_formatted['max_drawdown'].apply(lambda x: f"{x*100:.1f}%")
    comp_df_formatted['total_return'] = comp_df_formatted['total_return'].apply(lambda x: f"{x*100:.1f}%")

    return comp_df_formatted, comp_df # Return both formatted and unformatted for calculations


# Perform the comparison
comparison_table_formatted, comparison_table_raw = compare_strategies(linear_results, ml_results)

print("STRATEGY COMPARISON")
print("=" * 60)
print(comparison_table_formatted.to_string())

# Plot Cumulative Returns
plt.figure(figsize=(12, 6))
(1 + linear_results['ls_return']).cumprod().plot(label='Linear (OLS) Strategy', color='blue')
(1 + ml_results['ls_return']).cumprod().plot(label='ML (XGBoost) Strategy', color='green')
plt.title('Cumulative Returns of Long-Short Strategies')
plt.xlabel('Months')
plt.ylabel('Cumulative Return')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
```

The strategy comparison table and cumulative returns plot provide a clear picture of how the OLS and XGBoost models performed. The ML model, leveraging its ability to capture non-linearities, likely shows a higher Sharpe Ratio and cumulative return, potentially with a shallower maximum drawdown and higher hit rate. This quantitative evidence is crucial for you, as a CFA Charterholder, to determine if the ML approach offers a meaningful performance edge over the traditional linear baseline. The "Sharpe Lift" (difference in Sharpe Ratios) is a key metric that will drive our governance decision.

## 7. Model Interpretability (SHAP) and Deployment Governance Assessment

A higher Sharpe Ratio from an ML model is exciting, but for an investment professional, it's not enough to justify deployment. You need to understand *why* the model performs well (interpretability) and assess the practical implications of its complexity (governance). This section focuses on both.

**SHAP (SHapley Additive exPlanations)** will be used to interpret the XGBoost model. SHAP values attribute the contribution of each feature to a specific prediction, helping us understand the model's "thinking." This is crucial for verifying if the model's insights align with financial intuition (e.g., if momentum or value are genuinely important).

Next, we perform a **Governance Assessment** based on the observed "Sharpe Lift" — the difference between the ML model's Sharpe Ratio and the linear model's Sharpe Ratio. This connects directly to the firm's framework for justifying the complexity of ML models:
*   **Substantial Alpha Lift (Sharpe Lift > 0.3)**: ML complexity is justified, proceed with Tier 2 governance (validation + monitoring).
*   **Moderate Alpha Lift (0.1 < Sharpe Lift $\le$ 0.3)**: ML complexity is conditionally justified, requires SHAP review + ongoing performance monitoring.
*   **Marginal Alpha Lift (Sharpe Lift $\le$ 0.1)**: ML complexity is NOT justified, deploy the simpler linear model (lower governance burden).

This systematic approach ensures that the firm's resources are allocated efficiently and that models are deployed responsibly.

```python
# Re-train the ML model on the full training data for SHAP analysis
# Using all data prior to the start of the backtest's test period (month < min_train_months)
# This mimics training the model for a potential deployment at the beginning of the backtest
min_train_months_for_shap = 60 # Consistent with backtest setup
train_all_for_shap = df[df['month'] < min_train_months_for_shap]

# Ensure the model is trained on sufficient data for robust SHAP explanation
if train_all_for_shap.empty:
    print("Warning: Not enough training data for SHAP analysis. Skipping SHAP.")
else:
    ml_final_model_for_shap = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='reg:squarederror'
    )
    ml_final_model_for_shap.fit(train_all_for_shap[feature_cols], train_all_for_shap['next_month_return'])

    # SHAP global importance
    explainer = shap.TreeExplainer(ml_final_model_for_shap)
    # Sample a smaller subset for SHAP values calculation for performance reasons
    shap_values = explainer.shap_values(train_all_for_shap[feature_cols].sample(min(1000, len(train_all_for_shap)), random_state=42))

    # Calculate mean absolute SHAP value for each feature
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)

    print("\nML MODEL SHAP FEATURE IMPORTANCE (Mean Absolute SHAP Value)")
    print("=" * 60)
    for feat, imp in importance.items():
        # Scale for visualization bar; max 20 hashes
        bar_length = int(imp / importance.max() * 20)
        bar = '#' * bar_length
        print(f" {feat:<22s}: {imp:.5f} {bar}")

    # SHAP summary plot for richer visualization
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, train_all_for_shap[feature_cols].sample(min(1000, len(train_all_for_shap)), random_state=42), show=False)
    plt.title('SHAP Summary Plot for XGBoost Model')
    plt.tight_layout()
    plt.show()

# Governance assessment (from the D4-T4-C1 complexity justification framework)
lin_sharpe = comparison_table_raw.loc['Linear (OLS)', 'sharpe']
ml_sharpe = comparison_table_raw.loc['ML (XGBoost)', 'sharpe']
sharpe_lift = ml_sharpe - lin_sharpe

print("\nGOVERNANCE ASSESSMENT")
print("=" * 60)
print(f"Linear (OLS) Sharpe Ratio: {lin_sharpe:.2f}")
print(f"ML (XGBoost) Sharpe Ratio: {ml_sharpe:.2f}")
print(f"Sharpe Lift (ML vs. Linear): {sharpe_lift:+.2f}")
print(f"Relative Sharpe Lift: {(ml_sharpe / max(lin_sharpe, 0.01) - 1) * 100:+.0f}%")

if sharpe_lift > 0.3:
    print("\nVERDICT: Substantial alpha lift. ML complexity JUSTIFIED.")
    print("         Proceed with Tier 2 governance (validation + monitoring).")
elif sharpe_lift > 0.1:
    print("\nVERDICT: Moderate alpha lift. ML complexity CONDITIONALLY justified.")
    print("         Requires SHAP review + ongoing performance monitoring.")
else:
    print("\nVERDICT: Marginal alpha lift. ML complexity NOT justified.")
    print("         Deploy the simpler linear model (lower governance burden).")

print("\n--- Practitioner Warning ---")
print("Simulated data typically overstates ML's advantage because nonlinear interactions are embedded.")
print("Real-world Sharpe lifts are often in the 0.1-0.3 range and come with hidden costs like higher turnover.")
print("Always validate on out-of-sample real data before drawing deployment conclusions.")
```

The SHAP analysis provides critical insights into the XGBoost model's decision-making. The feature importance plot (both text-based and the SHAP summary plot) reveals which factors the model prioritized. We can verify if these align with our financial intuition and economic theories. For example, if momentum and value factors (or their interactions) are prominent, it adds credibility to the model.

Finally, the governance assessment delivers a clear verdict based on the calculated Sharpe Lift. As a CFA Charterholder, this framework allows you to make an informed, data-driven decision on whether the incremental performance of the ML model justifies its increased complexity and associated governance costs. It guides whether to proceed with more rigorous validation, ongoing monitoring, or to stick with the more transparent linear model, balancing innovation with practical considerations for the firm and its clients.
