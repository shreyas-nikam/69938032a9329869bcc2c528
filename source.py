import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For percentage formatting

# --- Data Generation ---
def generate_synthetic_data(n_stocks: int, n_months: int, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
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

# --- Prediction Models ---
def linear_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str = 'next_month_return') -> tuple[np.ndarray, LinearRegression]:
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

def ml_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str = 'next_month_return') -> tuple[np.ndarray, xgb.XGBRegressor]:
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

# --- Backtesting Framework ---
def walk_forward_backtest(df: pd.DataFrame, features: list[str], predict_fn,
                          min_train_months: int = 60, rebalance_freq: int = 1) -> pd.DataFrame:
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
        print(f"Warning: min_train_months ({min_train_months}) is greater than or equal to total unique_months ({len(unique_months)}). No backtest will be performed.")
        return pd.DataFrame() # Return empty DataFrame if no backtest can be performed

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
        if test_data['predicted_return'].nunique() < 5:
            # If not enough unique predicted values for 5 quintiles, qcut will drop labels.
            # We ensure we have at least 2 quintiles (long/short)
            num_quantiles = max(2, test_data['predicted_return'].nunique())
            labels = list(range(1, num_quantiles + 1))
            if num_quantiles > 0:
                test_data['quintile'] = pd.qcut(
                    test_data['predicted_return'], num_quantiles, labels=labels, duplicates='drop'
                )
                long_quintile_label = labels[-1] # Highest quintile
                short_quintile_label = labels[0] # Lowest quintile
            else:
                test_data['quintile'] = np.nan # No quantiles can be formed
                long_quintile_label = None
                short_quintile_label = None
        else:
            test_data['quintile'] = pd.qcut(
                test_data['predicted_return'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            )
            long_quintile_label = 5
            short_quintile_label = 1

        long_ret = test_data[test_data['quintile'] == long_quintile_label]['next_month_return'].mean() if long_quintile_label else np.nan
        short_ret = test_data[test_data['quintile'] == short_quintile_label]['next_month_return'].mean() if short_quintile_label else np.nan

        if pd.isna(long_ret):
            long_ret = 0.0
        if pd.isna(short_ret):
            short_ret = 0.0

        ls_ret = long_ret - short_ret

        portfolio_returns.append({
            'month': test_month,
            'long_return': long_ret,
            'short_return': short_ret,
            'ls_return': ls_ret,
            'n_long': (test_data['quintile'] == long_quintile_label).sum() if long_quintile_label else 0,
            'n_short': (test_data['quintile'] == short_quintile_label).sum() if short_quintile_label else 0,
        })

    return pd.DataFrame(portfolio_returns)

# --- Performance Metrics and Comparison ---
def compute_metrics(results_df: pd.DataFrame, label: str) -> dict:
    """
    Computes standard financial performance metrics for a given strategy.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ls_return' column from backtest.
        label (str): Label for the strategy (e.g., 'Linear (OLS)', 'ML (XGBoost)').

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    if results_df.empty:
        return {
            'strategy': label,
            'ann_return': np.nan,
            'ann_volatility': np.nan,
            'sharpe': np.nan,
            'hit_rate': np.nan,
            'max_drawdown': np.nan,
            'total_return': np.nan,
        }

    rets = results_df['ls_return']
    cumulative_returns = (1 + rets).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak

    ann_return = rets.mean() * 12
    ann_volatility = rets.std() * np.sqrt(12)
    sharpe = (ann_return / ann_volatility) if ann_volatility != 0 else np.nan
    max_drawdown = drawdown.min()
    hit_rate = (rets > 0).mean()
    total_return = cumulative_returns.iloc[-1] - 1 if not cumulative_returns.empty else 0.0

    return {
        'strategy': label,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe': sharpe,
        'hit_rate': hit_rate,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
    }

def compare_strategies(linear_results: pd.DataFrame, ml_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares linear and ML strategies on key performance metrics.

    Args:
        linear_results (pd.DataFrame): Backtest results for the linear model.
        ml_results (pd.DataFrame): Backtest results for the ML model.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A formatted DataFrame comparing the metrics of both strategies.
            - pd.DataFrame: A raw (unformatted) DataFrame comparing the metrics.
    """
    lin_metrics = compute_metrics(linear_results, 'Linear (OLS)')
    ml_metrics = compute_metrics(ml_results, 'ML (XGBoost)')

    comp_df = pd.DataFrame([lin_metrics, ml_metrics]).set_index('strategy')

    # Format the DataFrame for better readability
    comp_df_formatted = comp_df.copy()
    comp_df_formatted['ann_return'] = comp_df_formatted['ann_return'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    comp_df_formatted['ann_volatility'] = comp_df_formatted['ann_volatility'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    comp_df_formatted['sharpe'] = comp_df_formatted['sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    comp_df_formatted['hit_rate'] = comp_df_formatted['hit_rate'].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "N/A")
    comp_df_formatted['max_drawdown'] = comp_df_formatted['max_drawdown'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    comp_df_formatted['total_return'] = comp_df_formatted['total_return'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")

    return comp_df_formatted, comp_df # Return both formatted and unformatted for calculations

# --- SHAP Analysis ---
def perform_shap_analysis(df: pd.DataFrame, feature_cols: list[str],
                          min_train_months_for_shap: int = 60,
                          shap_sample_size: int = 1000) -> tuple[pd.Series | None, np.ndarray | None, pd.DataFrame | None]:
    """
    Trains an XGBoost model and performs SHAP analysis on it.

    Args:
        df (pd.DataFrame): The full panel data.
        feature_cols (list): List of feature column names.
        min_train_months_for_shap (int): Minimum number of months for the initial training set for SHAP.
        shap_sample_size (int): Number of samples to use for SHAP value calculation.

    Returns:
        tuple: (pd.Series) Feature importance based on mean absolute SHAP values,
               (np.ndarray) Raw SHAP values,
               (pd.DataFrame) Data used for SHAP calculation.
               Returns (None, None, None) if not enough training data.
    """
    train_all_for_shap = df[df['month'] < min_train_months_for_shap]

    if train_all_for_shap.empty:
        print("Warning: Not enough training data for SHAP analysis. Skipping SHAP.")
        return None, None, None
    
    # Adjust sample size if training data is smaller than requested sample size
    effective_shap_sample_size = min(shap_sample_size, len(train_all_for_shap))
    if effective_shap_sample_size == 0:
        print("Warning: No samples available for SHAP analysis after filtering.")
        return None, None, None

    shap_sample_data = train_all_for_shap[feature_cols].sample(effective_shap_sample_size, random_state=42)

    ml_final_model_for_shap = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='reg:squarederror'
    )
    ml_final_model_for_shap.fit(train_all_for_shap[feature_cols], train_all_for_shap['next_month_return'])

    explainer = shap.TreeExplainer(ml_final_model_for_shap)
    shap_values = explainer.shap_values(shap_sample_data)

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)

    return importance, shap_values, shap_sample_data

# --- Governance Assessment ---
def assess_governance(comparison_table_raw: pd.DataFrame) -> dict:
    """
    Assesses the justification for ML complexity based on Sharpe Ratio lift.

    Args:
        comparison_table_raw (pd.DataFrame): Raw (unformatted) comparison of strategy metrics.

    Returns:
        dict: A dictionary containing Sharpe ratios, lift, relative lift, and a verdict string.
    """
    lin_sharpe = comparison_table_raw.loc['Linear (OLS)', 'sharpe']
    ml_sharpe = comparison_table_raw.loc['ML (XGBoost)', 'sharpe']

    if pd.isna(lin_sharpe) or pd.isna(ml_sharpe):
        sharpe_lift = np.nan
        relative_sharpe_lift = np.nan
        verdict = "Not enough data for a meaningful Sharpe comparison."
    else:
        sharpe_lift = ml_sharpe - lin_sharpe
        # Avoid division by zero or very small numbers for relative lift
        relative_sharpe_lift = (ml_sharpe / max(lin_sharpe, 0.01) - 1) * 100 if lin_sharpe != 0 else np.inf

        if sharpe_lift > 0.3:
            verdict = "Substantial alpha lift. ML complexity JUSTIFIED. Proceed with Tier 2 governance (validation + monitoring)."
        elif sharpe_lift > 0.1:
            verdict = "Moderate alpha lift. ML complexity CONDITIONALLY justified. Requires SHAP review + ongoing performance monitoring."
        else:
            verdict = "Marginal alpha lift. ML complexity NOT justified. Deploy the simpler linear model (lower governance burden)."

    return {
        'lin_sharpe': lin_sharpe,
        'ml_sharpe': ml_sharpe,
        'sharpe_lift': sharpe_lift,
        'relative_sharpe_lift': relative_sharpe_lift,
        'verdict': verdict
    }

# --- Plotting Functions ---
def plot_cumulative_returns(linear_results: pd.DataFrame, ml_results: pd.DataFrame, title: str = 'Cumulative Returns of Long-Short Strategies'):
    """
    Plots the cumulative returns of linear and ML strategies.

    Args:
        linear_results (pd.DataFrame): Backtest results for the linear model.
        ml_results (pd.DataFrame): Backtest results for the ML model.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    if not linear_results.empty:
        (1 + linear_results['ls_return']).cumprod().plot(label='Linear (OLS) Strategy', color='blue')
    if not ml_results.empty:
        (1 + ml_results['ls_return']).cumprod().plot(label='ML (XGBoost) Strategy', color='green')
    plt.title(title)
    plt.xlabel('Months')
    plt.ylabel('Cumulative Return')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_shap_summary(shap_values: np.ndarray, shap_sample_data: pd.DataFrame, title: str = 'SHAP Summary Plot for XGBoost Model'):
    """
    Plots the SHAP summary plot.

    Args:
        shap_values (np.ndarray): SHAP values generated by the explainer.
        shap_sample_data (pd.DataFrame): Data used for SHAP calculation.
        title (str): Title of the plot.
    """
    if shap_values is not None and shap_sample_data is not None and not shap_sample_data.empty:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, shap_sample_data, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        print("SHAP values or sample data are not available or are empty for plotting.")


# --- Main Analysis Function ---
def run_stock_selection_analysis(
    n_stocks: int = 200,
    n_months: int = 120,
    min_train_months: int = 60,
    rebalance_freq: int = 1,
    seed: int = 42,
    shap_sample_size: int = 1000
) -> dict:
    """
    Executes the full stock selection analysis pipeline, including data generation,
    backtesting, performance comparison, SHAP analysis, and governance assessment.

    Args:
        n_stocks (int): Number of unique stocks for synthetic data.
        n_months (int): Number of months for synthetic data.
        min_train_months (int): Minimum number of months for the initial training set
                                for walk-forward backtest and SHAP analysis.
        rebalance_freq (int): Frequency in months to rebalance the portfolio.
        seed (int): Random seed for reproducibility.
        shap_sample_size (int): Number of samples to use for SHAP value calculation.

    Returns:
        dict: A dictionary containing all results:
              - 'df': The generated synthetic DataFrame.
              - 'feature_cols': List of feature column names.
              - 'linear_results': DataFrame of linear strategy backtest results.
              - 'ml_results': DataFrame of ML strategy backtest results.
              - 'comparison_table_formatted': Formatted DataFrame of strategy metrics.
              - 'comparison_table_raw': Raw DataFrame of strategy metrics.
              - 'shap_importance': Series of SHAP feature importance.
              - 'shap_values': SHAP values for plotting.
              - 'shap_data_for_plot': Data used for SHAP plotting.
              - 'governance_assessment': Dictionary with governance verdict details.
    """
    # 1. Generate Data
    df, feature_cols = generate_synthetic_data(n_stocks, n_months, seed)

    # 2. Run Backtests
    print("\nRunning walk-forward backtests for OLS and XGBoost strategies...")
    linear_results = walk_forward_backtest(df, feature_cols, linear_predict, min_train_months, rebalance_freq)
    ml_results = walk_forward_backtest(df, feature_cols, ml_predict, min_train_months, rebalance_freq)

    # 3. Compare Strategies
    comparison_table_formatted, comparison_table_raw = compare_strategies(linear_results, ml_results)

    # 4. SHAP Analysis
    shap_importance, shap_values, shap_data_for_plot = perform_shap_analysis(
        df, feature_cols, min_train_months_for_shap=min_train_months, shap_sample_size=shap_sample_size
    )

    # 5. Governance Assessment
    governance_assessment = assess_governance(comparison_table_raw)

    return {
        'df': df,
        'feature_cols': feature_cols,
        'linear_results': linear_results,
        'ml_results': ml_results,
        'comparison_table_formatted': comparison_table_formatted,
        'comparison_table_raw': comparison_table_raw,
        'shap_importance': shap_importance,
        'shap_values': shap_values,
        'shap_data_for_plot': shap_data_for_plot,
        'governance_assessment': governance_assessment
    }

# --- Main Execution Block for standalone run ---
if __name__ == "__main__":
    print("Starting Stock Selection Analysis...")

    # Define parameters
    N_STOCKS = 200
    N_MONTHS = 120
    MIN_TRAIN_MONTHS = 60 # 5 years for initial training
    REBALANCE_FREQ = 1    # Monthly rebalancing
    RANDOM_SEED = 42
    SHAP_SAMPLE_SIZE = 1000

    # Run the full analysis pipeline
    results = run_stock_selection_analysis(
        n_stocks=N_STOCKS,
        n_months=N_MONTHS,
        min_train_months=MIN_TRAIN_MONTHS,
        rebalance_freq=REBALANCE_FREQ,
        seed=RANDOM_SEED,
        shap_sample_size=SHAP_SAMPLE_SIZE
    )

    df = results['df']
    feature_cols = results['feature_cols']
    linear_results = results['linear_results']
    ml_results = results['ml_results']
    comparison_table_formatted = results['comparison_table_formatted']
    shap_importance = results['shap_importance']
    shap_values = results['shap_values']
    shap_data_for_plot = results['shap_data_for_plot']
    governance_assessment = results['governance_assessment']

    # --- Print Data Generation Info ---
    print(f"\nPanel Data Generated: {N_STOCKS} stocks x {N_MONTHS} months = {len(df)} observations")
    print(f"Number of Features: {len(feature_cols)}")
    print(f"First 5 rows of the generated data:\n{df.head().to_string()}")
    print(f"\nReturn distribution: mean={df['next_month_return'].mean():.4f}, std={df['next_month_return'].std():.4f}")

    # --- Initial Model Checks (optional, for demonstration) ---
    # For illustration, we'll use a fixed split (first MIN_TRAIN_MONTHS-1 months for train, next one for test)
    if N_MONTHS > MIN_TRAIN_MONTHS:
        train_check_data = df[df['month'] < MIN_TRAIN_MONTHS-1]
        test_check_data = df[df['month'] == MIN_TRAIN_MONTHS-1]

        if not train_check_data.empty and not test_check_data.empty and len(train_check_data) > 0 and len(test_check_data) > 0:
            lin_pred_check, lin_model_check = linear_predict(train_check_data, test_check_data, feature_cols)
            print("\nLinear Model Coefficients (In-Sample Check):")
            print("=" * 45)
            for f, c in zip(feature_cols, lin_model_check.coef_):
                print(f" {f:<22s}: {c:+.6f}")

            ml_pred_check, _ = ml_predict(train_check_data, test_check_data, feature_cols)
            print(f"\nML Model (XGBoost) Prediction Sample (first 5): {ml_pred_check[:5]}")
        else:
            print(f"\nSkipping initial model checks: Not enough data for a meaningful split for demonstration (need at least {MIN_TRAIN_MONTHS} months).")

    # --- Print Backtest Results Summaries ---
    print(f"\nLinear (OLS) Strategy: {len(linear_results)} months of backtest results generated." if not linear_results.empty else "\nLinear (OLS) Strategy: No backtest results generated.")
    if not linear_results.empty:
        print(f"Mean L/S Return (OLS): {linear_results['ls_return'].mean()*100:.2f}%/month")

    print(f"\nML (XGBoost) Strategy: {len(ml_results)} months of backtest results generated." if not ml_results.empty else "\nML (XGBoost) Strategy: No backtest results generated.")
    if not ml_results.empty:
        print(f"Mean L/S Return (XGBoost): {ml_results['ls_return'].mean()*100:.2f}%/month")

    print("\nSample of Long-Short Returns (first 5 months of ML strategy):")
    if not ml_results.empty:
        print(ml_results[['month', 'ls_return']].head().to_string())
    else:
        print("No ML backtest results available.")

    # --- Print Strategy Comparison ---
    print("\nSTRATEGY COMPARISON")
    print("=" * 60)
    print(comparison_table_formatted.to_string())

    # --- Plot Cumulative Returns ---
    plot_cumulative_returns(linear_results, ml_results)

    # --- Print SHAP Feature Importance ---
    if shap_importance is not None and not shap_importance.empty:
        print("\nML MODEL SHAP FEATURE IMPORTANCE (Mean Absolute SHAP Value)")
        print("=" * 60)
        max_imp = shap_importance.max()
        for feat, imp in shap_importance.items():
            # Scale for visualization bar; max 20 hashes
            bar_length = int(imp / max_imp * 20) if max_imp > 0 else 0
            bar = '#' * bar_length
            print(f" {feat:<22s}: {imp:.5f} {bar}")

        # --- Plot SHAP Summary ---
        plot_shap_summary(shap_values, shap_data_for_plot)
    else:
        print("\nSHAP analysis was skipped due to insufficient training data.")

    # --- Print Governance Assessment ---
    print("\nGOVERNANCE ASSESSMENT")
    print("=" * 60)
    gov_results = governance_assessment
    print(f"Linear (OLS) Sharpe Ratio: {gov_results['lin_sharpe']:.2f}" if pd.notna(gov_results['lin_sharpe']) else "N/A")
    print(f"ML (XGBoost) Sharpe Ratio: {gov_results['ml_sharpe']:.2f}" if pd.notna(gov_results['ml_sharpe']) else "N/A")
    print(f"Sharpe Lift (ML vs. Linear): {gov_results['sharpe_lift']:+.2f}" if pd.notna(gov_results['sharpe_lift']) else "N/A")
    print(f"Relative Sharpe Lift: {gov_results['relative_sharpe_lift']:+.0f}%" if pd.notna(gov_results['relative_sharpe_lift']) else "N/A")
    print(f"\nVERDICT: {gov_results['verdict']}")

    print("\n--- Practitioner Warning ---")
    print("Simulated data typically overstates ML's advantage because nonlinear interactions are embedded.")
    print("Real-world Sharpe lifts are often in the 0.1-0.3 range and come with hidden costs like higher turnover.")
    print("Always validate on out-of-sample real data before drawing deployment conclusions.")
