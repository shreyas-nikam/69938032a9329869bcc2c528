import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shap
import xgboost as xgb
from source import *

# --- Page Configuration ---
st.set_page_config(page_title="QuLab: Lab 49: ML-Enhanced Factor Model ", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 49: ML-Enhanced Factor Model ")
st.divider()

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'

if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []

if 'n_stocks' not in st.session_state:
    st.session_state.n_stocks = 200
if 'n_months' not in st.session_state:
    st.session_state.n_months = 120
if 'min_train_months' not in st.session_state:
    st.session_state.min_train_months = 60
if 'rebalance_freq' not in st.session_state:
    st.session_state.rebalance_freq = 1

if 'backtest_run' not in st.session_state:
    st.session_state.backtest_run = False
if 'linear_results' not in st.session_state:
    st.session_state.linear_results = pd.DataFrame()
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = pd.DataFrame()

if 'comparison_table_formatted' not in st.session_state:
    st.session_state.comparison_table_formatted = pd.DataFrame()
if 'comparison_table_raw' not in st.session_state:
    st.session_state.comparison_table_raw = pd.DataFrame()

if 'shap_data_ready' not in st.session_state:
    st.session_state.shap_data_ready = False
if 'ml_final_model_for_shap' not in st.session_state:
    st.session_state.ml_final_model_for_shap = None
if 'train_all_for_shap' not in st.session_state:
    st.session_state.train_all_for_shap = pd.DataFrame()
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'importance' not in st.session_state:
    st.session_state.importance = pd.Series()

if 'cumulative_returns_plot' not in st.session_state:
    st.session_state.cumulative_returns_plot = None
if 'shap_summary_plot' not in st.session_state:
    st.session_state.shap_summary_plot = None

# --- Sidebar Navigation ---
st.sidebar.title("ML vs. Linear Factor Model")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Introduction",
        "1. Data Generation",
        "2. Backtesting & Portfolio Construction",
        "3. Performance Analysis",
        "4. Interpretability & Governance",
    ],
    index=[
        "Introduction",
        "1. Data Generation",
        "2. Backtesting & Portfolio Construction",
        "3. Performance Analysis",
        "4. Interpretability & Governance",
    ].index(st.session_state.page),
    key='page_selectbox'
)

if page_selection != st.session_state.page:
    st.session_state.page = page_selection
    st.rerun()

# --- Page Logic ---

if st.session_state.page == "Introduction":
    st.markdown(f"## Introduction: Enhancing Stock Selection for CFA Charterholders")
    st.markdown(f"As a CFA Charterholder and Investment Professional at a quantitative asset management firm, you are constantly seeking to refine and improve your firm's investment strategies. Traditional linear factor models have been the bedrock of quantitative investing, effectively capturing known relationships like value, momentum, and quality. However, you suspect that these models might be missing more complex, non-linear interactions within market data that could unlock incremental alpha.")
    st.markdown(f"This application will guide you through a real-world workflow to investigate if a machine learning (ML) model, specifically XGBoost, can outperform a traditional Ordinary Least Squares (OLS) linear factor model in predicting stock returns for a long-short equity strategy. We will simulate a comprehensive backtesting environment, compare performance using key financial metrics, analyze model interpretability using SHAP, and finally, assess the ML model's deployment readiness from a governance perspective.")
    st.markdown(f"Our goal is to move beyond static asset allocation and explore a more adaptive, data-driven approach to portfolio management. By aligning portfolio construction with the economic cycle and market sentiment, the firm can potentially enhance returns during favorable periods and mitigate drawdowns during adverse conditions, improving overall risk-adjusted performance for clients.")
    st.markdown(f"### Learning Outcomes:")
    st.markdown(f"* Implement OLS and XGBoost models for cross-sectional return prediction.")
    st.markdown(f"* Execute a robust walk-forward backtest with monthly rebalancing.")
    st.markdown(f"* Form long-short quintile portfolios based on model predictions.")
    st.markdown(f"* Compare model performance using Sharpe ratio, maximum drawdown, return, volatility, hit rate, and turnover.")
    st.markdown(f"* Conduct SHAP analysis to interpret the ML model's feature importance.")
    st.markdown(f"* Assess the governance implications of deploying a more complex ML model based on its performance lift.")

elif st.session_state.page == "1. Data Generation":
    st.markdown(f"## 1. Generating Synthetic Panel Data with Non-Linear Alpha Signals")
    st.markdown(f"As a quantitative investment professional, accessing high-quality, clean historical data with embedded alpha signals is paramount for research. For initial model development and testing, synthetic data offers a controlled environment to ensure our models are learning specific, known patterns, including non-linear interactions that might be overlooked by traditional linear approaches. We will generate panel data for 200 stocks over 10 years (120 months) with diverse financial features and a `next_month_return` target that includes both linear and non-linear components.")
    st.markdown(f"The generation process will embed specific relationships:")
    st.markdown(f"* **Linear factors**: Value (inverted P/E), Quality (ROE), Momentum, Earnings Growth, Book Value, Analyst Revisions, Market Cap, Short Interest.")
    st.markdown(f"* **Non-linear interactions**:")
    st.markdown(f"    * Value-Momentum Interaction: Low P/E (value) combined with positive momentum, leading to higher returns. This is expressed as `(0.003 * (-pe_z)) * (momentum_12m > 0)`.")
    st.markdown(f"    * Volatility-Conditional Momentum: Volatility hurting returns in down-momentum periods, expressed as `(-0.002 * volatility * (momentum_12m < 0))`.")
    st.markdown(f"    * Size Effect: Smaller companies (lower `log_mcap`) having a weaker negative impact or even a positive impact on returns, contrasting with a general negative `log_mcap` coefficient.")
    st.markdown(f"This controlled environment helps us evaluate if ML models can indeed \"discover\" these pre-defined non-linearities.")

    st.subheader("Data Generation Parameters")
    st.session_state.n_stocks = st.slider("Number of Stocks", 100, 500, st.session_state.n_stocks, key='n_stocks_slider')
    st.session_state.n_months = st.slider("Number of Months (e.g., 120 months = 10 years)", 60, 240, st.session_state.n_months, key='n_months_slider')

    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating data..."):
            df, feature_cols = generate_synthetic_data(st.session_state.n_stocks, st.session_state.n_months)
            st.session_state.df = df
            st.session_state.feature_cols = feature_cols
            st.session_state.data_generated = True
            st.session_state.backtest_run = False
            st.session_state.shap_data_ready = False
            st.success("Data Generated!")

    if st.session_state.data_generated:
        st.markdown(f"Panel Data Generated: `{st.session_state.n_stocks} stocks x {st.session_state.n_months} months = {len(st.session_state.df)} observations`")
        st.markdown(f"Number of Features: `{len(st.session_state.feature_cols)}`")
        st.markdown(f"First 5 rows of the generated data:")
        st.dataframe(st.session_state.df.head())
        st.markdown(f"Return distribution: mean=`{st.session_state.df['next_month_return'].mean():.4f}`, std=`{st.session_state.df['next_month_return'].std():.4f}`")
        st.markdown(f"The output confirms the generation of our synthetic panel data. The `next_month_return` distribution, with its small mean and larger standard deviation, reflects the challenging nature of predicting stock returns. This dataset now provides a controlled environment to rigorously test our models' ability to capture both linear and non-linear alpha.")

elif st.session_state.page == "2. Backtesting & Portfolio Construction":
    st.markdown(f"## 2. Walk-Forward Backtesting and Long-Short Portfolio Construction")
    st.markdown(f"Developing predictive models is only half the battle; the true measure of an investment strategy's efficacy is its performance in a realistic, out-of-sample simulation. As a CFA Charterholder, you understand the critical importance of robust backtesting to prevent \"look-ahead bias\" and ensure that any observed alpha is genuinely attainable. We will implement a walk-forward backtest, which simulates trading over time by progressively expanding the training window and rebalancing the portfolio monthly.")
    st.markdown(f"For each rebalancing point, we will:")
    st.markdown(f"1. Train the model (either OLS or XGBoost) on historical data up to that point.")
    st.markdown(f"2. Predict returns for the *next* month's stocks.")
    st.markdown(f"3. Form a **long-short quintile portfolio**:")
    st.markdown(f"    * Go long the top quintile (20% of stocks) with the highest predicted returns.")
    st.markdown(f"    * Go short the bottom quintile (20% of stocks) with the lowest predicted returns.")
    st.markdown(r"    * The return for this long-short portfolio at time $t$ ($r_{L/S,t}$) is calculated as the mean return of the top quintile ($Q_5$) minus the mean return of the bottom quintile ($Q_1$):")
    st.markdown(r"$$ r_{L/S,t} = \frac{{1}}{{|Q_5|}} \sum_{{i \in Q_5}} r_{{i,t}} - \frac{{1}}{{|Q_1|}} \sum_{{i \in Q_1}} r_{{i,t}} $$")
    st.markdown(r"where $r_{{i,t}}$ is the actual return of stock $i$ at month $t$.")
    st.markdown(f"This approach directly monetizes the models' predictive power by identifying the strongest buys and sells.")

    if not st.session_state.data_generated:
        st.warning("Please generate synthetic data first on the '1. Data Generation' page.")
    else:
        st.subheader("Backtest Parameters")
        st.session_state.min_train_months = st.slider("Minimum Training Months for Backtest", 36, 96, st.session_state.min_train_months, key='min_train_months_slider')
        st.session_state.rebalance_freq = st.slider("Rebalance Frequency (months)", 1, 3, st.session_state.rebalance_freq, key='rebalance_freq_slider')

        if st.button("Run Walk-Forward Backtest"):
            with st.spinner("Running backtests for OLS and XGBoost... This may take a moment."):
                # Run OLS Backtest
                linear_results = walk_forward_backtest(
                    st.session_state.df, st.session_state.feature_cols, linear_predict,
                    min_train_months=st.session_state.min_train_months,
                    rebalance_freq=st.session_state.rebalance_freq
                )
                # Run ML Backtest
                ml_results = walk_forward_backtest(
                    st.session_state.df, st.session_state.feature_cols, ml_predict,
                    min_train_months=st.session_state.min_train_months,
                    rebalance_freq=st.session_state.rebalance_freq
                )
                
                st.session_state.linear_results = linear_results
                st.session_state.ml_results = ml_results
                st.session_state.backtest_run = True

                # Comparison Tables
                comparison_table_formatted, comparison_table_raw = compare_strategies(linear_results, ml_results)
                st.session_state.comparison_table_formatted = comparison_table_formatted
                st.session_state.comparison_table_raw = comparison_table_raw

                # SHAP Analysis Preparation
                min_train_months_for_shap = st.session_state.min_train_months
                train_all_for_shap = st.session_state.df[st.session_state.df['month'] < min_train_months_for_shap]
                
                if not train_all_for_shap.empty and len(train_all_for_shap) > 1000:
                    ml_final_model_for_shap = xgb.XGBRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        objective='reg:squarederror'
                    )
                    ml_final_model_for_shap.fit(train_all_for_shap[st.session_state.feature_cols], train_all_for_shap['next_month_return'])

                    explainer = shap.TreeExplainer(ml_final_model_for_shap)
                    shap_sample_data = train_all_for_shap[st.session_state.feature_cols].sample(min(1000, len(train_all_for_shap)), random_state=42)
                    shap_values = explainer.shap_values(shap_sample_data)
                    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=st.session_state.feature_cols).sort_values(ascending=False)

                    st.session_state.ml_final_model_for_shap = ml_final_model_for_shap
                    st.session_state.train_all_for_shap = train_all_for_shap
                    st.session_state.shap_values = shap_values
                    st.session_state.importance = importance

                    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, shap_sample_data, show=False, plot_size=(10,6))
                    # Note: shap.summary_plot directly plots on the current figure/axis if handled carefully, 
                    # but typically manages its own figure unless 'show=False' and passed 'ax' (if supported) or captured from plt.
                    # The 'ax' parameter in summary_plot is available in newer versions, but 'show=False' allows manipulation of plt.gcf().
                    # We will rely on plt.gcf() if specific axis control is tricky with the installed shap version, 
                    # or just use the figure returned/active.
                    # For robustness with standard shap usage:
                    plt.title('SHAP Summary Plot for XGBoost Model')
                    plt.tight_layout()
                    st.session_state.shap_summary_plot = plt.gcf()
                    
                    st.session_state.shap_data_ready = True
                else:
                    st.warning("Not enough training data for robust SHAP analysis. Skipping SHAP generation.")
                    st.session_state.shap_data_ready = False

                st.success("Backtest Completed!")

        if st.session_state.backtest_run:
            st.markdown(f"Linear (OLS) Strategy: `{len(st.session_state.linear_results)}` months of backtest results generated.")
            st.markdown(f"Mean L/S Return (OLS): `{st.session_state.linear_results['ls_return'].mean()*100:.2f}%/month`")
            st.markdown(f"ML (XGBoost) Strategy: `{len(st.session_state.ml_results)}` months of backtest results generated.")
            st.markdown(f"Mean L/S Return (XGBoost): `{st.session_state.ml_results['ls_return'].mean()*100:.2f}%/month`")
            st.markdown(f"Sample of Long-Short Returns (first 5 months of ML strategy):")
            st.dataframe(st.session_state.ml_results[['month', 'ls_return']].head())
            st.markdown(f"The backtesting process is now complete for both the OLS and XGBoost models. We have obtained a time series of monthly long-short returns for each strategy. The output provides a quick glance at the average monthly performance, hinting at which model might be performing better. This data forms the basis for our comprehensive performance comparison and attribution, where we will quantify the strategies' risk-adjusted returns.")

elif st.session_state.page == "3. Performance Analysis":
    st.markdown(f"## 3. Performance Analysis and Strategy Comparison")
    st.markdown(f"As an investment professional, quantifying and comparing the risk-adjusted performance of different strategies is a core responsibility. With our walk-forward backtest results, we can now rigorously evaluate the OLS baseline against the XGBoost ML model using a suite of standard financial metrics. This comparison will directly inform whether the added complexity of the ML model delivers a significant \"alpha lift\" that warrants further consideration.")
    st.markdown(f"We will calculate the following key metrics:")
    st.markdown(f"* **Annualized Return**:")
    st.markdown(r"$$ \text{Annualized Return} = \mu \times 12 $$")
    st.markdown(r"where $\mu$ is the monthly mean return.")
    st.markdown(f"* **Annualized Volatility**:")
    st.markdown(r"$$ \text{Annualized Volatility} = \sigma \times \sqrt{{12}} $$")
    st.markdown(r"where $\sigma$ is the monthly standard deviation of returns.")
    st.markdown(f"* **Sharpe Ratio (SR)**: Measures risk-adjusted return. For a long-short strategy, it's the annualized mean return divided by the annualized standard deviation of returns.")
    st.markdown(r"$$ SR = \frac{{\mu_{{L/S}} \times 12}}{{\sigma_{{L/S}} \times \sqrt{{12}}}} $$")
    st.markdown(r"where $\mu_{{L/S}}$ is the monthly mean of long-short returns and $\sigma_{{L/S}}$ is the monthly standard deviation of long-short returns.")
    st.markdown(f"* **Maximum Drawdown**: The largest peak-to-trough decline over a specific period.")
    st.markdown(r"$$ \text{{Max Drawdown}} = \max_t \left( \frac{{V_{{peak,t}} - V_t}}{{V_{{peak,t}}}} \right) $$")
    st.markdown(r"where $V_t$ is the portfolio value at time $t$, and $V_{{peak,t}}$ is the maximum portfolio value up to time $t$.")
    st.markdown(f"* **Hit Rate**: The percentage of months with positive long-short returns.")
    st.markdown(f"* **Total Return**: Cumulative return over the entire backtesting period.")
    st.markdown(f"* **Turnover**: The amount of buying and selling required to maintain the portfolio. While not directly calculated in `compute_metrics`, it's a critical consideration for transaction costs. We will visualize cumulative returns and implicitly consider turnover later.")

    if not st.session_state.backtest_run:
        st.warning("Please run the backtest first on the '2. Backtesting & Portfolio Construction' page.")
    else:
        st.subheader("Strategy Comparison")
        st.dataframe(st.session_state.comparison_table_formatted)

        st.subheader("Cumulative Returns")
        fig_cum_ret, ax_cum_ret = plt.subplots(figsize=(12, 6))
        (1 + st.session_state.linear_results['ls_return']).cumprod().plot(label='Linear (OLS) Strategy', color='blue', ax=ax_cum_ret)
        (1 + st.session_state.ml_results['ls_return']).cumprod().plot(label='ML (XGBoost) Strategy', color='green', ax=ax_cum_ret)
        ax_cum_ret.set_title('Cumulative Returns of Long-Short Strategies')
        ax_cum_ret.set_xlabel('Months')
        ax_cum_ret.set_ylabel('Cumulative Return')
        ax_cum_ret.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax_cum_ret.grid(True, linestyle='--', alpha=0.6)
        ax_cum_ret.legend()
        plt.tight_layout()
        
        st.session_state.cumulative_returns_plot = fig_cum_ret
        st.pyplot(st.session_state.cumulative_returns_plot)
        plt.close(fig_cum_ret)

        st.markdown(f"The strategy comparison table and cumulative returns plot provide a clear picture of how the OLS and XGBoost models performed. The ML model, leveraging its ability to capture non-linearities, likely shows a higher Sharpe Ratio and cumulative return, potentially with a shallower maximum drawdown and higher hit rate. This quantitative evidence is crucial for you, as a CFA Charterholder, to determine if the ML approach offers a meaningful performance edge over the traditional linear baseline. The \"Sharpe Lift\" (difference in Sharpe Ratios) is a key metric that will drive our governance decision.")

elif st.session_state.page == "4. Interpretability & Governance":
    st.markdown(f"## 4. Model Interpretability (SHAP) and Deployment Governance Assessment")
    st.markdown(f"A higher Sharpe Ratio from an ML model is exciting, but for an investment professional, it's not enough to justify deployment. You need to understand *why* the model performs well (interpretability) and assess the practical implications of its complexity (governance). This section focuses on both.")
    st.markdown(f"**SHAP (SHapley Additive exPlanations)** will be used to interpret the XGBoost model. SHAP values attribute the contribution of each feature to a specific prediction, helping us understand the model's \"thinking.\" This is crucial for verifying if the model's insights align with financial intuition (e.g., if momentum or value are genuinely important).")
    st.markdown(f"Next, we perform a **Governance Assessment** based on the observed \"Sharpe Lift\" — the difference between the ML model's Sharpe Ratio and the linear model's Sharpe Ratio. This connects directly to the firm's framework for justifying the complexity of ML models:")
    st.markdown(f"* **Substantial Alpha Lift**: ML complexity is justified, proceed with Tier 2 governance (validation + monitoring).")
    st.markdown(r"$$ \text{Sharpe Lift} > 0.3 $$")
    st.markdown(f"* **Moderate Alpha Lift**: ML complexity is conditionally justified, requires SHAP review + ongoing performance monitoring.")
    st.markdown(r"$$ 0.1 < \text{Sharpe Lift} \le 0.3 $$")
    st.markdown(f"* **Marginal Alpha Lift**: ML complexity is NOT justified, deploy the simpler linear model (lower governance burden).")
    st.markdown(r"$$ \text{Sharpe Lift} \le 0.1 $$")
    st.markdown(f"This systematic approach ensures that the firm's resources are allocated efficiently and that models are deployed responsibly.")

    if not st.session_state.backtest_run:
        st.warning("Please run the backtest first on the '2. Backtesting & Portfolio Construction' page.")
    elif not st.session_state.shap_data_ready:
        st.warning("SHAP data not ready. This might happen if there was insufficient training data for SHAP analysis during backtest. Try generating data or adjusting parameters.")
    else:
        st.subheader("ML Model SHAP Feature Importance")
        st.markdown(f"Mean Absolute SHAP Value (higher value means greater importance):")
        for feat, imp in st.session_state.importance.items():
            bar_length = int(imp / st.session_state.importance.max() * 20)
            bar = '#' * bar_length
            st.markdown(f"`{feat:<22s}: {imp:.5f} {bar}`")

        if st.session_state.shap_summary_plot:
            st.pyplot(st.session_state.shap_summary_plot)

        st.subheader("Governance Assessment")
        lin_sharpe = st.session_state.comparison_table_raw.loc['Linear (OLS)', 'sharpe']
        ml_sharpe = st.session_state.comparison_table_raw.loc['ML (XGBoost)', 'sharpe']
        sharpe_lift = ml_sharpe - lin_sharpe

        st.markdown(f"Linear (OLS) Sharpe Ratio: `{lin_sharpe:.2f}`")
        st.markdown(f"ML (XGBoost) Sharpe Ratio: `{ml_sharpe:.2f}`")
        st.markdown(f"Sharpe Lift (ML vs. Linear): `{sharpe_lift:+.2f}`")
        st.markdown(f"Relative Sharpe Lift: `{(ml_sharpe / max(lin_sharpe, 0.01) - 1) * 100:+.0f}%`")

        if sharpe_lift > 0.3:
            st.markdown(f"**VERDICT: Substantial alpha lift. ML complexity JUSTIFIED.**")
            st.markdown(f"         Proceed with Tier 2 governance (validation + monitoring).")
        elif sharpe_lift > 0.1:
            st.markdown(f"**VERDICT: Moderate alpha lift. ML complexity CONDITIONALLY justified.**")
            st.markdown(f"         Requires SHAP review + ongoing performance monitoring.")
        else:
            st.markdown(f"**VERDICT: Marginal alpha lift. ML complexity NOT justified.**")
            st.markdown(f"         Deploy the simpler linear model (lower governance burden).")

        st.markdown(f"---")
        st.markdown(f"**Practitioner Warning**")
        st.markdown(f"Simulated data typically overstates ML's advantage because nonlinear interactions are embedded.")
        st.markdown(f"Real-world Sharpe lifts are often in the 0.1-0.3 range and come with hidden costs like higher turnover.")
        st.markdown(f"Always validate on out-of-sample real data before drawing deployment conclusions.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
