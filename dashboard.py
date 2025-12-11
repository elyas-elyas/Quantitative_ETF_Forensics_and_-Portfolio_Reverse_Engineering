import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import nnls, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.utils import resample
import warnings

# ==============================================================================
# GENERAL CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Finance Optimization Dashboard (Pro)", 
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings('ignore')

# ==============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ==============================================================================

class DashboardAnalyzer:
    """Class grouping financial calculations for the dashboard"""
    
    @staticmethod
    def calculate_returns(prices):
        """Calculates log-returns"""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def calculate_metrics(returns):
        """Calculates risk and performance metrics (PRO: Includes Sortino, VaR, CVaR)"""
        metrics = pd.DataFrame(index=returns.columns)
        
        # 1. Annualized Performance & Volatility
        metrics['Annualized_Return'] = returns.mean() * 252 
        metrics['Volatility'] = returns.std() * np.sqrt(252)
        
        # 2. Ratios
        metrics['Sharpe_Ratio'] = metrics['Annualized_Return'] / metrics['Volatility']
        
        # Sortino Ratio (Downside risk only)
        downside = returns[returns < 0].std() * np.sqrt(252)
        metrics['Sortino_Ratio'] = metrics['Annualized_Return'] / downside
        
        # 3. Extreme Risk (VaR & CVaR 95%)
        metrics['VaR_95'] = returns.quantile(0.05)
        # CVaR: Mean of returns below VaR
        metrics['CVaR_95'] = [returns[c][returns[c] <= metrics.loc[c, 'VaR_95']].mean() for c in returns.columns]
        
        # 4. Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['Max_Drawdown'] = drawdown.min()
        
        return metrics

    @staticmethod
    def classify_assets(etf_returns, asset_returns, threshold=0.45):
        """Detects 'Forgotten Assets' and best correlations"""
        corr_matrix = pd.DataFrame(index=etf_returns.columns, columns=asset_returns.columns)
        for e in etf_returns.columns:
            for a in asset_returns.columns:
                corr_matrix.loc[e, a] = etf_returns[e].corr(asset_returns[a])
        
        corr_matrix = corr_matrix.astype(float)
        
        # Classification
        max_corr = corr_matrix.max(axis=1)
        best_match = corr_matrix.idxmax(axis=1)
        
        classification = pd.DataFrame({
            'Best_Asset': best_match,
            'Correlation': max_corr
        })
        
        classification['Category'] = np.where(
            classification['Correlation'] < threshold, 
            'UNKNOWN/FORGOTTEN', 
            classification['Best_Asset']
        )
        
        return classification, corr_matrix

    @staticmethod
    def solve_bootstrap(etf_returns, mystery_returns, n_boot=200):
        """Solves allocation using Bootstrap to estimate uncertainty (Boxplot)"""
        y = mystery_returns.squeeze()
        aligned_X, aligned_y = etf_returns.align(y, join='inner', axis=0)
        
        boot_weights = []
        
        # Progress bar
        progress_text = f"Bootstrap Simulation ({n_boot} iterations)..."
        bar = st.progress(0, text=progress_text)
        
        for i in range(n_boot):
            # Resample (sampling with replacement)
            X_res, y_res = resample(aligned_X, aligned_y, random_state=None)
            
            # NNLS
            w, _ = nnls(X_res.values, y_res.values)
            if w.sum() > 0: w /= w.sum()
            boot_weights.append(w)
            
            if i % 10 == 0:
                bar.progress(min((i+1)/n_boot, 1.0), text=progress_text)
                
        bar.empty()
        
        boot_df = pd.DataFrame(boot_weights, columns=aligned_X.columns)
        return boot_df

    @staticmethod
    def solve_monthly(etf_returns, mystery_returns):
        """
        Solves allocation with Monthly Rebalancing.
        Optimizes weights for each calendar month.
        """
        mystery_returns = mystery_returns.squeeze()
        
        # Align data to keep only common dates
        common_idx = etf_returns.index.intersection(mystery_returns.index)
        X_full = etf_returns.loc[common_idx]
        y_full = mystery_returns.loc[common_idx]
        
        # Create monthly periods
        periods = X_full.index.to_period('M')
        unique_periods = periods.unique()
        
        monthly_weights = []
        period_dates = []
        fitted_values_list = []
        fitted_index = []

        progress_text = "Monthly Optimization in progress..."
        progress_bar = st.progress(0, text=progress_text)
        
        for i, period in enumerate(unique_periods):
            # Update progress bar
            progress_bar.progress(min((i + 1) / len(unique_periods), 1.0), text=f"{progress_text} ({period})")
            
            # Mask for current month
            mask = (periods == period)
            X_month = X_full[mask].values
            y_month = y_full[mask].values
            
            # Ignore months with too little data (< 5 trading days)
            if len(y_month) < 5:
                continue

            n_assets = X_month.shape[1]

            # Objective function: Minimize sum of squared errors (Tracking Error)
            def objective(w):
                # fit = X @ w, error = sum((y - fit)^2)
                return np.sum((y_month - X_month @ w)**2)

            # Constraints: Sum of weights = 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            # Bounds: 0 <= weight <= 1 (Long only)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            init_guess = np.ones(n_assets) / n_assets
            
            # Optimization via SLSQP (Sequential Least SQuares Programming)
            res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Store results
            optimal_weights = res.x
            # Store end of month date for chronological display
            end_date = X_full[mask].index[-1]
            
            monthly_weights.append(optimal_weights)
            period_dates.append(end_date)
            
            # Calculate fitted values for this month with optimized weights
            month_fit = (X_month @ optimal_weights)
            fitted_values_list.extend(month_fit)
            fitted_index.extend(X_full[mask].index)

        progress_bar.empty()
        
        # DataFrame of weights (Index = End of Month)
        weights_df = pd.DataFrame(monthly_weights, index=period_dates, columns=etf_returns.columns)
        
        # Replicated returns series (Index = Trading Days)
        fitted_series = pd.Series(fitted_values_list, index=fitted_index).sort_index()
        
        return weights_df, fitted_series

# ==============================================================================
# DATA LOADING (WITH CACHE)
# ==============================================================================

@st.cache_data
def load_data(etf_file, assets_file, m1_file, m2_file):
    try:
        etf = pd.read_csv(etf_file, index_col=0, parse_dates=True)
        assets = pd.read_csv(assets_file, index_col=0, parse_dates=True)
        m1 = pd.read_csv(m1_file, index_col=0, parse_dates=True)
        m2 = pd.read_csv(m2_file, index_col=0, parse_dates=True)
        
        for df in [etf, assets, m1, m2]:
            # Robust date parsing
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, format='%d/%m/%Y', errors='coerce')
            
            df.dropna(how='all', inplace=True)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        common = etf.index.intersection(assets.index)
        
        if len(common) == 0:
            st.error("Error: No common dates found between files.")
            return None, None, None, None

        return etf.loc[common], assets.loc[common], m1, m2
    except Exception as e:
        st.error(f"Critical error during loading: {e}")
        return None, None, None, None

# ==============================================================================
# USER INTERFACE (MAIN)
# ==============================================================================

st.title("📊 Finance Pro Dashboard: GP Analysis")
st.markdown("---")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("📂 Configuration")
use_default = st.sidebar.checkbox("Use default files", value=True)

files = {}
if use_default:
    # Ensure these files exist in the same folder
    files = {
        'etf': 'Anonymized ETFs.csv',
        'assets': 'Main Asset Classes.csv',
        'm1': 'Mystery Allocation 1.csv',
        'm2': 'Mystery Allocation 2.csv'
    }
else:
    st.sidebar.markdown("### 📤 Upload")
    up_etf = st.sidebar.file_uploader("1. ETFs", type='csv')
    up_asset = st.sidebar.file_uploader("2. Assets", type='csv')
    up_m1 = st.sidebar.file_uploader("3. Mystery 1", type='csv')
    up_m2 = st.sidebar.file_uploader("4. Mystery 2", type='csv')
    
    if up_etf and up_asset and up_m1 and up_m2:
        files = {'etf': up_etf, 'assets': up_asset, 'm1': up_m1, 'm2': up_m2}
    else:
        st.warning("Waiting for files...")
        st.stop()

if files:
    etf_prices, asset_prices, m1_prices, m2_prices = load_data(
        files['etf'], files['assets'], files['m1'], files['m2']
    )

    if etf_prices is not None:
        etf_ret = DashboardAnalyzer.calculate_returns(etf_prices)
        asset_ret = DashboardAnalyzer.calculate_returns(asset_prices)
        m1_ret = DashboardAnalyzer.calculate_returns(m1_prices)
        m2_ret = DashboardAnalyzer.calculate_returns(m2_prices)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Analysis & Clusters", 
            "🔗 Forgotten Assets", 
            "🕵️ Mystery 1 (Bootstrap)", 
            "📅 Mystery 2 (Monthly)"
        ])

        # ==========================================================================
        # TAB 1: ETFs & CLUSTERS
        # ==========================================================================
        with tab1:
            st.header("Risk-Return Analysis & Cluster Profiles")
            
            # Calculate complete metrics
            metrics = DashboardAnalyzer.calculate_metrics(etf_ret)
            
            col_conf, col_graph = st.columns([1, 4])
            with col_conf:
                st.subheader("Clustering")
                n_clusters = st.slider("Number of Clusters", 2, 10, 6)
            
            # Clustering K-Means
            scaler = StandardScaler()
            features = ['Annualized_Return', 'Volatility', 'Sharpe_Ratio', 'Max_Drawdown']
            X_scaled = scaler.fit_transform(metrics[features].fillna(0))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            metrics['Cluster'] = kmeans.fit_predict(X_scaled)
            metrics['Cluster'] = metrics['Cluster'].astype(str)

            with col_graph:
                # 1. Efficient Frontier
                fig_ef = px.scatter(
                    metrics, 
                    x='Volatility', y='Annualized_Return',
                    color='Cluster', symbol='Cluster',
                    hover_name=metrics.index,
                    hover_data={'Sharpe_Ratio':':.2f', 'Sortino_Ratio':':.2f', 'Max_Drawdown':':.2f'},
                    title="Efficient Frontier (Risk vs Return)",
                    template="plotly_dark", height=500
                )
                fig_ef.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_ef, use_container_width=True)
                
                # 2. Average Cluster Performance
                st.subheader("Average Cumulative Performance by Cluster")
                cluster_prices = pd.DataFrame(index=etf_prices.index)
                
                for c in sorted(metrics['Cluster'].unique()):
                    etfs_in_c = metrics[metrics['Cluster'] == c].index
                    if len(etfs_in_c) > 0:
                        # Mean of normalized prices (Base 100)
                        subset = etf_prices[etfs_in_c]
                        subset_norm = subset / subset.iloc[0] * 100
                        cluster_prices[f'Cluster {c}'] = subset_norm.mean(axis=1)
                
                fig_perf = px.line(cluster_prices, title="Cluster Evolution (Base 100)")
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with st.expander("📊 View Detailed Table (with Sortino, VaR, CVaR)"):
                numeric_cols = ['Annualized_Return', 'Volatility', 'Sharpe_Ratio', 
                               'Sortino_Ratio', 'VaR_95', 'CVaR_95', 'Max_Drawdown']
                
                st.dataframe(
                    metrics.style
                    .format("{:.4f}", subset=numeric_cols)
                    .background_gradient(cmap='RdYlGn', subset=['Sharpe_Ratio'])
                )

        # ==========================================================================
        # TAB 2: CORRELATIONS & FORGOTTEN ASSETS
        # ==========================================================================
        with tab2:
            st.header("Asset Classification & Forgotten Assets")
            
            thresh = st.slider("'Forgotten' Correlation Threshold", 0.1, 0.9, 0.45, 0.05)
            
            # Call classification function
            classification, corr_matrix = DashboardAnalyzer.classify_assets(etf_ret, asset_ret, threshold=thresh)
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                fig_corr = px.imshow(
                    corr_matrix, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Cross-Correlation Heatmap"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_right:
                st.subheader("Distribution")
                distrib = classification['Category'].value_counts()
                st.bar_chart(distrib)
                
                st.subheader("⚠️ Forgotten Assets")
                forgotten = classification[classification['Category'] == 'UNKNOWN/FORGOTTEN']
                st.write(f"Number of assets detected: **{len(forgotten)}**")
                if not forgotten.empty:
                    st.dataframe(forgotten[['Correlation']])
                else:
                    st.success("No 'forgotten' assets found with this threshold.")

        # ==========================================================================
        # TAB 3: MYSTERY 1 (BOOTSTRAP)
        # ==========================================================================
        with tab3:
            st.header("Mystery 1: Uncertainty Analysis (Bootstrap)")
            st.markdown("Uses NNLS method with resampling to test robustness.")
            
            col_b_opt, col_b_res = st.columns([1, 3])
            
            with col_b_opt:
                n_boot = st.number_input("Bootstrap Iterations", min_value=50, max_value=1000, value=200, step=50)
                run_boot = st.button("Run Monte Carlo Simulation", type="primary")
                
            if run_boot:
                boot_df = DashboardAnalyzer.solve_bootstrap(etf_ret, m1_ret, n_boot=n_boot)
                
                with col_b_res:
                    # Filter for display (Top 15 by mean)
                    top_assets = boot_df.mean().nlargest(15).index
                    viz_df = boot_df[top_assets]
                    
                    fig_box = px.box(
                        viz_df, 
                        title=f"Weight Distribution (Uncertainty) - Top 15 Assets",
                        points="outliers"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    st.write("Summary Statistics:")
                    stats = pd.DataFrame({
                        'Mean': boot_df.mean(),
                        'Std Dev': boot_df.std(),
                        'Min': boot_df.min(),
                        'Max': boot_df.max()
                    }).sort_values('Mean', ascending=False).head(10)
                    st.dataframe(stats.style.format("{:.2%}"))

        # ==========================================================================
        # TAB 4: MYSTERY 2 (MONTHLY REBALANCING)
        # ==========================================================================
        with tab4:
            st.header("Mystery 2: Monthly Rebalancing Analysis")
            st.markdown("""
            This analysis performs a **monthly weight optimization**.
            The algorithm recalculates optimal weights (minimizing error) for each calendar month.
            """)
            
            if st.button("🚀 Run Monthly Analysis", type="primary"):
                # 1. Calculation (Call new function)
                weights_df, fitted_returns = DashboardAnalyzer.solve_monthly(etf_ret, m2_ret)
                
                # Align for metrics calculation
                common_idx = m2_ret.index.intersection(fitted_returns.index)
                y_true = m2_ret.loc[common_idx].squeeze()
                y_pred = fitted_returns.loc[common_idx]
                
                # 2. Statistical Metrics
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                corr = np.corrcoef(y_true, y_pred)[0, 1]
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("R² Score", f"{r2:.4f}")
                col_m2.metric("RMSE", f"{rmse:.6f}")
                col_m3.metric("Correlation", f"{corr:.4f}")
                
                st.divider()
                
                # 3. Cumulative Performance Graph (Base 100)
                cum_mystery = (1 + y_true).cumprod() * 100
                cum_fitted = (1 + y_pred).cumprod() * 100
                
                comp_df = pd.DataFrame({
                    'Mystery Allocation': cum_mystery,
                    'Replicated (Monthly)': cum_fitted
                })
                
                fig_perf = px.line(
                    comp_df, 
                    title="Cumulative Performance: Real vs Replicated (Base 100)",
                    color_discrete_map={'Mystery Allocation': '#2563eb', 'Replicated (Monthly)': '#16a34a'}
                )
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # 4. Allocation Graph (Stacked Area)
                # Filter for readability: only show assets that exceeded 1% at least once
                significant_etfs = weights_df.columns[weights_df.max() > 0.01]
                
                fig_alloc = px.area(
                    weights_df[significant_etfs],
                    title="Evolution of Monthly Allocation (Stacked Area)",
                    labels={"value": "Weight", "index": "Month"},
                    groupnorm='percent' 
                )
                st.plotly_chart(fig_alloc, use_container_width=True)
                
                # 5. Export
                with st.expander("📥 View Monthly Weights (Table)"):
                    st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)

    else:
        st.info("Please load the data.")