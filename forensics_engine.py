"""
================================================================================
PYTHON FOR FINANCE - FINAL PROJECT 
================================================================================
Features:
1. Complete ETF Analysis: Returns, Volatility, Sharpe, Sortino, VaR, CVaR.
2. Clustering: K-Means + Group Profiling (Risk vs Return).
3. 'Forgotten Assets' Detection: Weak correlation analysis (< 0.45).
4. Mystery 1: NNLS resolution with confidence interval (Bootstrap 500 iterations).
5. Mystery 2: Dynamic Allocation (Rolling Window) + Stability Analysis.
6. Reporting: Automatic generation of charts and CSV/Log files.
"""

import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import nnls, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import r2_score

# ==============================================================================
# 0. CONFIGURATION & LOGGER
# ==============================================================================

@dataclass
class ProjectConfig:
    # Input Files (Adjust names if necessary)
    etf_file: Path = Path('Anonymized ETFs.csv')
    assets_file: Path = Path('Main Asset Classes.csv')
    mystery1_file: Path = Path('Mystery Allocation 1.csv')
    mystery2_file: Path = Path('Mystery Allocation 2.csv')
    
    # Output Parameters
    output_dir: Path = Path('Results') 
    
    # Financial Parameters
    risk_free_rate: float = 0.0
    correlation_threshold: float = 0.45 
    rolling_window: int = 60            
    bootstrap_iterations: int = 500     

    def setup_dirs(self):
        """Creates the directory structure without the Run_... subfolder"""
        root = self.output_dir 
        
        dirs = {
            'root': root,
            'graphs': root / 'Graphs',
            'reports': root / 'Reports',
            'logs': root / 'Logs'
        }
        
        # Creating directories
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        return dirs

class DualLogger:
    """Redirects prints to console AND a log file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Global Chart Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ==============================================================================
# 1. DATA ENGINE
# ==============================================================================
class DataEngine:
    def __init__(self, config: ProjectConfig):
        self.cfg = config

    def load_and_clean(self):
        print("\n" + "="*80)
        print(f"Loading data from: {Path.cwd()}")
        print("="*80)

        def _load_csv(path):
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Date format conversion (DD/MM/YYYY) if necessary
            df.index = pd.to_datetime(df.index, format='%d/%m/%Y', errors='coerce')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            return df

        try:
            etfs = _load_csv(self.cfg.etf_file)
            assets = _load_csv(self.cfg.assets_file)
            m1 = _load_csv(self.cfg.mystery1_file)
            m2 = _load_csv(self.cfg.mystery2_file)

            # Date intersection (Strict alignment)
            common_idx = etfs.index.intersection(assets.index)
            etfs = etfs.loc[common_idx]
            assets = assets.loc[common_idx]
            
            # Aligning mysteries
            m1 = m1.loc[m1.index.intersection(common_idx)]
            m2 = m2.loc[m2.index.intersection(common_idx)]

            print(f"✓ Data aligned on {len(common_idx)} trading days.")
            print(f"✓ ETFs loaded: {etfs.shape[1]} | Assets: {assets.shape[1]}")
            
            # Calculating log returns
            returns = {
                'etfs': np.log(etfs / etfs.shift(1)).dropna(),
                'assets': np.log(assets / assets.shift(1)).dropna(),
                'm1': np.log(m1 / m1.shift(1)).dropna(),
                'm2': np.log(m2 / m2.shift(1)).dropna()
            }
            
            # Aligned prices (for charts)
            prices = {'etfs': etfs, 'assets': assets, 'm1': m1, 'm2': m2}
            
            return returns, prices

        except Exception as e:
            print(f"FATAL ERROR: {e}")
            sys.exit(1)

# ==============================================================================
# 2. ANALYTICS MODULE (Metrics, Clustering, Correlations)
# ==============================================================================
class PortfolioAnalyzer:
    def __init__(self, etf_returns):
        self.returns = etf_returns
        self.metrics = None

    def compute_advanced_metrics(self):
        print(">> Computing advanced metrics (VaR, CVaR, Sortino)...")
        r = self.returns
        
        # Annualized Return & Volatility
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = ann_ret / ann_vol
        
        # Sortino (Downside risk only)
        downside = r[r < 0].std() * np.sqrt(252)
        sortino = ann_ret / downside
        
        # Max Drawdown
        cum_ret = (1 + r).cumprod()
        dd = (cum_ret - cum_ret.expanding().max()) / cum_ret.expanding().max()
        max_dd = dd.min()
        
        # VaR & CVaR (95%)
        var_95 = r.quantile(0.05)
        cvar_95 = r[r <= var_95].mean()
        
        self.metrics = pd.DataFrame({
            'Ann_Return': ann_ret,
            'Volatility': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max_Drawdown': max_dd,
            'VaR_95': var_95,
            'CVaR_95': cvar_95
        })
        return self.metrics

    def perform_clustering(self, n_clusters=6):
        print(f">> K-Means Clustering ({n_clusters} groups)...")
        features = ['Ann_Return', 'Volatility', 'Sharpe', 'Max_Drawdown']
        X = StandardScaler().fit_transform(self.metrics[features].fillna(0))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.metrics['Cluster'] = kmeans.fit_predict(X)
        
        # Profiling (Mean per cluster for interpretation)
        profile = self.metrics.groupby('Cluster')[features].mean()
        profile['Count'] = self.metrics['Cluster'].value_counts()
        return profile

class RelationshipManager:
    def __init__(self, etf_ret, asset_ret):
        self.etf_ret = etf_ret
        self.asset_ret = asset_ret

    def detect_forgotten_assets(self, threshold=0.45):
        """Identifies assets without strong correlation with main classes"""
        print(">> Analyzing cross-correlations & Forgotten Assets...")
        
        # Full matrix
        combined = pd.concat([self.etf_ret, self.asset_ret], axis=1)
        corr_matrix = combined.corr().loc[self.etf_ret.columns, self.asset_ret.columns]
        
        max_corr = corr_matrix.max(axis=1)
        best_match = corr_matrix.idxmax(axis=1)
        
        classification = pd.DataFrame({
            'Best_Asset': best_match,
            'Correlation': max_corr
        })
        
        # "Forgotten" Logic
        classification['Category'] = np.where(
            classification['Correlation'] < threshold, 
            'UNKNOWN/FORGOTTEN', 
            classification['Best_Asset']
        )
        
        unknowns = classification[classification['Category'] == 'UNKNOWN/FORGOTTEN']
        print(f"   -> {len(unknowns)} ETFs classified as 'Forgotten' (< {threshold})")
        
        return classification, corr_matrix

# ==============================================================================
# 3. SOLVER ENGINE (Mystery Allocations)
# ==============================================================================
class MysterySolver:
    def __init__(self, etf_returns):
        self.X = etf_returns

    def solve_static_with_uncertainty(self, mystery_ret, n_boot=500):
        """NNLS Solution + Bootstrap for uncertainty"""
        print(f">> Solving Mystery 1 (Static + Bootstrap {n_boot} iter)...")
        y = mystery_ret.squeeze()
        X = self.X.loc[y.index]
        
        # 1. Point estimate
        w_point, _ = nnls(X.values, y.values)
        if w_point.sum() > 0: w_point /= w_point.sum()
        
        # 2. Bootstrap (Monte Carlo)
        boot_weights = []
        for _ in range(n_boot):
            X_res, y_res = resample(X, y, random_state=None) # Sampling with replacement
            w, _ = nnls(X_res.values, y_res.values)
            if w.sum() > 0: w /= w.sum()
            boot_weights.append(w)
            
        boot_df = pd.DataFrame(boot_weights, columns=X.columns)
        
        # Statistical Summary
        stats = pd.DataFrame({
            'Weight_Est': w_point,
            'Mean_Boot': boot_df.mean(),
            'Std_Dev': boot_df.std(),
            'CI_Low_5%': boot_df.quantile(0.05),
            'CI_High_95%': boot_df.quantile(0.95)
        }).sort_values('Weight_Est', ascending=False)
        
        return stats, boot_df

    def solve_dynamic(self, mystery_ret, window=60):
        """Rolling Window NNLS"""
        print(f">> Solving Mystery 2 (Dynamic, Window={window})...")
        y = mystery_ret.squeeze()
        X = self.X.loc[y.index]
        
        dates, weights, r2_scores = [], [], []
        fitted_values = []
        
        # Step of 5 days to speed up calculation without losing too much precision
        step = 5 
        
        for i in range(0, len(y) - window, step):
            y_win = y.iloc[i : i + window]
            X_win = X.iloc[i : i + window]
            
            # Cleaning zero-variance columns in the window
            valid_cols = X_win.columns[X_win.var() > 1e-8]
            
            if len(valid_cols) > 0:
                w, _ = nnls(X_win[valid_cols].values, y_win.values)
                if w.sum() > 0: w /= w.sum()
                
                # Reconstructing full vector
                w_full = pd.Series(0.0, index=X.columns)
                w_full[valid_cols] = w
                
                # Calculate Fit quality (R2) on the window
                fit = X_win[valid_cols].values @ w
                r2 = r2_score(y_win, fit)
                
                weights.append(w_full.values)
                dates.append(y.index[i + window - 1])
                r2_scores.append(r2)
            
        df_weights = pd.DataFrame(weights, index=dates, columns=X.columns)
        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        print(f"   -> Average R² over rolling period: {avg_r2:.4f}")
        
        return df_weights, avg_r2

# ==============================================================================
# 4. VISUALIZATION ENGINE (ENRICHED)
# ==============================================================================
class Visualizer:
    def __init__(self, output_dir: Path, prices_dict: dict, metrics_df: pd.DataFrame):
        self.path = output_dir
        self.prices = prices_dict  # We need PRICES for curves, not just returns
        self.metrics = metrics_df  # For clusters

    def save(self, name):
        plt.savefig(self.path / name, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   -> Graph generated: {name}")

    def plot_efficient_frontier(self):
        """Risk-Return Chart colored by Cluster"""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=self.metrics, x='Volatility', y='Ann_Return',
            hue='Cluster', style='Cluster', palette='viridis', s=100, alpha=0.8
        )
        # Annotate "Stars" (Top Sharpe)
        top = self.metrics.sort_values('Sharpe', ascending=False).head(5)
        for idx, row in top.iterrows():
            plt.text(row['Volatility']+0.002, row['Ann_Return'], idx, fontsize=9, fontweight='bold')
            
        plt.title("Efficient Frontier & K-Means Clusters", fontweight='bold')
        plt.xlabel("Annualized Volatility (Risk)")
        plt.ylabel("Annualized Return")
        self.save("01_Efficient_Frontier.png")

    def plot_correlation_heatmap(self, corr_matrix):
        plt.figure(figsize=(18, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    linewidths=0.5, linecolor='gray')
        plt.title("Cross-Correlation Matrix (ETFs vs Asset Classes)", fontweight='bold')
        self.save("02_Correlation_Heatmap.png")

    def plot_cumulative_comparison(self):
        """Global comparison: Mystery 1, Mystery 2 vs Top 3 Assets"""
        plt.figure(figsize=(14, 8))
        
        # Base 100 Normalization
        p_m1 = self.prices['m1'] / self.prices['m1'].iloc[0] * 100
        p_m2 = self.prices['m2'] / self.prices['m2'].iloc[0] * 100
        p_assets = self.prices['assets'] / self.prices['assets'].iloc[0] * 100
        
        # Plot Mystery
        plt.plot(p_m1, label='Mystery Allocation 1', linewidth=2.5, color='black', linestyle='--')
        plt.plot(p_m2, label='Mystery Allocation 2', linewidth=2.5, color='red')
        
        # Plot Top 3 major Assets (often the first columns)
        for col in p_assets.columns[:3]:
            plt.plot(p_assets[col], label=f'Asset: {col}', alpha=0.5, linewidth=1)
            
        plt.title("Performance Comparison: Mystery Allocations vs Market (Base 100)", fontweight='bold')
        plt.legend()
        self.save("03_Global_Performance_Comparison.png")

    def plot_asset_tracking_grid(self, classification_df):
        """
        Generates a grid of charts. 
        For each Asset Class, shows the Asset (in bold) and linked ETFs.
        """
        groups = classification_df.groupby('Best_Asset')
        # Only take assets that have at least 1 associated ETF
        target_groups = [g for g in groups if len(g[1]) > 0]
        
        # Limit to 9 subplots for readability (3x3)
        n_plots = min(len(target_groups), 9)
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i in range(n_plots):
            asset_name, group_df = target_groups[i]
            ax = axes[i]
            
            # 1. Asset Class Plot (Benchmark)
            if asset_name in self.prices['assets'].columns:
                bench = self.prices['assets'][asset_name]
                bench = bench / bench.iloc[0] * 100
                ax.plot(bench, color='black', linewidth=2, label=f'BENCH: {asset_name}')
            
            # 2. Plot associated ETFs (Top 5 max to avoid clutter)
            etfs_in_group = group_df.index[:5] 
            for etf in etfs_in_group:
                if etf in self.prices['etfs'].columns:
                    p = self.prices['etfs'][etf]
                    p = p / p.iloc[0] * 100
                    ax.plot(p, alpha=0.6, linewidth=1, label=etf)
            
            ax.set_title(f"Group: {asset_name} ({len(group_df)} ETFs)", fontsize=10, fontweight='bold')
            ax.legend(fontsize='x-small', loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        self.save("04_Asset_Tracking_Details.png")

    def plot_cluster_performance(self):
        """Shows average performance of each Cluster"""
        plt.figure(figsize=(14, 8))
        
        # Add cluster column to prices
        etf_prices = self.prices['etfs'].copy()
        clusters = self.metrics['Cluster']
        
        for c_id in sorted(clusters.unique()):
            # Get ETFs in this cluster
            etfs_in_cluster = clusters[clusters == c_id].index
            if len(etfs_in_cluster) == 0: continue
            
            # Calculate average cluster price
            subset = etf_prices[etfs_in_cluster]
            subset_norm = subset / subset.iloc[0] * 100
            mean_path = subset_norm.mean(axis=1)
            
            plt.plot(mean_path, label=f'Cluster {c_id} (n={len(etfs_in_cluster)})', linewidth=2)
            
        plt.title("Average Performance by Cluster (Risk Profiles)", fontweight='bold')
        plt.legend()
        self.save("05_Cluster_Performance.png")

    def plot_mystery1_uncertainty(self, boot_df):
        top_assets = boot_df.mean().nlargest(15).index
        subset = boot_df[top_assets]
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=subset, orient='h', palette='Blues_r')
        plt.title("Mystery 1: Weight Uncertainty (Bootstrap)", fontweight='bold')
        self.save("06_Mystery1_Uncertainty.png")

    def plot_mystery2_dynamic(self, weights_df):
        top_cols = weights_df.sum().nlargest(10).index
        others = weights_df.columns.difference(top_cols)
        plot_data = weights_df[top_cols].copy()
        plot_data['Others'] = weights_df[others].sum(axis=1)
        
        plt.figure(figsize=(16, 8))
        plt.stackplot(plot_data.index, plot_data.T, labels=plot_data.columns, alpha=0.85, cmap='tab20')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Mystery 2: Dynamic Allocation Evolution", fontweight='bold')
        self.save("07_Mystery2_Dynamic_Evolution.png")


# ==============================================================================
# MAIN EXECUTION 
# ==============================================================================
def main():
    # 1. Setup
    cfg = ProjectConfig()
    dirs = cfg.setup_dirs()
    sys.stdout = DualLogger(dirs['logs'] / "execution_log.txt")
    
    start_time = datetime.datetime.now()

    # 2. Data Loading
    engine = DataEngine(cfg)
    returns, prices = engine.load_and_clean()

    # 3. ETF Analysis
    analyzer = PortfolioAnalyzer(returns['etfs'])
    metrics = analyzer.compute_advanced_metrics()
    clusters_profile = analyzer.perform_clustering() # Returns the profile
    
    metrics.to_csv(dirs['reports'] / "ETF_Metrics_Detailed.csv")
    clusters_profile.to_csv(dirs['reports'] / "Cluster_Profiles.csv")

    # 4. Relationships
    rel_mgr = RelationshipManager(returns['etfs'], returns['assets'])
    classif, corr_matrix = rel_mgr.detect_forgotten_assets(cfg.correlation_threshold)
    classif.to_csv(dirs['reports'] / "ETF_Classification.csv")
    corr_matrix.to_csv(dirs['reports'] / "Correlation_Matrix.csv")

    # 5. Solver Mystery 1
    solver = MysterySolver(returns['etfs'])
    m1_stats, m1_boot = solver.solve_static_with_uncertainty(returns['m1'], n_boot=cfg.bootstrap_iterations)
    m1_stats.to_csv(dirs['reports'] / "Mystery1_Results.csv")

    # 6. Solver Mystery 2
    m2_weights, m2_r2 = solver.solve_dynamic(returns['m2'], window=cfg.rolling_window)
    m2_weights.to_csv(dirs['reports'] / "Mystery2_Dynamic_Weights.csv")

    # 7. Visualization
    print("\n>> Generating enriched charts...")
    # Passing 'prices' and 'metrics' (which contains the Cluster column)
    viz = Visualizer(dirs['graphs'], prices, analyzer.metrics)
    
    viz.plot_efficient_frontier()
    viz.plot_correlation_heatmap(corr_matrix)
    viz.plot_cumulative_comparison()
    viz.plot_asset_tracking_grid(classif)
    viz.plot_cluster_performance()
    viz.plot_mystery1_uncertainty(m1_boot)
    viz.plot_mystery2_dynamic(m2_weights)

    # 8. End
    duration = datetime.datetime.now() - start_time
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETED in {duration}")
    print(f"7 Charts generated in: {dirs['graphs']}")
    print("="*80)

if __name__ == "__main__":
    main()