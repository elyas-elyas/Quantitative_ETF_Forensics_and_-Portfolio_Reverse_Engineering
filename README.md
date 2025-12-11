# Quantitative ETF Forensics & Portfolio Reverse-Engineering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

A financial toolkit designed for **Asset Classification**, **Risk Profiling**, and the **Reverse-Engineering** of "Mystery" portfolios. This project combines a high-performance calculation engine for batch reporting and an interactive dashboard for real-time analysis.

Link to the Dashboard: 
https://dashboardpythfin.streamlit.app/
---

## Project Objective

Provide quantitative analysts and portfolio managers with a dual-interface system to dissect ETF universes and uncover the composition of unknown allocations:

1.  **Forensics Engine (`forensics_engine.py`)**: A batch-processing script that generates static, high-resolution reports (PDF/PNG) and CSV logs using advanced clustering and tracking algorithms.
2.  **Interactive Dashboard (`dashboard.py`)**: A web-based interface to explore risk clusters, visualize confidence intervals via Bootstrap, and test dynamic rebalancing strategies.

---

## Key Features

| Feature | Description | Analysis Section |
|-------------------------------------|--------------------------------------------------------------------------------|--------------------------|
| **Advanced Risk Metrics** | Calculation of VaR (95%), CVaR (Expected Shortfall), Sortino Ratio, and Max Drawdown. | Risk Profiling |
| **Unsupervised Clustering** | K-Means clustering to categorize ETFs based on their risk/return profiles (Sharpe, Volatility, Return). | Clustering |
| **"Forgotten Assets" Detection** | Identification of assets with low correlation thresholds ($<0.45$) to main asset classes. | Forensics |
| **Portfolio Reverse-Engineering** | Uses Non-Negative Least Squares (NNLS) to solve for the weights of a target "Mystery" return series. | Solver |
| **Bootstrap Uncertainty** | Estimates confidence intervals for portfolio weights using Monte Carlo resampling (200-500 iterations). | Solver (Mystery 1) |
| **Dynamic Style Analysis** | Rolling-window analysis or Monthly Rebalancing optimization to track allocation shifts over time. | Solver (Mystery 2) |

---

## Methodology

### 1. Quantitative Core (Metrics)
- **Risk Measures**: Beyond standard volatility, the engine calculates **Conditional Value at Risk (CVaR)** to assess tail risk and **Sortino Ratio** to focus on downside deviation.
- **Normalization**: Data is strictly aligned on common trading days and converted to log-returns for statistical properties.

### 2. Machine Learning Clustering (K-Means)
- **Model**: K-Means Algorithm.
- **Features**: Annualized Return, Volatility, Sharpe Ratio, Max Drawdown.
- **Goal**: Group ETFs into distinct "Risk Profiles" to simplify the universe and identify outliers.

### 3. Allocation Solver (NNLS)
- **Mathematical Model**: $\min_x ||Ax - b||_2$ subject to $x \geq 0$.
- **Constraint**: Long-only (no short selling).
- **Application**: Finds the combination of ETFs ($A$) that best replicates the Mystery Returns ($b$).

### 4. Bootstrap Simulation
- **Method**: Resampling with replacement.
- **Process**: The solver runs $N$ times on resampled historical data.
- **Output**: Generates a distribution of weights (Boxplot), revealing the stability and statistical significance of the detected assets.

---

## Technologies Used

- **Python 3.8+**
- `streamlit` - Interactive Dashboard
- `scipy` - Optimization (`nnls`, `minimize`)
- `sklearn` - Machine Learning (`KMeans`, `StandardScaler`, `resample`)
- `pandas` & `numpy` - Vectorized financial calculations
- `plotly` - Interactive plotting (Dashboard)
- `seaborn` & `matplotlib` - Static reporting (Engine)

---

## Project Structure
```text
quantitative-etf-forensics/
│
├── forensics_engine.py       # Batch processing: generates static reports & logs
├── dashboard.py              # Interactive Streamlit application
├── Results/                  # Output directory (Graphs, Reports, Logs)
└── requirements.txt          # Python dependencies
```

---

## Installation and Usage

### Prerequisites

```bash
Python 3.8 or higher
pip
```

### Installation

1.  **Clone the repository**

```bash
git clone https://github.com/elyas-elyas/Quantitative_ETF_Forensics_and_Portfolio_Reverse_Engineering
cd quantitative-etf-forensics
```

2.  **Install dependencies**

```bash
pip install -r requirements.txt
```

3.  **Run the Forensics Engine (Batch Mode)**  
    *Generates static reports in the `Results/` folder.*

```bash
python forensics_engine.py
```

4.  **Launch the Dashboard (Interactive Mode)**  
    *Opens the web interface for dynamic exploration.*
    https://dashboardpythfin.streamlit.app/

```bash
streamlit run dashboard.py
```

---

## Strengths & Limitations

### Strengths

- **Dual Workflow**: Combines the depth of a static report engine with the flexibility of a web dashboard.
- **Robustness**: The use of Bootstrap prevents overfitting to a specific time period, offering "Confidence Intervals" on the detected weights.
- **Advanced Forensics**: The "Forgotten Assets" module helps identify diversification opportunities often overlooked by standard correlation matrices.
- **Tail Risk Awareness**: Includes CVaR and Sortino, essential for professional risk management.

### Limitations

- **Long-Only Constraint**: The NNLS solver assumes the mystery portfolio does not contain short positions.
- **Universe Bias**: The solver can only replicate the mystery portfolio using the provided list of ETFs.
- **Linearity**: Assumes linear relationships between assets (Pearson correlation).

---

## Key Concepts

| Concept | Explanation |
|-------------------------|----------------------------------------------------------|
| **NNLS** | Non-Negative Least Squares. An optimization technique used here to replicate a return series without short selling. |
| **Bootstrap** | A statistical technique involving resampling data to estimate the accuracy/uncertainty of a model (e.g., "Is this weight 10% by luck or skill?"). |
| **CVaR (95%)** | Conditional Value at Risk. The average loss given that the loss is greater than the VaR threshold (focus on extreme crashes). |
| **Rolling Window** | A technique to analyze how relationships (correlations, weights) change over time by moving a fixed-size window across the dataset. |
| **K-Means** | An unsupervised learning algorithm that partitions data into $k$ distinct clusters based on feature similarity. |

---

## Resources and References


* **Computer training: Advanced Python for Optimisation and Finance**. IRFA, Université Panthéon-Sorbonne.
* **Sharpe, W. F. (1992)**. "Asset Allocation: Management Style and Performance Measurement". *The Journal of Portfolio Management*. (Foundational paper for the Return-Based Style Analysis and NNLS methodology used in this project).
* **Markowitz, H. (1952)**. "Portfolio Selection". *The Journal of Finance*. (Basis for the Efficient Frontier and Risk-Return analysis).
* **Jorion, P. (2006)**. "Value at Risk: The New Benchmark for Managing Financial Risk". *McGraw-Hill*. (Reference for the VaR and CVaR risk metrics implementation).
* **Hilpisch, Y. (2018)**. "Python for Finance: Mastering Data-Driven Finance". *O'Reilly Media*.
