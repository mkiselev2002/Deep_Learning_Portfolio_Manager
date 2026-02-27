"""
generate_data.py
────────────────
Generates 3 years of synthetic-but-realistic daily ETF price data using a
correlated Geometric Brownian Motion model, then writes the CSV to data/.

Run once before starting the Streamlit app:
    python generate_data.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
SEED = 42
START_DATE = "2022-01-03"
END_DATE   = "2024-12-31"

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "VNQ", "XLE", "XLF"]

# Approximate starting prices (early 2022)
S0 = {
    "SPY": 477.0,
    "QQQ": 403.0,
    "IWM": 222.0,
    "EFA":  80.0,
    "EEM":  50.0,
    "GLD": 175.0,
    "TLT": 147.0,
    "VNQ": 115.0,
    "XLE":  62.0,
    "XLF":  40.0,
}

# Annual drift (μ) and volatility (σ)
MU = {
    "SPY": 0.12, "QQQ": 0.15, "IWM": 0.10, "EFA": 0.08, "EEM": 0.06,
    "GLD": 0.05, "TLT": 0.02, "VNQ": 0.09, "XLE": 0.10, "XLF": 0.11,
}
SIGMA = {
    "SPY": 0.18, "QQQ": 0.24, "IWM": 0.22, "EFA": 0.16, "EEM": 0.20,
    "GLD": 0.14, "TLT": 0.14, "VNQ": 0.20, "XLE": 0.28, "XLF": 0.22,
}

# Correlation matrix  [SPY QQQ IWM EFA EEM GLD TLT VNQ XLE XLF]
RAW_CORR = np.array([
    [ 1.00, 0.92, 0.88, 0.80, 0.72,  0.10, -0.18,  0.70,  0.65,  0.82],
    [ 0.92, 1.00, 0.80, 0.72, 0.65,  0.05, -0.14,  0.60,  0.55,  0.72],
    [ 0.88, 0.80, 1.00, 0.72, 0.65,  0.08, -0.16,  0.65,  0.62,  0.78],
    [ 0.80, 0.72, 0.72, 1.00, 0.75,  0.12, -0.14,  0.60,  0.58,  0.70],
    [ 0.72, 0.65, 0.65, 0.75, 1.00,  0.15, -0.09,  0.55,  0.60,  0.62],
    [ 0.10, 0.05, 0.08, 0.12, 0.15,  1.00,  0.28,  0.20,  0.18,  0.05],
    [-0.18,-0.14,-0.16,-0.14,-0.09,  0.28,  1.00, -0.14, -0.18, -0.22],
    [ 0.70, 0.60, 0.65, 0.60, 0.55,  0.20, -0.14,  1.00,  0.55,  0.65],
    [ 0.65, 0.55, 0.62, 0.58, 0.60,  0.18, -0.18,  0.55,  1.00,  0.60],
    [ 0.82, 0.72, 0.78, 0.70, 0.62,  0.05, -0.22,  0.65,  0.60,  1.00],
])


def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """Project to nearest positive-semidefinite matrix (Higham 1988)."""
    B = (A + A.T) / 2
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    # Ensure diagonal stays at 1
    D = np.sqrt(np.diag(A3))
    A3 = A3 / np.outer(D, D)
    return A3


def _make_cholesky(corr: np.ndarray) -> np.ndarray:
    """Return lower-triangular Cholesky factor, fixing near-PSD issues."""
    corr_psd = _nearest_psd(corr + np.eye(len(corr)) * 1e-8)
    return np.linalg.cholesky(corr_psd)


def generate_prices(seed: int = SEED) -> pd.DataFrame:
    """Simulate correlated log-normal price paths for all ETFs."""
    rng = np.random.default_rng(seed)

    bdays = pd.bdate_range(start=START_DATE, end=END_DATE)
    n = len(bdays)
    dt = 1 / 252  # daily step

    mu_vec    = np.array([MU[t]    for t in TICKERS])
    sigma_vec = np.array([SIGMA[t] for t in TICKERS])
    s0_vec    = np.array([S0[t]    for t in TICKERS])

    L = _make_cholesky(RAW_CORR)

    # Correlated standard normals: shape (n, 10)
    z_indep = rng.standard_normal((n, len(TICKERS)))
    z_corr  = z_indep @ L.T

    # Daily log-returns with GBM drift correction
    log_returns = (mu_vec - 0.5 * sigma_vec**2) * dt + sigma_vec * np.sqrt(dt) * z_corr

    # Cumulative product → price paths
    log_prices = np.cumsum(log_returns, axis=0)
    prices = s0_vec * np.exp(log_prices)

    df = pd.DataFrame(prices, index=bdays, columns=TICKERS)
    df.index.name = "Date"
    df = df.round(2)
    return df


def main():
    out_path = Path(__file__).parent / "data" / "etf_prices.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic ETF prices …")
    df = generate_prices()
    df.to_csv(out_path)

    print(f"Saved {len(df)} trading days × {len(df.columns)} ETFs to {out_path}")
    print("\nFinal prices (last row):")
    print(df.tail(1).to_string())
    print("\nDate range:", df.index[0].date(), "→", df.index[-1].date())


if __name__ == "__main__":
    main()
