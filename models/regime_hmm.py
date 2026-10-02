"""
Project Chronos - 3-State Continuous Gaussian Hidden Markov Model (HMM)
Dynamic Regime-Conditioned Alpha Allocation & Macro Risk Governor

States:
  State 0: Trending / Directional     (High trend persistence, moderate realized vol)
  State 1: Choppy / Mean-Reverting   (Low trend persistence, low-to-moderate vol)
  State 2: Volatile Shock / Regime   (High realized vol, extreme dispersion)

Factor Families:
  - Momentum / Trend / Breakout: wpZvNlv1_INV, 88j2G0ZV_INV, ALPHA_04_VOL_MOMENTUM, 3qXJ27LN, A1NmeMQY, 78NeYnPb, 2rm1ePwb, N1VXbVbo_INV
  - Mean-Reversion / Quality / Value: rKOZWNJJ, d5bNdNVX, E5p9K3gr, MPa2Mmlz, 9qjkndAe, omLZoqXl, gJbG5Qrm, vRro8eqa_INV, vR2KelW3_INV
"""

import os
import json
import numpy as np
import pandas as pd
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMMLEARN = True
except ImportError:
    HAVE_HMMLEARN = False

class RegimeState(IntEnum):
    TRENDING = 0
    CHOPPY = 1
    VOLATILE_SHOCK = 2

STATE_NAMES = {
    RegimeState.TRENDING: "State 0: Trending / Directional",
    RegimeState.CHOPPY: "State 1: Choppy / Mean-Reverting",
    RegimeState.VOLATILE_SHOCK: "State 2: Volatile Shock / Regime Break"
}

STATE_DESCRIPTIONS = {
    RegimeState.TRENDING: "High trend persistence, moderate vol. Momentum/Trend alphas boosted 1.5x, Mean-Reversion dampened 0.5x.",
    RegimeState.CHOPPY: "Low trend persistence, range-bound vol. Mean-Reversion/Quality boosted 1.5x, Momentum dampened 0.5x.",
    RegimeState.VOLATILE_SHOCK: "Extreme volatility, regime break. Capital protection mode: gross leverage slashed by 45%."
}

# Categorization of active alphas into Factor Families
MOMENTUM_FAMILY = [
    "wpZvNlv1_INV", "88j2G0ZV_INV", "ALPHA_04_VOL_MOMENTUM",
    "3qXJ27LN", "A1NmeMQY", "78NeYnPb", "2rm1ePwb", "N1VXbVbo_INV"
]

MEAN_REVERSION_FAMILY = [
    "rKOZWNJJ", "d5bNdNVX", "E5p9K3gr", "MPa2Mmlz",
    "9qjkndAe", "omLZoqXl", "gJbG5Qrm", "vRro8eqa_INV", "vR2KelW3_INV"
]

# Baseline Master Ensemble Risk-Parity Factor Weights
BASE_ALPHA_WEIGHTS = {
    "rKOZWNJJ": 0.0672,
    "E5p9K3gr": 0.0560,
    "d5bNdNVX": 0.0694,
    "MPa2Mmlz": 0.0694,
    "9qjkndAe": 0.0560,
    "3qXJ27LN": 0.0560,
    "omLZoqXl": 0.0560,
    "A1NmeMQY": 0.0560,
    "gJbG5Qrm": 0.0560,
    "78NeYnPb": 0.0694,
    "2rm1ePwb": 0.0694,
    "88j2G0ZV_INV": 0.0655,
    "wpZvNlv1_INV": 0.0686,
    "vRro8eqa_INV": 0.0682,
    "vR2KelW3_INV": 0.0683,
    "N1VXbVbo_INV": 0.0489
}

@dataclass
class RegimeInference:
    """Telemetry report of current HMM market regime."""
    timestamp: str
    current_state: int
    state_name: str
    description: str
    probabilities: Dict[str, float]
    factor_tilts: Dict[str, float]
    conditioned_alpha_weights: Dict[str, float]
    target_gross_leverage: float
    transition_matrix: List[List[float]]
    realized_vol_20d: float
    trend_dispersion_20d: float

class PureNumpyGaussianHMM:
    """
    Robust pure NumPy implementation of a continuous 3-state Gaussian HMM
    using Baum-Welch (EM) training, Forward-Backward algorithm, and Viterbi decoding.
    Guarantees deterministic execution even in environments lacking external C libraries.
    """
    def __init__(self, n_states=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.start_prob = None
        self.trans_mat = None
        self.means = None
        self.covars = None

    def _init_params(self, X):
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        self.start_prob = np.full(self.n_states, 1.0 / self.n_states)
        self.trans_mat = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        # Initialize means using quantiles along first feature (volatility)
        q = np.linspace(0.2, 0.8, self.n_states)
        vol_q = np.quantile(X[:, 0], q)
        self.means = np.zeros((self.n_states, D))
        for k in range(self.n_states):
            mask = np.abs(X[:, 0] - vol_q[k]) <= np.std(X[:, 0])
            if np.any(mask):
                self.means[k] = np.mean(X[mask], axis=0)
            else:
                self.means[k] = X[k % N]
        self.covars = np.tile(np.eye(D) * (np.var(X, axis=0) + 1e-4), (self.n_states, 1, 1))

    def _gaussian_pdf(self, x, mean, cov):
        D = len(x)
        diff = x - mean
        reg_cov = cov + np.eye(D) * 1e-5
        sign, logdet = np.linalg.slogdet(reg_cov)
        if sign <= 0:
            reg_cov += np.eye(D) * 1e-3
            sign, logdet = np.linalg.slogdet(reg_cov)
        inv_cov = np.linalg.inv(reg_cov)
        maha = np.dot(diff, np.dot(inv_cov, diff))
        denom = np.sqrt(((2 * np.pi) ** D) * np.exp(logdet))
        return max(1e-12, np.exp(-0.5 * maha) / max(denom, 1e-12))

    def _emission_probs(self, X):
        N = len(X)
        B = np.zeros((N, self.n_states))
        for t in range(N):
            for k in range(self.n_states):
                B[t, k] = self._gaussian_pdf(X[t], self.means[k], self.covars[k])
        return B

    def _forward(self, B):
        N = len(B)
        alpha = np.zeros((N, self.n_states))
        alpha[0] = self.start_prob * B[0]
        s0 = np.sum(alpha[0])
        alpha[0] /= (s0 + 1e-12)
        for t in range(1, N):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.trans_mat[:, j]) * B[t, j]
            st = np.sum(alpha[t])
            alpha[t] /= (st + 1e-12)
        return alpha

    def _backward(self, B):
        N = len(B)
        beta = np.zeros((N, self.n_states))
        beta[-1] = 1.0
        for t in range(N - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.trans_mat[i, :] * B[t+1, :] * beta[t+1, :])
            st = np.sum(beta[t])
            beta[t] /= (st + 1e-12)
        return beta

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        self._init_params(X)
        for iteration in range(self.max_iter):
            B = self._emission_probs(X)
            alpha = self._forward(B)
            beta = self._backward(B)
            # Posterior state probabilities gamma_t(i)
            gamma = alpha * beta
            gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-12)
            # Transition posteriors xi_t(i, j)
            xi = np.zeros((N - 1, self.n_states, self.n_states))
            for t in range(N - 1):
                denom = np.sum(alpha[t, :, None] * self.trans_mat * B[t+1, None, :] * beta[t+1, None, :])
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.trans_mat[i, j] * B[t+1, j] * beta[t+1, j]) / (denom + 1e-12)
            # M-step update
            self.start_prob = gamma[0]
            self.trans_mat = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0)[:, None] + 1e-12)
            self.trans_mat /= (np.sum(self.trans_mat, axis=1, keepdims=True) + 1e-12)
            for k in range(self.n_states):
                gamma_k = gamma[:, k][:, None]
                sum_gamma_k = np.sum(gamma_k) + 1e-12
                self.means[k] = np.sum(gamma_k * X, axis=0) / sum_gamma_k
                diff = X - self.means[k]
                self.covars[k] = np.dot((gamma_k * diff).T, diff) / sum_gamma_k + np.eye(D) * 1e-4
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        B = self._emission_probs(X)
        alpha = self._forward(B)
        beta = self._backward(B)
        gamma = alpha * beta
        gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-12)
        return gamma

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class MarketRegimeHMM:
    """
    Production 3-State Gaussian Hidden Markov Model with canonical state alignment:
      - State 0: Trending / Directional
      - State 1: Choppy / Mean-Reverting
      - State 2: Volatile Shock
    """
    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.state_map = {0: 0, 1: 1, 2: 2} # internal -> canonical mapping
        self.transition_matrix = np.eye(3)
        self.state_means = np.zeros((3, 2))
        self.state_covariances = np.zeros((3, 2, 2))
        self.feature_means = None
        self.feature_stds = None

    def extract_features(self, panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extracts daily macro regime features:
          1. 20-day annualized realized universe volatility
          2. 20-day cross-sectional trend dispersion / momentum persistence
        """
        df = panel_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate individual daily returns
        df.sort_values(['symbol', 'date'], inplace=True)
        df['ret'] = df.groupby('symbol')['Close'].pct_change().fillna(0.0)

        # 1. Daily equal-weighted universe market return
        daily_mkt = df.groupby('date')['ret'].mean().reset_index()
        daily_mkt.columns = ['date', 'mkt_ret']
        daily_mkt.sort_values('date', inplace=True)
        
        # 20-day rolling annualized realized market volatility
        daily_mkt['realized_vol_20d'] = (
            daily_mkt['mkt_ret'].rolling(20, min_periods=10).std() * np.sqrt(252)
        ).fillna(0.15)

        # 2. 20-day cross-sectional dispersion across universe
        daily_cs_disp = df.groupby('date')['ret'].std().reset_index()
        daily_cs_disp.columns = ['date', 'cs_disp']
        daily_cs_disp['trend_dispersion_20d'] = (
            daily_cs_disp['cs_disp'].rolling(20, min_periods=10).mean()
        ).fillna(0.015)

        # Merge daily features
        feat_df = pd.merge(daily_mkt, daily_cs_disp, on='date').dropna()
        feat_df.sort_values('date', inplace=True)

        features = feat_df[['realized_vol_20d', 'trend_dispersion_20d']].values
        return feat_df, features

    def fit(self, panel_df: pd.DataFrame) -> "MarketRegimeHMM":
        """Fits 3-state Gaussian HMM and canonicalizes states."""
        feat_df, X_raw = self.extract_features(panel_df)
        if len(X_raw) < 50:
            raise ValueError(f"Insufficient historical data ({len(X_raw)} rows) to fit 3-state Gaussian HMM.")

        # Standardize features
        self.feature_means = np.mean(X_raw, axis=0)
        self.feature_stds = np.std(X_raw, axis=0) + 1e-6
        X_scaled = (X_raw - self.feature_means) / self.feature_stds

        # 1. Train Model
        if HAVE_HMMLEARN:
            try:
                hmm = GaussianHMM(
                    n_components=3,
                    covariance_type="full",
                    n_iter=200,
                    random_state=self.random_state,
                    tol=1e-4
                )
                hmm.fit(X_scaled)
                self.model = hmm
                raw_means = hmm.means_
                raw_trans = hmm.transmat_
            except Exception:
                fallback = PureNumpyGaussianHMM(n_states=3, max_iter=100, random_state=self.random_state)
                fallback.fit(X_scaled)
                self.model = fallback
                raw_means = fallback.means
                raw_trans = fallback.trans_mat
        else:
            fallback = PureNumpyGaussianHMM(n_states=3, max_iter=100, random_state=self.random_state)
            fallback.fit(X_scaled)
            self.model = fallback
            raw_means = fallback.means
            raw_trans = fallback.trans_mat

        # 2. Canonical State Alignment:
        # State 2 = Highest Volatility state (Volatile Shock)
        # Between remaining two:
        # State 0 = Trending / Directional (Higher trend dispersion / directional persistence)
        # State 1 = Choppy / Mean-Reverting (Lower trend dispersion / lower persistence)
        vol_levels = raw_means[:, 0]
        shock_idx = int(np.argmax(vol_levels))
        remaining = [i for i in [0, 1, 2] if i != shock_idx]

        disp_levels = [raw_means[i, 1] for i in remaining]
        if disp_levels[0] >= disp_levels[1]:
            trending_idx = remaining[0]
            choppy_idx = remaining[1]
        else:
            trending_idx = remaining[1]
            choppy_idx = remaining[0]

        # Map internal index -> canonical state:
        # trending_idx -> 0 (TRENDING)
        # choppy_idx   -> 1 (CHOPPY)
        # shock_idx    -> 2 (VOLATILE_SHOCK)
        self.state_map = {
            trending_idx: int(RegimeState.TRENDING),
            choppy_idx: int(RegimeState.CHOPPY),
            shock_idx: int(RegimeState.VOLATILE_SHOCK)
        }

        # Build canonical transition matrix A[canonical_i, canonical_j]
        canonical_trans = np.zeros((3, 3))
        for i_raw in range(3):
            for j_raw in range(3):
                c_i = self.state_map[i_raw]
                c_j = self.state_map[j_raw]
                canonical_trans[c_i, c_j] = raw_trans[i_raw, j_raw]

        # Row-normalize
        canonical_trans /= np.sum(canonical_trans, axis=1, keepdims=True)
        self.transition_matrix = canonical_trans

        # Canonicalize state means in original feature scale
        canonical_means = np.zeros((3, 2))
        for raw_k, can_k in self.state_map.items():
            canonical_means[can_k] = raw_means[raw_k] * self.feature_stds + self.feature_means
        self.state_means = canonical_means
        self.is_fitted = True

        return {
            "transition_matrix": self.transition_matrix,
            "state_means": self.state_means,
            "state_covariances": self.state_covariances
        }

    def predict_probabilities(self, panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Returns daily posterior probabilities [P(State 0), P(State 1), P(State 2)] for each date."""
        if not self.is_fitted:
            self.fit(panel_df)

        feat_df, X_raw = self.extract_features(panel_df)
        X_scaled = (X_raw - self.feature_means) / self.feature_stds

        if hasattr(self.model, "predict_proba"):
            raw_posteriors = self.model.predict_proba(X_scaled)
        else:
            raw_posteriors = self.model.predict_proba(X_scaled)

        # Remap columns to canonical order: State 0, State 1, State 2
        canonical_posteriors = np.zeros_like(raw_posteriors)
        for raw_k, can_k in self.state_map.items():
            canonical_posteriors[:, can_k] = raw_posteriors[:, raw_k]

        # Normalize rows to sum to 1.0
        canonical_posteriors /= np.sum(canonical_posteriors, axis=1, keepdims=True)
        
        feat_df['P_State0_Trending'] = canonical_posteriors[:, 0]
        feat_df['P_State1_Choppy'] = canonical_posteriors[:, 1]
        feat_df['P_State2_Shock'] = canonical_posteriors[:, 2]
        feat_df['Predicted_Regime'] = np.argmax(canonical_posteriors, axis=1)

        return feat_df, canonical_posteriors

    def get_current_regime(self, panel_df: pd.DataFrame) -> RegimeInference:
        """Computes current real-time regime inference and conditioned alpha allocations."""
        feat_df, posteriors = self.predict_probabilities(panel_df)
        last_row = feat_df.iloc[-1]
        last_posteriors = posteriors[-1]

        p0 = float(last_posteriors[0])
        p1 = float(last_posteriors[1])
        p2 = float(last_posteriors[2])

        current_state = int(np.argmax(last_posteriors))
        state_name = STATE_NAMES[RegimeState(current_state)]
        desc = STATE_DESCRIPTIONS[RegimeState(current_state)]

        # Dynamic Factor Scaling:
        # State 0 (Trending): Momentum 1.5x, Mean-Reversion 0.5x, Gross 1.0
        # State 1 (Choppy):   Mean-Reversion 1.5x, Momentum 0.5x, Gross 1.0
        # State 2 (Shock):    Momentum 1.0x, Mean-Reversion 1.0x, Gross 0.55 (-45% reduction)
        mom_mult = (p0 * 1.5) + (p1 * 0.5) + (p2 * 1.0)
        mr_mult  = (p0 * 0.5) + (p1 * 1.5) + (p2 * 1.0)
        gross_leverage = (p0 * 1.0) + (p1 * 1.0) + (p2 * 0.55)

        factor_tilts = {
            "Momentum_Breakout_Multiplier": round(mom_mult, 3),
            "Mean_Reversion_Quality_Multiplier": round(mr_mult, 3),
            "Gross_Portfolio_Leverage": round(gross_leverage, 3)
        }

        # Apply tilts to baseline risk-parity factor weights
        conditioned_weights = {}
        for aid, base_w in BASE_ALPHA_WEIGHTS.items():
            if aid in MOMENTUM_FAMILY:
                tilt = mom_mult
            elif aid in MEAN_REVERSION_FAMILY:
                tilt = mr_mult
            else:
                tilt = 1.0
            conditioned_weights[aid] = base_w * tilt

        # Normalize weights so sum equals gross_leverage
        tot_raw = sum(conditioned_weights.values())
        if tot_raw > 0:
            scale = gross_leverage / tot_raw
            for aid in conditioned_weights:
                conditioned_weights[aid] = round(conditioned_weights[aid] * scale, 5)

        trans_list = [[round(float(val), 4) for val in row] for row in self.transition_matrix]

        return RegimeInference(
            timestamp=pd.to_datetime(last_row['date']).strftime('%Y-%m-%d'),
            current_state=current_state,
            state_name=state_name,
            description=desc,
            probabilities={
                "State_0_Trending": round(p0, 4),
                "State_1_Choppy": round(p1, 4),
                "State_2_Volatile_Shock": round(p2, 4)
            },
            factor_tilts=factor_tilts,
            conditioned_alpha_weights=conditioned_weights,
            target_gross_leverage=round(gross_leverage, 3),
            transition_matrix=trans_list,
            realized_vol_20d=round(float(last_row['realized_vol_20d']), 4),
            trend_dispersion_20d=round(float(last_row['trend_dispersion_20d']), 4)
        )

    def get_latest_regime(self, panel_df: pd.DataFrame) -> Dict[str, Any]:
        """Convenience method returning dictionary telemetry for latest regime."""
        inf = self.get_current_regime(panel_df)
        p0 = inf.probabilities.get("State_0_Trending", 0.33)
        p1 = inf.probabilities.get("State_1_Choppy", 0.33)
        p2 = inf.probabilities.get("State_2_Volatile_Shock", 0.33)
        mom_mult = inf.factor_tilts.get("Momentum_Breakout_Multiplier", 1.0)
        mr_mult = inf.factor_tilts.get("Mean_Reversion_Quality_Multiplier", 1.0)
        return {
            "state": inf.current_state,
            "state_name": inf.state_name,
            "probabilities": [p0, p1, p2],
            "realized_vol": inf.realized_vol_20d,
            "trend_dispersion": inf.trend_dispersion_20d,
            "factor_tilts": {"Momentum": mom_mult, "MeanReversion": mr_mult},
            "gross_leverage": inf.target_gross_leverage,
            "transition_matrix": self.transition_matrix,
            "state_means": self.state_means,
            "state_covariances": self.state_covariances,
            "inference": inf
        }
