"""
Heston Model Calibration with Accelerated Optimization
=======================================================
A comprehensive class for Heston model calibration and risk-neutral PDF derivation.

This module provides:
    - Vectorized characteristic functions (Heston + Bates jump extension)
    - Optimized option pricing via damped Fourier integration
    - Custom parallel differential evolution optimizer
    - PDF extraction via Breeden-Litzenberger and FFT methods

Uses NumPy for complex arithmetic (reliable) with parallel evaluation via
joblib/multiprocessing for the optimizer population.

Author: Auto-generated from hc3.py patterns
"""

import logging
import traceback
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

logger = logging.getLogger(__name__)

# =============================================================================
# Numba-Accelerated Functions (Experimental - disabled by default)
# =============================================================================

# Numba JIT compilation is disabled by default due to potential stability issues
# (segfaults on some systems). Set NUMBA_ENABLED = True to enable.
NUMBA_ENABLED = False  # Set to True to enable Numba acceleration

NUMBA_AVAILABLE = False
if NUMBA_ENABLED:
    try:
        from numba import njit, prange, complex128, float64, int64
        NUMBA_AVAILABLE = True
        logger.info("Numba enabled - using JIT-compiled pricing functions")
    except ImportError:
        logger.debug("Numba not installed")

if not NUMBA_AVAILABLE:
    # Create dummy decorators when Numba is disabled or unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


@njit(cache=True)
def _numba_complex_sqrt(z_real: float, z_imag: float) -> Tuple[float, float]:
    """Complex square root for Numba."""
    r = np.sqrt(z_real**2 + z_imag**2)
    if r < 1e-14:
        return 0.0, 0.0
    # sqrt(z) = sqrt((r + Re(z))/2) + i*sign(Im(z))*sqrt((r - Re(z))/2)
    sqrt_real = np.sqrt((r + z_real) / 2)
    sqrt_imag = np.sqrt((r - z_real) / 2)
    if z_imag < 0:
        sqrt_imag = -sqrt_imag
    return sqrt_real, sqrt_imag


@njit(cache=True)
def _numba_complex_exp(z_real: float, z_imag: float) -> Tuple[float, float]:
    """Complex exponential for Numba."""
    exp_real_part = np.exp(z_real)
    return exp_real_part * np.cos(z_imag), exp_real_part * np.sin(z_imag)


@njit(cache=True)
def _numba_complex_log(z_real: float, z_imag: float) -> Tuple[float, float]:
    """Complex logarithm for Numba."""
    r = np.sqrt(z_real**2 + z_imag**2)
    if r < 1e-14:
        return -30.0, 0.0  # Approximate log(small)
    return np.log(r), np.arctan2(z_imag, z_real)


@njit(cache=True)
def _numba_complex_div(a_r: float, a_i: float, b_r: float, b_i: float) -> Tuple[float, float]:
    """Complex division for Numba: (a_r + i*a_i) / (b_r + i*b_i)."""
    denom = b_r**2 + b_i**2
    if denom < 1e-28:
        denom = 1e-28
    return (a_r * b_r + a_i * b_i) / denom, (a_i * b_r - a_r * b_i) / denom


@njit(cache=True)
def _numba_heston_charfunc_pj(
    phi: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    tau: float,
    r: float,
    q: float,
    j: int
) -> Tuple[float, float]:
    """
    Numba-optimized Heston characteristic function for single phi value.
    Returns (real, imag) parts of CF.
    """
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa
    
    a = kappa * theta
    
    # rho*sigma*i*phi - b = (-b, rho*sigma*phi)
    rs_phi_r = -b
    rs_phi_i = rho * sigma * phi
    
    # d^2 = (rho*sigma*i*phi - b)^2 - sigma^2*(2*u*i*phi - phi^2)
    # (rho*sigma*i*phi - b)^2 = (rs_phi_r + i*rs_phi_i)^2
    d2_r = rs_phi_r**2 - rs_phi_i**2 - sigma**2 * (-phi**2)
    d2_i = 2 * rs_phi_r * rs_phi_i - sigma**2 * (2 * u * phi)
    
    # d = sqrt(d^2)
    d_r, d_i = _numba_complex_sqrt(d2_r, d2_i)
    
    # g_numer = b - rho*sigma*i*phi + d = (b + d_r, -rho*sigma*phi + d_i)
    g_num_r = b + d_r
    g_num_i = -rho * sigma * phi + d_i
    
    # g_denom = b - rho*sigma*i*phi - d = (b - d_r, -rho*sigma*phi - d_i)
    g_den_r = b - d_r
    g_den_i = -rho * sigma * phi - d_i
    
    # g = g_numer / g_denom
    g_r, g_i = _numba_complex_div(g_num_r, g_num_i, g_den_r, g_den_i)
    
    # exp(d * tau)
    exp_d_tau_r, exp_d_tau_i = _numba_complex_exp(d_r * tau, d_i * tau)
    
    # g * exp(d*tau)
    g_exp_r = g_r * exp_d_tau_r - g_i * exp_d_tau_i
    g_exp_i = g_r * exp_d_tau_i + g_i * exp_d_tau_r
    
    # 1 - g * exp(d*tau)
    one_minus_gexp_r = 1.0 - g_exp_r
    one_minus_gexp_i = -g_exp_i
    
    # 1 - g
    one_minus_g_r = 1.0 - g_r
    one_minus_g_i = -g_i
    
    # log((1 - g*exp(d*tau)) / (1 - g))
    log_num_r, log_num_i = _numba_complex_log(one_minus_gexp_r, one_minus_gexp_i)
    log_den_r, log_den_i = _numba_complex_log(one_minus_g_r, one_minus_g_i)
    log_ratio_r = log_num_r - log_den_r
    log_ratio_i = log_num_i - log_den_i
    
    # C = (r-q)*i*phi*tau + (a/sigma^2) * ((b - rho*sigma*i*phi + d)*tau - 2*log(...))
    drift_r = 0.0
    drift_i = (r - q) * phi * tau
    
    coeff = a / (sigma**2)
    inner_r = g_num_r * tau - 2 * log_ratio_r
    inner_i = g_num_i * tau - 2 * log_ratio_i
    
    C_r = drift_r + coeff * inner_r
    C_i = drift_i + coeff * inner_i
    
    # D = ((b - rho*sigma*i*phi + d) / sigma^2) * (1 - exp(d*tau)) / (1 - g*exp(d*tau))
    one_minus_exp_r = 1.0 - exp_d_tau_r
    one_minus_exp_i = -exp_d_tau_i
    
    # (1 - exp) / (1 - g*exp)
    frac_r, frac_i = _numba_complex_div(one_minus_exp_r, one_minus_exp_i, 
                                         one_minus_gexp_r, one_minus_gexp_i)
    
    # g_num / sigma^2 * frac
    D_r = (g_num_r * frac_r - g_num_i * frac_i) / (sigma**2)
    D_i = (g_num_r * frac_i + g_num_i * frac_r) / (sigma**2)
    
    # exp(C + D*v0)
    exp_arg_r = C_r + D_r * v0
    exp_arg_i = C_i + D_i * v0
    
    char_r, char_i = _numba_complex_exp(exp_arg_r, exp_arg_i)
    
    # Check for NaN/Inf
    if np.isnan(char_r) or np.isnan(char_i) or np.isinf(char_r) or np.isinf(char_i):
        return 0.0, 0.0
    
    return char_r, char_i


@njit(cache=True, parallel=True)
def _numba_heston_prices(
    S0: float,
    K: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    tau: float,
    r: float,
    q: float,
    N: int,
    umax: float
) -> np.ndarray:
    """
    Numba-parallelized Heston call pricing.
    """
    n_strikes = len(K)
    prices = np.zeros(n_strikes)
    
    dphi = umax / N
    F = S0 * np.exp((r - q) * tau)
    
    for k_idx in prange(n_strikes):
        K_k = K[k_idx]
        log_FK = np.log(F / K_k)
        
        # Integrate P1 and P2
        int1_sum = 0.0
        int2_sum = 0.0
        
        for i in range(1, N + 1):
            phi = i * dphi
            
            # P1 characteristic function
            cf1_r, cf1_i = _numba_heston_charfunc_pj(
                phi, v0, kappa, theta, xi, rho, tau, r, q, 1
            )
            
            # P2 characteristic function
            cf2_r, cf2_i = _numba_heston_charfunc_pj(
                phi, v0, kappa, theta, xi, rho, tau, r, q, 2
            )
            
            # exp(i*phi*log(F/K))
            cos_phi = np.cos(phi * log_FK)
            sin_phi = np.sin(phi * log_FK)
            
            # exp_factor * cf / (i*phi)
            # cf / (i*phi) = (cf_r + i*cf_i) / (i*phi) = (cf_i/phi) + i*(-cf_r/phi)
            # Real part of exp_factor * (cf/(i*phi)):
            # Re[(cos + i*sin) * (cf_i/phi - i*cf_r/phi)]
            # = cos*cf_i/phi + sin*cf_r/phi
            
            int1_sum += (cos_phi * cf1_i + sin_phi * cf1_r) / phi
            int2_sum += (cos_phi * cf2_i + sin_phi * cf2_r) / phi
        
        P1 = 0.5 + int1_sum * dphi / np.pi
        P2 = 0.5 + int2_sum * dphi / np.pi
        
        # Clip probabilities
        P1 = max(0.0, min(1.0, P1))
        P2 = max(0.0, min(1.0, P2))
        
        price = S0 * np.exp(-q * tau) * P1 - K_k * np.exp(-r * tau) * P2
        prices[k_idx] = max(0.0, price)
    
    return prices


@njit(cache=True)
def _numba_bs_price(S0: float, K: float, tau: float, r: float, sigma: float, q: float) -> float:
    """Numba Black-Scholes call price."""
    if sigma <= 0 or tau <= 0:
        return max(0.0, S0 * np.exp(-q * tau) - K * np.exp(-r * tau))
    
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    
    # Approximate normal CDF using error function
    def norm_cdf(x):
        return 0.5 * (1.0 + np.tanh(x * 0.7978845608))  # Fast approximation
    
    return S0 * np.exp(-q * tau) * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)


@njit(cache=True)
def _numba_iv_newton(price: float, S0: float, K: float, tau: float, r: float, q: float) -> float:
    """Numba Newton-Raphson IV inversion."""
    sigma = 0.3
    
    for _ in range(30):
        bs_price = _numba_bs_price(S0, K, tau, r, sigma, q)
        
        # Vega
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        vega = S0 * np.exp(-q * tau) * sqrt_tau * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        
        if vega < 1e-10:
            break
        
        diff = bs_price - price
        if abs(diff) < 1e-8:
            break
        
        sigma = sigma - diff / vega
        sigma = max(0.01, min(5.0, sigma))
    
    return sigma


@njit(cache=True, parallel=True)
def _numba_compute_ivs(
    prices: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    tau: float,
    r: float,
    q: float
) -> np.ndarray:
    """Numba-parallelized IV computation."""
    n = len(prices)
    ivs = np.full(n, np.nan)
    
    for i in prange(n):
        if prices[i] > 0:
            ivs[i] = _numba_iv_newton(prices[i], S0, strikes[i], tau, r, q)
    
    return ivs


@njit(cache=True)
def _numba_objective(
    params: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    tau: float,
    r: float,
    q: float,
    N: int,
    umax: float,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    bounds: np.ndarray
) -> float:
    """Numba objective function for calibration."""
    kappa = max(bounds[0, 0], min(bounds[0, 1], params[0]))
    theta = max(bounds[1, 0], min(bounds[1, 1], params[1]))
    v0 = max(bounds[2, 0], min(bounds[2, 1], params[2]))
    rho = max(bounds[3, 0], min(bounds[3, 1], params[3]))
    xi = max(bounds[4, 0], min(bounds[4, 1], params[4]))
    
    if v0 <= 0 or theta <= 0 or xi <= 0 or kappa <= 0:
        return 1e12
    
    # Compute prices
    prices = _numba_heston_prices(S0, strikes, v0, kappa, theta, xi, rho, tau, r, q, N, umax)
    
    # Compute IVs
    model_ivs = _numba_compute_ivs(prices, S0, strikes, tau, r, q)
    
    # Compute weighted RMSE
    sum_weighted_sq = 0.0
    sum_weights = 0.0
    
    for i in range(len(strikes)):
        if valid_mask[i] and not np.isnan(model_ivs[i]) and not np.isnan(market_ivs[i]):
            diff = model_ivs[i] - market_ivs[i]
            sum_weighted_sq += weights[i] * diff * diff
            sum_weights += weights[i]
    
    if sum_weights < 1e-10:
        return 1e12
    
    return np.sqrt(sum_weighted_sq / sum_weights)


@njit(cache=True)
def _numba_select_indices(i: int, pop_size: int) -> Tuple[int, int, int]:
    """Select 3 distinct random indices different from i."""
    indices = np.arange(pop_size)
    # Swap i to end
    indices[i], indices[pop_size-1] = indices[pop_size-1], indices[i]
    # Shuffle first pop_size-1 elements (Fisher-Yates partial)
    for k in range(min(3, pop_size-1)):
        j = k + int(np.random.random() * (pop_size - 1 - k))
        indices[k], indices[j] = indices[j], indices[k]
    return int(indices[0]), int(indices[1]), int(indices[2])


@njit(cache=True, parallel=True)
def _numba_de_iteration(
    population: np.ndarray,
    fitness: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    tau: float,
    r: float,
    q: float,
    N: int,
    umax: float,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    bounds: np.ndarray,
    F_mut: float,
    CR: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Single DE iteration with parallel trial evaluation."""
    pop_size, n_params = population.shape
    trial_pop = np.zeros_like(population)
    trial_fitness = np.zeros(pop_size)
    
    for i in prange(pop_size):
        # Select 3 random distinct indices
        r1, r2, r3 = _numba_select_indices(i, pop_size)
        
        # Mutation and crossover
        j_rand = int(np.random.random() * n_params)
        for j in range(n_params):
            if np.random.random() < CR or j == j_rand:
                trial_pop[i, j] = population[r1, j] + F_mut * (population[r2, j] - population[r3, j])
                # Clip to bounds
                trial_pop[i, j] = max(bounds[j, 0], min(bounds[j, 1], trial_pop[i, j]))
            else:
                trial_pop[i, j] = population[i, j]
        
        # Evaluate trial
        trial_fitness[i] = _numba_objective(
            trial_pop[i], S0, strikes, market_ivs, tau, r, q,
            N, umax, weights, valid_mask, bounds
        )
    
    # Selection
    new_pop = np.zeros_like(population)
    new_fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if trial_fitness[i] < fitness[i]:
            new_pop[i] = trial_pop[i]
            new_fitness[i] = trial_fitness[i]
        else:
            new_pop[i] = population[i]
            new_fitness[i] = fitness[i]
    
    return new_pop, new_fitness


# =============================================================================
# Data Classes for Clean Parameter Handling
# =============================================================================

@dataclass
class HestonParams:
    """Container for Heston model parameters."""
    v0: float       # Initial variance
    kappa: float    # Mean reversion speed
    theta: float    # Long-term variance
    xi: float       # Volatility of volatility
    rho: float      # Correlation between asset and variance
    lambd_sv: float = 0.0  # Risk premium (usually 0 for risk-neutral)


@dataclass
class BatesParams(HestonParams):
    """Container for Heston-Bates model parameters (with jumps)."""
    lambda_j: float = 0.0   # Jump intensity (jumps per year)
    mu_j: float = 0.0       # Mean jump size (log-normal)
    sigma_j: float = 0.0    # Jump volatility


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    params: BatesParams
    rmse: float
    success: bool
    iterations: int
    message: str = ""


# =============================================================================
# Vectorized Characteristic Functions (NumPy-based, from hc3.py)
# =============================================================================

def heston_bates_charfunc(
    phi: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambd_sv: float,
    tau: float,
    r: float,
    q: float = 0.0,
    lambda_j: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0
) -> np.ndarray:
    """
    Stable 'little trap' Heston-Bates Characteristic Function (vectorized).
    
    Parameters
    ----------
    phi : np.ndarray
        Frequency array (can be complex for damped integration)
    v0, kappa, theta, xi, rho : float
        Heston parameters
    lambd_sv : float
        Stochastic volatility risk premium
    tau : float
        Time to maturity
    r, q : float
        Risk-free rate and dividend yield
    lambda_j, mu_j, sigma_j : float
        Bates jump parameters
    
    Returns
    -------
    np.ndarray (complex)
        Characteristic function values
    """
    phi = np.asarray(phi, dtype=complex).ravel()  # Ensure 1D

    i = 1j
    a = kappa * theta
    u = i * phi + phi**2
    b = kappa + lambd_sv
    rspi = rho * xi * phi * i

    tmp = (rspi - b)**2 + xi**2 * u
    d = np.sqrt(tmp)

    # Little-trap formulation
    denom = (b - rspi - d)
    denom = np.where(np.abs(denom) < 1e-14, 1e-14 * (1 + 1j), denom)
    g = (b - rspi + d) / denom

    g_exp = g * np.exp(-d * tau)
    term1 = 1 - g_exp
    term2 = 1 - g
    term2 = np.where(np.abs(term2) < 1e-14, 1e-14 * (1 + 1j), term2)
    pow_term = (term1 / term2) ** (-2 * a / xi**2)
    pow_term = np.where(np.isnan(pow_term) | np.isinf(pow_term.real), 1.0 + 0j, pow_term)

    one_minus_g_exp = 1 - g_exp
    one_minus_g_exp = np.where(np.abs(one_minus_g_exp) < 1e-14, 1e-14 * (1 + 1j), one_minus_g_exp)
    exp_arg = (a / xi**2) * (b - rspi + d) * tau + \
              (v0 / xi**2) * (b - rspi + d) * (1 - np.exp(-d * tau)) / one_minus_g_exp
    exp_arg = np.clip(exp_arg.real, -50, 50) + 1j * np.clip(exp_arg.imag, -50, 50)

    char_sv = np.exp(r * phi * i * tau) * pow_term * np.exp(exp_arg)

    # Bates Jump Component
    jump_term = np.exp(lambda_j * tau * (np.exp(i * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1))

    char = char_sv * jump_term
    char = np.where(np.isnan(char) | np.isinf(char.real) | np.isinf(char.imag), 0j, char)
    
    return char


# =============================================================================
# Option Pricing - Gatheral Formulation with Correct P1/P2
# =============================================================================

def _heston_charfunc_pj(
    phi: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,  # vol of vol (xi)
    rho: float,
    tau: float,
    r: float,
    q: float,
    j: int,  # j=1 for P1 (stock measure), j=2 for P2 (risk-neutral)
    lambda_j: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0
) -> np.ndarray:
    """
    Heston characteristic function for P1/P2 (log-return formulation).
    
    Based on Gatheral's "The Volatility Surface" and Albrecher et al.
    
    This computes the CF of the log-return X = log(S_T/S_0), NOT log(S_T).
    The drift term (r-q)*i*phi*tau is included.
    
    Parameters
    ----------
    phi : np.ndarray
        Integration variable (real)
    j : int
        1 for P1 (delta/stock measure), 2 for P2 (bond/risk-neutral measure)
    
    Key differences between P1 and P2:
    - P1: u = 0.5, b = kappa - rho*sigma
    - P2: u = -0.5, b = kappa
    """
    phi = np.asarray(phi, dtype=complex).ravel()
    
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa
    
    a = kappa * theta
    
    # d^2 = (rho*sigma*i*phi - b)^2 - sigma^2*(2*u*i*phi - phi^2)
    d = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
    
    # g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d)
    g_numer = b - rho * sigma * 1j * phi + d
    g_denom = b - rho * sigma * 1j * phi - d
    g_denom = np.where(np.abs(g_denom) < 1e-14, 1e-14, g_denom)
    g = g_numer / g_denom
    
    exp_d_tau = np.exp(d * tau)
    
    # C coefficient
    C = (r - q) * 1j * phi * tau + (a / sigma**2) * (
        (b - rho * sigma * 1j * phi + d) * tau - 
        2 * np.log((1 - g * exp_d_tau) / (1 - g + 1e-14))
    )
    
    # D coefficient
    D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * (
        (1 - exp_d_tau) / (1 - g * exp_d_tau + 1e-14)
    )
    
    # Characteristic function (no log(S0) term - that's handled in integration)
    char_sv = np.exp(C + D * v0)
    
    # Bates jump component
    if lambda_j > 0:
        jump_cf = np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2)
        jump_term = np.exp(lambda_j * tau * (jump_cf - 1))
        char_sv = char_sv * jump_term
    
    # Clean numerical issues
    char_sv = np.where(np.isnan(char_sv) | np.isinf(char_sv.real) | np.isinf(char_sv.imag), 0j, char_sv)
    
    return char_sv


def heston_call_price_damped(
    S0: float,
    K: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambd_sv: float,
    tau: float,
    r: float,
    q: float = 0.0,
    alpha: float = 0.75,
    N: int = 4096,
    umax: float = 200.0,
    lambda_j: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0
) -> np.ndarray:
    """
    Heston-Bates call pricing using Gil-Pelaez inversion.
    
    C(K) = S0*e^{-q*tau}*P1 - K*e^{-r*tau}*P2
    
    where P_j = 0.5 + (1/pi) * integral Re[e^{i*phi*log(F/K)} * char_j(phi) / (i*phi)] dphi
    """
    K = np.atleast_1d(K).astype(float)
    n_strikes = len(K)
    
    dphi = umax / N
    phi = np.linspace(dphi, umax, N)
    
    F = S0 * np.exp((r - q) * tau)
    
    # Characteristic functions
    char1 = _heston_charfunc_pj(phi, v0, kappa, theta, xi, rho, tau, r, q, j=1,
                                 lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    char2 = _heston_charfunc_pj(phi, v0, kappa, theta, xi, rho, tau, r, q, j=2,
                                 lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    
    prices = np.zeros(n_strikes)
    
    for idx in range(n_strikes):
        k = K[idx]
        log_moneyness = np.log(F / k)
        
        exp_factor = np.exp(1j * phi * log_moneyness)
        
        integrand1 = np.real(exp_factor * char1 / (1j * phi))
        integrand2 = np.real(exp_factor * char2 / (1j * phi))
        
        integrand1 = np.nan_to_num(integrand1, nan=0.0, posinf=0.0, neginf=0.0)
        integrand2 = np.nan_to_num(integrand2, nan=0.0, posinf=0.0, neginf=0.0)
        
        P1 = 0.5 + np.trapezoid(integrand1, phi) / np.pi
        P2 = 0.5 + np.trapezoid(integrand2, phi) / np.pi
        
        P1 = np.clip(P1, 0.0, 1.0)
        P2 = np.clip(P2, 0.0, 1.0)
        
        call = S0 * np.exp(-q * tau) * P1 - k * np.exp(-r * tau) * P2
        prices[idx] = max(call, 0.0)
    
    return prices


def heston_call_price_vectorized(
    S0: float,
    K: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    tau: float,
    r: float,
    q: float = 0.0,
    N: int = 4096,
    umax: float = 200.0,
    lambda_j: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0
) -> np.ndarray:
    """
    Fully vectorized Heston pricing over strikes.
    """
    K = np.atleast_1d(K).astype(float)
    
    dphi = umax / N
    phi = np.linspace(dphi, umax, N)
    
    F = S0 * np.exp((r - q) * tau)
    
    char1 = _heston_charfunc_pj(phi, v0, kappa, theta, xi, rho, tau, r, q, j=1,
                                 lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    char2 = _heston_charfunc_pj(phi, v0, kappa, theta, xi, rho, tau, r, q, j=2,
                                 lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    
    log_moneyness = np.log(F / K)
    exp_factor = np.exp(1j * np.outer(phi, log_moneyness))
    
    integrand1 = np.real(exp_factor * (char1 / (1j * phi))[:, np.newaxis])
    integrand2 = np.real(exp_factor * (char2 / (1j * phi))[:, np.newaxis])
    
    integrand1 = np.nan_to_num(integrand1, nan=0.0, posinf=0.0, neginf=0.0)
    integrand2 = np.nan_to_num(integrand2, nan=0.0, posinf=0.0, neginf=0.0)
    
    P1 = 0.5 + np.trapezoid(integrand1, phi, axis=0) / np.pi
    P2 = 0.5 + np.trapezoid(integrand2, phi, axis=0) / np.pi
    
    P1 = np.clip(P1, 0.0, 1.0)
    P2 = np.clip(P2, 0.0, 1.0)
    
    prices = S0 * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
    return np.maximum(prices, 0.0)


# =============================================================================
# IV Computation (Vectorized)
# =============================================================================

def bs_price(S0, K, tau, r, sigma, q=0.0, is_call=True):
    """Vectorized Black-Scholes price."""
    S0, K, tau, r, sigma, q = [np.asarray(x) for x in [S0, K, tau, r, sigma, q]]
    is_call = np.asarray(is_call)
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call_price = S0 * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S0 * np.exp(-q * tau) * norm.cdf(-d1)
    
    return np.where(is_call, call_price, put_price)


def vectorized_bs_iv(prices, S0, K, tau, r, q=0.0, is_call=True, max_iv=5.0, tol=1e-6, max_iter=50):
    """Newton-Raphson IV inversion (vectorized)."""
    prices = np.asarray(prices)
    K = np.asarray(K)
    is_call = np.asarray(is_call) if isinstance(is_call, np.ndarray) else np.full_like(prices, is_call, dtype=bool)
    
    sigma = np.full_like(prices, 0.3)
    sqrt_tau = np.sqrt(tau)
    
    for _ in range(max_iter):
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        
        model_price = bs_price(S0, K, tau, r, sigma, q, is_call)
        vega = S0 * np.exp(-q * tau) * norm.pdf(d1) * sqrt_tau
        
        error = model_price - prices
        delta_sigma = np.clip(error / (vega + 1e-10), -0.5, 0.5)
        sigma -= delta_sigma
        
        if np.all(np.abs(delta_sigma) < tol):
            break
    
    sigma = np.clip(sigma, 1e-6, max_iv)
    sigma[vega < 1e-4] = np.nan
    return sigma


def compute_ivs(prices, S0, strikes, tau, r, q):
    """Compute IVs using OTM convention."""
    ivs = np.full_like(strikes, np.nan, dtype=float)
    
    mask_otm_call = strikes >= S0
    if np.any(mask_otm_call):
        ivs[mask_otm_call] = vectorized_bs_iv(
            prices[mask_otm_call], S0, strikes[mask_otm_call], tau, r, q, is_call=True
        )
    
    mask_otm_put = strikes < S0
    if np.any(mask_otm_put):
        put_prices = prices[mask_otm_put] - S0 * np.exp(-q * tau) + strikes[mask_otm_put] * np.exp(-r * tau)
        put_prices = np.maximum(put_prices, 0)
        ivs[mask_otm_put] = vectorized_bs_iv(
            put_prices, S0, strikes[mask_otm_put], tau, r, q, is_call=False
        )
    
    return ivs


# =============================================================================
# Parallel Differential Evolution Optimizer
# =============================================================================

def _evaluate_single(args):
    """Evaluate objective for a single parameter vector (for parallel execution)."""
    params, S0, strikes, market_ivs, tau, r, q, N, umax, weights, valid_mask, bounds = args
    
    kappa, theta, v0, rho, xi, lambda_j, mu_j, sigma_j = params
    
    # Clip to bounds
    kappa = np.clip(kappa, bounds[0][0], bounds[0][1])
    theta = np.clip(theta, bounds[1][0], bounds[1][1])
    v0 = np.clip(v0, bounds[2][0], bounds[2][1])
    rho = np.clip(rho, bounds[3][0], bounds[3][1])
    xi = np.clip(xi, bounds[4][0], bounds[4][1])
    lambda_j = np.clip(lambda_j, bounds[5][0], bounds[5][1])
    mu_j = np.clip(mu_j, bounds[6][0], bounds[6][1])
    sigma_j = np.clip(sigma_j, bounds[7][0], bounds[7][1])
    
    if v0 <= 0 or theta <= 0 or xi <= 0 or kappa <= 0:
        return 1e12
    
    try:
        model_prices = heston_call_price_vectorized(
            S0, strikes, v0, kappa, theta, xi, rho,
            tau, r, q, N, umax, lambda_j, mu_j, sigma_j
        )
        
        if not np.any(np.isfinite(model_prices)) or np.all(model_prices <= 0):
            return 1e12
        
        model_ivs = compute_ivs(model_prices, S0, strikes, tau, r, q)
        valid = valid_mask & np.isfinite(model_ivs) & np.isfinite(market_ivs)
        
        if not np.any(valid):
            return 1e12
        
        errors = (model_ivs[valid] - market_ivs[valid]) ** 2
        weighted_errors = weights[valid] * errors
        weighted_rmse = np.sqrt(np.sum(weighted_errors) / np.sum(weights[valid]))
        
        return weighted_rmse
        
    except Exception:
        return 1e12


def differential_evolution_parallel(
    S0: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    tau: float,
    r: float,
    q: float,
    N: int,
    umax: float,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    bounds: list,
    pop_size: int = 30,
    max_iter: int = 100,
    F: float = 0.7,
    CR: float = 0.85,
    seed: int = 42,
    n_workers: int = 4
) -> Tuple[np.ndarray, float, int]:
    """
    Parallel Differential Evolution optimizer.
    
    Uses ThreadPoolExecutor for parallel fitness evaluation.
    """
    np.random.seed(seed)
    n_params = len(bounds)
    bounds_arr = np.array(bounds)
    
    # Initialize population
    population = np.zeros((pop_size, n_params))
    for j in range(n_params):
        population[:, j] = bounds_arr[j, 0] + np.random.random(pop_size) * (bounds_arr[j, 1] - bounds_arr[j, 0])
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = _evaluate_single((
            population[i], S0, strikes, market_ivs, tau, r, q,
            N, umax, weights, valid_mask, bounds
        ))
    
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    logger.info(f"DE initial best fitness: {best_fitness:.6f}")
    
    # Main DE loop
    trial_pop = np.zeros((pop_size, n_params))
    
    for iteration in range(max_iter):
        # Generate trial vectors
        for i in range(pop_size):
            # Select 3 random distinct indices
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # Mutation and crossover
            j_rand = np.random.randint(n_params)
            for j in range(n_params):
                if np.random.random() < CR or j == j_rand:
                    trial_pop[i, j] = population[r1, j] + F * (population[r2, j] - population[r3, j])
                else:
                    trial_pop[i, j] = population[i, j]
            
            # Clip to bounds
            trial_pop[i] = np.clip(trial_pop[i], bounds_arr[:, 0], bounds_arr[:, 1])
        
        # Evaluate trials (can be parallelized)
        trial_fitness = np.zeros(pop_size)
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            args_list = [
                (trial_pop[i], S0, strikes, market_ivs, tau, r, q,
                 N, umax, weights, valid_mask, bounds)
                for i in range(pop_size)
            ]
            results = list(executor.map(_evaluate_single, args_list))
            trial_fitness = np.array(results)
        
        # Selection
        improved = trial_fitness < fitness
        population[improved] = trial_pop[improved]
        fitness[improved] = trial_fitness[improved]
        
        # Update best
        curr_best_idx = np.argmin(fitness)
        if fitness[curr_best_idx] < best_fitness:
            best_fitness = fitness[curr_best_idx]
            best_params = population[curr_best_idx].copy()
        
        # Progress logging
        if (iteration + 1) % 20 == 0:
            logger.info(f"DE iteration {iteration + 1}: best={best_fitness:.6f}")
        
        # Early stopping
        if best_fitness < 1e-4:
            return best_params, best_fitness, iteration + 1
    
    return best_params, best_fitness, max_iter


# =============================================================================
# Main Calibration Class
# =============================================================================

class HestonCalibrator:
    """
    Heston-Bates Model Calibrator with Parallel Optimization.
    
    This class provides calibration of the Heston stochastic volatility model
    with optional Bates jump extension using parallel differential evolution.
    
    Parameters
    ----------
    alpha : float
        Damping parameter for Fourier pricing (default: 0.75)
    N : int
        Number of integration points (default: 2048)
    umax : float
        Maximum frequency for integration (default: 500)
    
    Example
    -------
    >>> calibrator = HestonCalibrator()
    >>> result = calibrator.calibrate(strikes, market_prices, S0, tau, r)
    >>> pdf_strikes, pdf_values = calibrator.extract_pdf_fft(S0, result.params, tau, r)
    """
    
    DEFAULT_BOUNDS = [
        (0.1, 8.0),     # kappa
        (0.01, 0.2),    # theta
        (0.01, 0.15),   # v0
        (-0.95, -0.3),  # rho
        (0.3, 2.5),     # xi
        (0.0, 5.0),     # lambda_j
        (-0.15, 0.0),   # mu_j
        (0.01, 0.3)     # sigma_j
    ]
    
    def __init__(
        self,
        N: int = 2048,
        umax: float = 150.0
    ):
        """
        Initialize Heston calibrator.
        
        Parameters
        ----------
        N : int
            Number of integration points for Fourier pricing.
            Higher = more accurate but slower.
            Recommended: 1024-2048 for calibration, 4096 for PDF.
        umax : float
            Upper frequency limit for integration.
            Higher = captures more high-frequency features.
            Recommended: 100-200 depending on tau.
        
        Note: The old 'alpha' damping parameter is no longer needed.
        The Gil-Pelaez P1/P2 formulation is inherently stable.
        """
        self.N = N
        self.umax = umax
        self._last_market_ivs: Optional[np.ndarray] = None
        self._last_model_ivs: Optional[np.ndarray] = None
    
    @staticmethod
    def auto_select_params(
        tau: float,
        strikes: np.ndarray,
        S0: float
    ) -> Tuple[int, float]:
        """
        Auto-select optimal N and umax based on option characteristics.
        
        Parameters
        ----------
        tau : float
            Time to maturity in years
        strikes : np.ndarray
            Strike prices
        S0 : float
            Spot price
        
        Returns
        -------
        N, umax : int, float
            Recommended integration parameters
        """
        # Compute OTM ratio (max distance from spot)
        otm_ratio = np.max(np.abs(strikes - S0) / S0)
        
        # Base parameters by time to expiry
        if tau < 3/365:       # 0-3 days (0DTE territory)
            base_N = 4096
            base_umax = 250
        elif tau < 7/365:     # 3-7 days
            base_N = 2048
            base_umax = 200
        elif tau < 30/365:    # 1-4 weeks
            base_N = 2048
            base_umax = 150
        elif tau < 90/365:    # 1-3 months
            base_N = 1024
            base_umax = 120
        else:                 # 3+ months
            base_N = 1024
            base_umax = 100
        
        # Adjust for OTM depth
        if otm_ratio > 0.25:
            base_N = int(base_N * 1.5)
            base_umax *= 1.3
        elif otm_ratio > 0.15:
            base_N = int(base_N * 1.2)
            base_umax *= 1.1
        
        return base_N, base_umax
    
    def validate_params(
        self,
        S0: float,
        params: BatesParams,
        tau: float,
        r: float,
        strikes: np.ndarray,
        tolerance: float = 0.001
    ) -> Dict[str, Any]:
        """
        Validate that current N/umax are adequate by comparing to finer grid.
        
        Parameters
        ----------
        S0, params, tau, r, strikes : pricing inputs
        tolerance : float
            Max acceptable relative price difference (default 0.1%)
        
        Returns
        -------
        dict with:
            'adequate': bool - whether params are sufficient
            'max_diff': float - maximum relative difference
            'mean_diff': float - mean relative difference
            'recommendation': str - suggested action
        """
        # Compute prices with current settings
        prices_base = self.price_options(strikes, S0, params, tau, r)
        
        # Compute with doubled resolution
        N_fine = self.N * 2
        umax_fine = self.umax * 1.5
        
        prices_fine = heston_call_price_vectorized(
            S0, strikes, params.v0, params.kappa, params.theta,
            params.xi, params.rho, tau, r, q=0.0,
            N=N_fine, umax=umax_fine,
            lambda_j=params.lambda_j, mu_j=params.mu_j, sigma_j=params.sigma_j
        )
        
        # Compute relative differences (avoid division by zero)
        rel_diff = np.abs(prices_fine - prices_base) / (prices_base + 1e-8)
        max_diff = np.max(rel_diff)
        mean_diff = np.mean(rel_diff)
        
        adequate = max_diff < tolerance
        
        if adequate:
            recommendation = f"Current N={self.N}, umax={self.umax} are adequate."
        else:
            # Suggest increases
            new_N, new_umax = self.auto_select_params(tau, strikes, S0)
            new_N = max(new_N, int(self.N * 1.5))
            new_umax = max(new_umax, self.umax * 1.3)
            recommendation = f"Consider N={new_N}, umax={new_umax:.0f}"
        
        return {
            'adequate': adequate,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'recommendation': recommendation
        }
    
    def _compute_market_ivs(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        S0: float,
        tau: float,
        r: float,
        q: float
    ) -> np.ndarray:
        """Compute market IVs using OTM convention."""
        return compute_ivs(market_prices, S0, strikes, tau, r, q)
    
    def _compute_weights(
        self,
        strikes: np.ndarray,
        S0: float,
        weight_type: str = 'atm',
        flow_weights: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Compute calibration weights."""
        dist = np.abs(strikes - S0) / S0
        
        if weight_type == 'atm':
            weights = 1.0 / (dist + 0.01) ** 0.5
        elif weight_type == 'tail':
            weights = np.exp(dist * 2.0) + 0.3
        else:
            weights = np.ones_like(strikes)
        
        if flow_weights:
            flow_weight = np.zeros_like(weights)
            for key, values in flow_weights.items():
                if values is not None and len(values) == len(strikes):
                    flow_weight += np.abs(values)
            
            if np.max(flow_weight) > 1e-6:
                flow_weight = flow_weight / np.max(flow_weight) + 0.1
                weights *= flow_weight
        
        weights = weights / np.max(weights)
        return weights
    
    def calibrate(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        S0: float,
        tau: float,
        r: float,
        q: float = 0.0,
        weight_type: str = 'atm',
        flow_weights: Optional[Dict[str, np.ndarray]] = None,
        pop_size: int = 30,
        max_iter: int = 100,
        F: float = 0.7,
        CR: float = 0.85,
        bounds: Optional[list] = None,
        seed: Optional[int] = None,
        n_workers: int = 4
    ) -> CalibrationResult:
        """
        Calibrate Heston-Bates model to market data using parallel DE.
        
        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        market_prices : np.ndarray
            Market call prices
        S0 : float
            Current spot price
        tau : float
            Time to maturity (in years)
        r : float
            Risk-free rate
        q : float
            Dividend yield
        weight_type : str
            Weighting scheme: 'atm', 'tail', or 'uniform'
        flow_weights : dict, optional
            Dictionary of flow arrays {'gamma': arr, 'charm': arr, ...}
        pop_size : int
            Population size for differential evolution
        max_iter : int
            Maximum iterations
        F : float
            DE mutation factor
        CR : float
            DE crossover probability
        bounds : list, optional
            Parameter bounds override
        seed : int, optional
            Random seed
        n_workers : int
            Number of parallel workers
        
        Returns
        -------
        CalibrationResult
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        market_prices = np.asarray(market_prices, dtype=np.float64)
        
        # Compute market IVs
        market_ivs = self._compute_market_ivs(strikes, market_prices, S0, tau, r, q)
        self._last_market_ivs = market_ivs
        
        valid_mask = np.isfinite(market_ivs)
        
        if np.sum(valid_mask) < 3:
            logger.error("Not enough valid market IVs for calibration")
            return CalibrationResult(
                params=BatesParams(v0=0.04, kappa=2.0, theta=0.04, xi=1.0, rho=-0.7),
                rmse=np.inf,
                success=False,
                iterations=0,
                message="Insufficient valid market IVs"
            )
        
        weights = self._compute_weights(strikes, S0, weight_type, flow_weights)
        b = bounds if bounds else self.DEFAULT_BOUNDS
        
        if seed is None:
            seed = np.random.randint(0, 1000000)
        
        logger.info(f"Starting parallel DE calibration (pop={pop_size}, iter={max_iter})")
        
        try:
            best_params, best_rmse, iterations = differential_evolution_parallel(
                S0, strikes, market_ivs, tau, r, q,
                self.N, self.umax,
                weights, valid_mask, b,
                pop_size, max_iter, F, CR, seed, n_workers
            )
            
            # Compute final model IVs
            model_prices = heston_call_price_vectorized(
                S0, strikes, best_params[2], best_params[0], best_params[1],
                best_params[4], best_params[3], tau, r, q,
                self.N, self.umax,
                best_params[5], best_params[6], best_params[7]
            )
            self._last_model_ivs = compute_ivs(model_prices, S0, strikes, tau, r, q)
            
            params = BatesParams(
                v0=best_params[2],
                kappa=best_params[0],
                theta=best_params[1],
                xi=best_params[4],
                rho=best_params[3],
                lambda_j=best_params[5],
                mu_j=best_params[6],
                sigma_j=best_params[7]
            )
            
            logger.info(f"Calibration complete. RMSE={best_rmse:.6f}, iter={iterations}")
            
            return CalibrationResult(
                params=params,
                rmse=best_rmse,
                success=best_rmse < 1.0,
                iterations=iterations,
                message="Calibration successful"
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            logger.error(traceback.format_exc())
            return CalibrationResult(
                params=BatesParams(v0=0.04, kappa=2.0, theta=0.04, xi=1.0, rho=-0.7),
                rmse=np.inf,
                success=False,
                iterations=0,
                message=str(e)
            )
    
    def calibrate_numba(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        S0: float,
        tau: float,
        r: float,
        q: float = 0.0,
        weight_type: str = 'atm',
        flow_weights: Optional[Dict[str, np.ndarray]] = None,
        pop_size: int = 30,
        max_iter: int = 100,
        F: float = 0.7,
        CR: float = 0.85,
        bounds: Optional[list] = None,
        seed: Optional[int] = None
    ) -> CalibrationResult:
        """
        Numba-accelerated calibration (Heston only, no Bates jumps).
        
        This is significantly faster than the standard calibrate() method
        due to JIT-compiled pricing and parallel DE iterations.
        
        Note: This calibrates only the 5 core Heston parameters:
        (kappa, theta, v0, rho, xi). Jump parameters are set to zero.
        
        Requires Numba to be installed.
        """
        if not NUMBA_AVAILABLE:
            logger.warning("Numba not available, falling back to standard calibration")
            return self.calibrate(
                strikes, market_prices, S0, tau, r, q,
                weight_type, flow_weights, pop_size, max_iter, F, CR, bounds, seed
            )
        
        strikes = np.asarray(strikes, dtype=np.float64)
        market_prices = np.asarray(market_prices, dtype=np.float64)
        
        # Compute market IVs
        market_ivs = self._compute_market_ivs(strikes, market_prices, S0, tau, r, q)
        self._last_market_ivs = market_ivs
        
        valid_mask = np.isfinite(market_ivs).astype(np.float64)
        
        if np.sum(valid_mask) < 3:
            logger.error("Not enough valid market IVs for calibration")
            return CalibrationResult(
                params=BatesParams(v0=0.04, kappa=2.0, theta=0.04, xi=1.0, rho=-0.7),
                rmse=np.inf,
                success=False,
                iterations=0,
                message="Insufficient valid market IVs"
            )
        
        weights = self._compute_weights(strikes, S0, weight_type, flow_weights)
        
        # Heston-only bounds (5 params)
        heston_bounds = np.array([
            [0.05, 5.0],    # kappa
            [0.01, 0.15],   # theta
            [0.01, 0.15],   # v0
            [-0.95, -0.3],  # rho
            [0.3, 2.5],     # xi
        ])
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Starting Numba-accelerated DE calibration (pop={pop_size}, iter={max_iter})")
        
        try:
            # Initialize population
            n_params = 5
            population = np.zeros((pop_size, n_params))
            for j in range(n_params):
                population[:, j] = heston_bounds[j, 0] + np.random.random(pop_size) * (
                    heston_bounds[j, 1] - heston_bounds[j, 0]
                )
            
            # Initial fitness
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                fitness[i] = _numba_objective(
                    population[i], S0, strikes, market_ivs, tau, r, q,
                    self.N, self.umax, weights, valid_mask, heston_bounds
                )
            
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            logger.info(f"Numba DE initial best fitness: {best_fitness:.6f}")
            
            # Main DE loop using Numba
            for iteration in range(max_iter):
                population, fitness = _numba_de_iteration(
                    population, fitness, S0, strikes, market_ivs,
                    tau, r, q, self.N, self.umax, weights, valid_mask,
                    heston_bounds, F, CR
                )
                
                current_best = np.min(fitness)
                if current_best < best_fitness:
                    best_fitness = current_best
                
                if (iteration + 1) % 20 == 0:
                    logger.info(f"Numba DE iteration {iteration+1}: best={best_fitness:.6f}")
            
            # Get best solution
            best_idx = np.argmin(fitness)
            best_params = population[best_idx]
            
            # Compute final model prices/IVs
            model_prices = _numba_heston_prices(
                S0, strikes, best_params[2], best_params[0], best_params[1],
                best_params[4], best_params[3], tau, r, q, self.N, self.umax
            )
            self._last_model_ivs = compute_ivs(model_prices, S0, strikes, tau, r, q)
            
            params = BatesParams(
                v0=best_params[2],
                kappa=best_params[0],
                theta=best_params[1],
                xi=best_params[4],
                rho=best_params[3],
                lambda_j=0.0,
                mu_j=0.0,
                sigma_j=0.0
            )
            
            logger.info(f"Numba calibration complete. RMSE={best_fitness:.6f}, iter={max_iter}")
            
            return CalibrationResult(
                params=params,
                rmse=best_fitness,
                success=best_fitness < 1.0,
                iterations=max_iter,
                message="Numba calibration successful"
            )
            
        except Exception as e:
            logger.error(f"Numba calibration failed: {e}")
            logger.error(traceback.format_exc())
            logger.info("Falling back to standard calibration")
            return self.calibrate(
                strikes, market_prices, S0, tau, r, q,
                weight_type, flow_weights, pop_size, max_iter, F, CR, bounds, seed
            )
    
    def price_options_numba(
        self,
        strikes: np.ndarray,
        S0: float,
        params: BatesParams,
        tau: float,
        r: float,
        q: float = 0.0
    ) -> np.ndarray:
        """Price options using Numba-accelerated pricing (Heston only)."""
        if not NUMBA_AVAILABLE:
            return self.price_options(strikes, S0, params, tau, r, q)
        
        strikes = np.asarray(strikes, dtype=np.float64)
        return _numba_heston_prices(
            S0, strikes, params.v0, params.kappa, params.theta,
            params.xi, params.rho, tau, r, q, self.N, self.umax
        )
    
    def price_options(
        self,
        strikes: np.ndarray,
        S0: float,
        params: BatesParams,
        tau: float,
        r: float,
        q: float = 0.0
    ) -> np.ndarray:
        """Price options using calibrated parameters."""
        strikes = np.asarray(strikes, dtype=np.float64)
        
        return heston_call_price_vectorized(
            S0, strikes, params.v0, params.kappa, params.theta,
            params.xi, params.rho, tau, r, q,
            self.N, self.umax,
            params.lambda_j, params.mu_j, params.sigma_j
        )
    
    def extract_pdf_spline(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        r: float,
        tau: float,
        s: float = 0.05,
        k: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract risk-neutral PDF via Breeden-Litzenberger (spline-based)."""
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        prices = prices[sort_idx]
        
        spl = UnivariateSpline(strikes, prices, s=s, k=k)
        
        fine_strikes = np.linspace(strikes.min(), strikes.max(), 1000)
        second_deriv = spl.derivative(n=2)(fine_strikes)
        
        pdf = np.exp(r * tau) * second_deriv
        pdf = np.maximum(pdf, 0)
        
        integral = np.trapezoid(pdf, fine_strikes)
        if integral > 0:
            pdf /= integral
        
        return fine_strikes, pdf
    
    def extract_pdf_fft(
        self,
        S0: float,
        params: BatesParams,
        tau: float,
        r: float,
        q: float = 0.0,
        N: int = 4096,
        eta: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract risk-neutral PDF of S_T via FFT of characteristic function.
        
        Uses direct Fourier inversion:
        f_X(x) = (1/2π) ∫_{-∞}^{∞} exp(-i*u*x) * φ_X(u) du
        
        Since φ(-u) = conj(φ(u)) for real random variables:
        f_X(x) = (1/π) ∫_0^{∞} Re[exp(-i*u*x) * φ_X(u)] du
        
        Parameters
        ----------
        S0 : float
            Spot price
        params : BatesParams
            Calibrated Heston-Bates parameters
        tau : float
            Time to maturity
        r, q : float
            Risk-free rate and dividend yield
        N : int
            Number of integration points
        eta : float
            Grid spacing in frequency domain
        
        Returns
        -------
        strikes : np.ndarray
            Strike prices
        pdf : np.ndarray
            Probability density values
        """
        # Characteristic function of log-return X = log(S_T/S_0)
        # under risk-neutral measure: E^Q[exp(i*u*X)]
        def heston_cf_logreturn(u):
            """CF of log-return (not log-price)."""
            u = np.asarray(u, dtype=complex)
            
            sigma = params.xi
            kappa = params.kappa
            theta = params.theta
            rho = params.rho
            v0 = params.v0
            
            a = kappa * theta
            b = kappa
            
            # d^2 = (rho*sigma*i*u - b)^2 - sigma^2*(i*u - u^2)
            d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (1j * u - u**2))
            
            # g
            g_numer = b - rho * sigma * 1j * u + d
            g_denom = b - rho * sigma * 1j * u - d
            g_denom = np.where(np.abs(g_denom) < 1e-14, 1e-14, g_denom)
            g = g_numer / g_denom
            
            exp_d_tau = np.exp(d * tau)
            
            # C coefficient (includes drift)
            C = (r - q) * 1j * u * tau + (a / sigma**2) * (
                (b - rho * sigma * 1j * u + d) * tau -
                2 * np.log((1 - g * exp_d_tau) / (1 - g + 1e-14))
            )
            
            # D coefficient
            D = ((b - rho * sigma * 1j * u + d) / sigma**2) * (
                (1 - exp_d_tau) / (1 - g * exp_d_tau + 1e-14)
            )
            
            char = np.exp(C + D * v0)
            
            # Bates jumps
            if params.lambda_j > 0:
                jump_cf = np.exp(1j * u * params.mu_j - 0.5 * params.sigma_j**2 * u**2)
                char = char * np.exp(params.lambda_j * tau * (jump_cf - 1))
            
            return np.where(np.isnan(char) | np.isinf(char), 0j, char)
        
        # Forward and log-forward
        F = S0 * np.exp((r - q) * tau)
        log_F = np.log(F)
        
        # Create grid of log-strikes centered around log(F)
        # x_j = log(K_j) - log(S_0), we want to evaluate PDF at various x values
        dx = 2 * np.pi / (N * eta)  # Spacing in x-space (log-return space)
        x_min = -N * dx / 2
        x = x_min + dx * np.arange(N)  # Log-returns
        
        # Frequency grid
        u = eta * np.arange(N)
        u[0] = 1e-10  # Avoid u=0
        
        # Compute characteristic function
        char = heston_cf_logreturn(u)
        char[0] = 1.0 + 0j
        
        # FFT to get density of log-return
        # f(x) = (1/2π) * sum_j φ(u_j) * exp(-i*u_j*x) * du
        # Using FFT: fft_input = φ(u) * exp(-i*u*x_min) * eta
        fft_input = char * np.exp(-1j * u * x_min) * eta
        fft_output = np.fft.fft(fft_input)
        
        # PDF of log-return
        pdf_logreturn = np.real(fft_output) / (2 * np.pi)
        
        # Convert log-return to strike: K = S_0 * exp(x), so strike PDF = pdf_x / K
        strikes = S0 * np.exp(x)
        pdf_strike = np.maximum(pdf_logreturn / strikes, 0)
        
        # Sort by strike
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        pdf_strike = pdf_strike[sort_idx]
        
        # Filter and normalize
        valid = (strikes > S0 * 0.5) & (strikes < S0 * 1.5) & np.isfinite(pdf_strike) & (pdf_strike > 0)
        
        if np.sum(valid) > 10:
            integral = np.trapezoid(pdf_strike[valid], strikes[valid])
            if integral > 1e-10:
                pdf_strike = pdf_strike / integral
        
        return strikes, pdf_strike
    
    def extract_pdf_from_prices(
        self,
        S0: float,
        params: BatesParams,
        tau: float,
        r: float,
        q: float = 0.0,
        n_strikes: int = 100,
        strike_range: Tuple[float, float] = (0.7, 1.3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract PDF by computing option prices and using Breeden-Litzenberger.
        
        This is often more reliable than direct FFT inversion.
        """
        # Generate strike grid
        K_min = S0 * strike_range[0]
        K_max = S0 * strike_range[1]
        strikes = np.linspace(K_min, K_max, n_strikes)
        
        # Compute call prices
        prices = heston_call_price_vectorized(
            S0, strikes, params.v0, params.kappa, params.theta, params.xi, params.rho,
            tau, r, q, N=self.N, umax=self.umax,
            lambda_j=params.lambda_j, mu_j=params.mu_j, sigma_j=params.sigma_j
        )
        
        # Use spline-based extraction
        return self.extract_pdf_spline(strikes, prices, r, tau, s=0.001, k=4)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information from last calibration."""
        return {
            'market_ivs': self._last_market_ivs,
            'model_ivs': self._last_model_ivs
        }


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import time
    import pandas as pd
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    
    print("=" * 60)
    print("Heston Calibrator Test")
    print("=" * 60)
    
    # Test with synthetic data first
    np.random.seed(42)
    S0 = 100.0
    strikes = np.linspace(85, 115, 31)
    tau = 0.15
    r = 0.04
    
    # Create synthetic smile
    true_iv = 0.22 + 0.12 * ((strikes - S0) / S0) ** 2
    market_prices = np.array([bs_price(S0, K, tau, r, iv, is_call=(K >= S0)) 
                               for K, iv in zip(strikes, true_iv)])
    
    print(f"S0={S0}, tau={tau:.3f}, r={r}")
    print(f"Strikes: {len(strikes)} from {strikes.min():.1f} to {strikes.max():.1f}")
    print(f"Price range: {market_prices.min():.4f} to {market_prices.max():.4f}")
    
    # Test calibrator
    calibrator = HestonCalibrator(alpha=0.75, N=1024, umax=200)
    
    print("\nStarting calibration...")
    start = time.time()
    result = calibrator.calibrate(
        strikes=strikes,
        market_prices=market_prices,
        S0=S0,
        tau=tau,
        r=r,
        q=0.0,
        pop_size=25,
        max_iter=60,
        seed=42,
        n_workers=4
    )
    elapsed = time.time() - start
    
    print(f"\nCalibration time: {elapsed:.2f}s")
    print(f"Success: {result.success}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"Iterations: {result.iterations}")
    print(f"\nOptimal Parameters:")
    print(f"  kappa={result.params.kappa:.4f}")
    print(f"  theta={result.params.theta:.4f}")
    print(f"  v0={result.params.v0:.4f}")
    print(f"  xi={result.params.xi:.4f}")
    print(f"  rho={result.params.rho:.4f}")
    print(f"  lambda_j={result.params.lambda_j:.4f}")
    print(f"  mu_j={result.params.mu_j:.4f}")
    print(f"  sigma_j={result.params.sigma_j:.4f}")
    
    # Verify fit
    model_prices = calibrator.price_options(strikes, S0, result.params, tau, r)
    price_rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
    print(f"\nPrice RMSE: {price_rmse:.6f}")
    
    diag = calibrator.get_diagnostics()
    if diag['market_ivs'] is not None and diag['model_ivs'] is not None:
        valid = np.isfinite(diag['market_ivs']) & np.isfinite(diag['model_ivs'])
        iv_rmse = np.sqrt(np.nanmean((diag['model_ivs'][valid] - diag['market_ivs'][valid])**2))
        print(f"IV RMSE: {iv_rmse:.6f}")
    
    # Test PDF
    print("\nExtracting PDF...")
    pdf_strikes, pdf_values = calibrator.extract_pdf_fft(S0, result.params, tau, r)
    valid = (pdf_strikes > 80) & (pdf_strikes < 120)
    integral = np.trapezoid(pdf_values[valid], pdf_strikes[valid])
    print(f"PDF integral: {integral:.4f}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


# =============================================================================
# Flow-Aware PDF Adjustment
# =============================================================================

@dataclass
class FlowData:
    """Container for flow/exposure data at each strike."""
    strikes: np.ndarray
    gex: Optional[np.ndarray] = None      # Gamma exposure
    vex: Optional[np.ndarray] = None      # Vanna exposure  
    cex: Optional[np.ndarray] = None      # Charm exposure
    oi_chg: Optional[np.ndarray] = None   # Open interest change
    volume: Optional[np.ndarray] = None   # Volume


class FlowAwarePDF:
    """
    Adjusts a base PDF using dealer flow/exposure data.
    
    This class implements flow-aware PDF modifications based on:
    - Gamma exposure (GEX): Pinning vs amplification
    - Vanna exposure (VEX): Vol-price feedback  
    - Charm exposure (CEX): Time decay effects
    - OI changes: Position building/unwinding
    
    Key insight: These flows can have SIGNED effects:
    - Positive GEX → pinning (concentrate density)
    - Negative GEX → amplification (fatten tails)
    - Positive OI_chg → building conviction (strengthen signal)
    - Negative OI_chg → unwinding (weaken signal)
    
    References
    ----------
    See methodology.md for theoretical background.
    """
    
    def __init__(
        self,
        alpha_gex: float = 0.3,
        alpha_vex: float = 0.2,
        alpha_cex: float = 0.15,
        alpha_oi: float = 0.1,
        mode_bandwidth: float = 0.05,
        tail_threshold: float = 0.10
    ):
        """
        Initialize flow-aware PDF adjuster.
        
        Parameters
        ----------
        alpha_gex : float
            Sensitivity to gamma exposure (mode shaping)
        alpha_vex : float
            Sensitivity to vanna exposure (skew/tail shaping)
        alpha_cex : float
            Sensitivity to charm exposure (short-term adjustments)
        alpha_oi : float
            Sensitivity to OI changes (confidence scaling)
        mode_bandwidth : float
            Fraction of S0 considered "near the mode" for GEX effects
        tail_threshold : float
            Fraction of S0 beyond which strikes are considered "tails"
        """
        self.alpha_gex = alpha_gex
        self.alpha_vex = alpha_vex
        self.alpha_cex = alpha_cex
        self.alpha_oi = alpha_oi
        self.mode_bandwidth = mode_bandwidth
        self.tail_threshold = tail_threshold
        
        self.logger = logging.getLogger(__name__ + '.FlowAwarePDF')
    
    def _normalize_exposure(
        self, 
        exposure: np.ndarray, 
        signed: bool = True
    ) -> np.ndarray:
        """
        Normalize exposure to [-1, 1] (signed) or [0, 1] (unsigned).
        """
        if exposure is None or len(exposure) == 0:
            return None
        
        exp = np.asarray(exposure, dtype=float)
        max_abs = np.nanmax(np.abs(exp))
        
        if max_abs < 1e-10:
            return np.zeros_like(exp)
        
        if signed:
            return np.clip(exp / max_abs, -1, 1)
        else:
            return np.clip(np.abs(exp) / max_abs, 0, 1)
    
    def _compute_gex_adjustment(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        gex: np.ndarray,
        oi_confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute GEX-based PDF adjustment.
        
        Positive GEX (dealers long gamma) → stabilizing → concentrate density
        Negative GEX (dealers short gamma) → amplifying → redistribute to tails
        
        Returns adjustment multiplier for each strike.
        """
        gex_norm = self._normalize_exposure(gex, signed=True)
        if gex_norm is None:
            return np.ones_like(pdf)
        
        # Distance from spot (normalized)
        dist = (strikes - S0) / S0
        
        # Mode region mask
        is_mode = np.abs(dist) < self.mode_bandwidth
        is_tail = np.abs(dist) > self.tail_threshold
        
        adjustment = np.ones_like(pdf)
        
        # Positive GEX in mode region → increase density (pinning)
        # Negative GEX in mode region → decrease density (breakout likely)
        mode_effect = np.where(is_mode, gex_norm * self.alpha_gex, 0)
        
        # In tails: negative GEX → fatten tails (amplification)
        # Positive GEX in tails → thin tails (stabilization reaches there)
        tail_effect = np.where(is_tail, -gex_norm * self.alpha_gex * 0.5, 0)
        
        adjustment = 1 + mode_effect + tail_effect
        
        # Apply OI confidence scaling if available
        if oi_confidence is not None:
            adjustment = 1 + (adjustment - 1) * oi_confidence
        
        return np.clip(adjustment, 0.3, 3.0)
    
    def _compute_vanna_adjustment(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        vex: np.ndarray
    ) -> np.ndarray:
        """
        Compute Vanna-based PDF adjustment.
        
        High vanna in OTM regions → vol-price feedback → fatten tails
        Vanna affects skew: asymmetric vanna → asymmetric tails
        """
        vex_norm = self._normalize_exposure(vex, signed=True)
        if vex_norm is None:
            return np.ones_like(pdf)
        
        dist = (strikes - S0) / S0
        is_left_tail = dist < -self.tail_threshold
        is_right_tail = dist > self.tail_threshold
        
        adjustment = np.ones_like(pdf)
        
        # Vanna in left tail (puts): positive vanna → vol spike causes selling → fatter left tail
        # Vanna in right tail (calls): interpretation depends on sign
        left_vanna = np.where(is_left_tail, np.abs(vex_norm) * self.alpha_vex, 0)
        right_vanna = np.where(is_right_tail, np.abs(vex_norm) * self.alpha_vex * 0.7, 0)
        
        adjustment = 1 + left_vanna + right_vanna
        
        return np.clip(adjustment, 0.5, 2.5)
    
    def _compute_charm_adjustment(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        cex: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """
        Compute Charm-based PDF adjustment.
        
        Charm effects are strongest for short-dated options (tau < 7 days).
        High charm → rapid delta decay → overnight risks
        """
        cex_norm = self._normalize_exposure(cex, signed=True)
        if cex_norm is None:
            return np.ones_like(pdf)
        
        # Charm effect scales inversely with time (strongest near expiry)
        time_scale = np.exp(-tau * 52)  # Decay with weeks to expiry
        
        # High |charm| in tails → overnight tail risk
        dist = (strikes - S0) / S0
        is_tail = np.abs(dist) > self.tail_threshold * 0.5  # Broader for charm
        
        tail_charm = np.where(is_tail, np.abs(cex_norm) * self.alpha_cex * time_scale, 0)
        adjustment = 1 + tail_charm
        
        return np.clip(adjustment, 0.8, 1.8)
    
    def _compute_oi_confidence(
        self,
        oi_chg: np.ndarray
    ) -> np.ndarray:
        """
        Compute confidence weight from OI changes.
        
        Positive OI_chg → building positions → higher confidence in other signals
        Negative OI_chg → unwinding → lower confidence
        
        Returns multiplier in [0.5, 1.5] range.
        """
        if oi_chg is None:
            return None
        
        oi_norm = self._normalize_exposure(oi_chg, signed=True)
        if oi_norm is None:
            return None
        
        # Map [-1, 1] to [0.5, 1.5]
        confidence = 1 + oi_norm * self.alpha_oi * 5
        return np.clip(confidence, 0.5, 1.5)
    
    def adjust_pdf(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        flows: FlowData,
        tau: float = 0.0,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply flow-based adjustments to a base PDF.
        
        Parameters
        ----------
        strikes : np.ndarray
            Strike prices corresponding to pdf values
        pdf : np.ndarray
            Base PDF values (e.g., from Heston FFT)
        S0 : float
            Current spot price
        flows : FlowData
            Flow/exposure data
        tau : float
            Time to expiry (for charm scaling)
        normalize : bool
            Whether to renormalize PDF to integrate to 1
        
        Returns
        -------
        pdf_adjusted : np.ndarray
            Flow-adjusted PDF
        components : dict
            Individual adjustment components for analysis
        """
        # Interpolate flow data to PDF strikes if needed
        if len(flows.strikes) != len(strikes):
            gex = np.interp(strikes, flows.strikes, flows.gex) if flows.gex is not None else None
            vex = np.interp(strikes, flows.strikes, flows.vex) if flows.vex is not None else None
            cex = np.interp(strikes, flows.strikes, flows.cex) if flows.cex is not None else None
            oi_chg = np.interp(strikes, flows.strikes, flows.oi_chg) if flows.oi_chg is not None else None
        else:
            gex = flows.gex
            vex = flows.vex
            cex = flows.cex
            oi_chg = flows.oi_chg
        
        # Compute OI confidence first (used by other adjustments)
        oi_confidence = self._compute_oi_confidence(oi_chg)
        
        # Compute individual adjustments
        adj_gex = self._compute_gex_adjustment(strikes, pdf, S0, gex, oi_confidence)
        adj_vex = self._compute_vanna_adjustment(strikes, pdf, S0, vex)
        adj_cex = self._compute_charm_adjustment(strikes, pdf, S0, cex, tau)
        
        # Combined adjustment (multiplicative)
        total_adjustment = adj_gex * adj_vex * adj_cex
        
        # Apply to PDF
        pdf_adjusted = pdf * total_adjustment
        
        # Ensure non-negative
        pdf_adjusted = np.maximum(pdf_adjusted, 0)
        
        # Renormalize
        if normalize:
            integral = np.trapezoid(pdf_adjusted, strikes)
            if integral > 1e-10:
                pdf_adjusted = pdf_adjusted / integral
        
        components = {
            'gex_adjustment': adj_gex,
            'vex_adjustment': adj_vex,
            'cex_adjustment': adj_cex,
            'total_adjustment': total_adjustment,
            'oi_confidence': oi_confidence if oi_confidence is not None else np.ones_like(strikes)
        }
        
        return pdf_adjusted, components
    
    def compute_flow_score(
        self,
        strikes: np.ndarray,
        flows: FlowData,
        S0: float
    ) -> np.ndarray:
        """
        Compute a composite flow score for each strike.
        
        This can be used for:
        - Identifying key strikes (high absolute score)
        - Understanding pressure direction (sign of score)
        - Weighting in calibration
        
        Returns
        -------
        score : np.ndarray
            Flow score in range [-1, 1] where:
            - Positive = net pinning/stabilizing pressure
            - Negative = net breakout/volatility pressure
        """
        gex = self._normalize_exposure(flows.gex, signed=True)
        vex = self._normalize_exposure(flows.vex, signed=True)
        cex = self._normalize_exposure(flows.cex, signed=True)
        oi_conf = self._compute_oi_confidence(flows.oi_chg)
        
        if oi_conf is None:
            oi_conf = np.ones(len(strikes))
        
        # Weighted combination
        score = np.zeros(len(strikes))
        
        if gex is not None:
            # GEX: positive = pinning (stabilizing)
            score += gex * self.alpha_gex * 2
        
        if vex is not None:
            # Vanna: high magnitude = vol sensitivity (destabilizing in tails)
            dist = np.abs((strikes - S0) / S0)
            is_tail = dist > self.tail_threshold
            score -= np.where(is_tail, np.abs(vex) * self.alpha_vex, 0)
        
        if cex is not None:
            # Charm: high magnitude = time decay risk
            score -= np.abs(cex) * self.alpha_cex * 0.5
        
        # Scale by OI confidence
        score = score * oi_conf
        
        # Normalize to [-1, 1]
        max_score = np.max(np.abs(score))
        if max_score > 1e-10:
            score = score / max_score
        
        return score
    
    def identify_key_strikes(
        self,
        strikes: np.ndarray,
        flows: FlowData,
        S0: float,
        n_strikes: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Identify the most significant strikes based on flow analysis.
        
        Returns
        -------
        dict with keys:
            'pinning': Top N strikes with positive (stabilizing) pressure
            'breakout': Top N strikes with negative (destabilizing) pressure
            'high_activity': Top N strikes by |GEX| + |volume/OI|
        """
        score = self.compute_flow_score(strikes, flows, S0)
        
        # Top pinning strikes (highest positive score)
        pin_idx = np.argsort(score)[-n_strikes:][::-1]
        pinning = strikes[pin_idx]
        
        # Top breakout strikes (most negative score)
        break_idx = np.argsort(score)[:n_strikes]
        breakout = strikes[break_idx]
        
        # High activity (|GEX| + normalized volume)
        activity = np.zeros(len(strikes))
        if flows.gex is not None:
            activity += np.abs(self._normalize_exposure(flows.gex, signed=False))
        if flows.volume is not None:
            activity += self._normalize_exposure(flows.volume, signed=False) * 0.5
        if flows.oi_chg is not None:
            activity += np.abs(self._normalize_exposure(flows.oi_chg, signed=False)) * 0.3
        
        act_idx = np.argsort(activity)[-n_strikes:][::-1]
        high_activity = strikes[act_idx]
        
        return {
            'pinning': pinning,
            'pinning_scores': score[pin_idx],
            'breakout': breakout,
            'breakout_scores': score[break_idx],
            'high_activity': high_activity,
            'activity_scores': activity[act_idx]
        }
    
    def compute_walls_and_magnet(
        self,
        option_chain: 'pd.DataFrame',
        S0: float,
        weight_col: str = 'openinterest',
        atm_n: int = 25
    ) -> Dict[str, Any]:
        """
        Compute Call Wall, Put Wall, Magnet, and their strengths from option chain.
        
        Based on Bhuyan and Chaudhury (2005) methodology.
        
        Parameters
        ----------
        option_chain : pd.DataFrame
            Option chain with columns: strike, type, and weight_col
        S0 : float
            Current spot price
        weight_col : str
            Column to use as weight (openinterest, volume, gexp, etc.)
        atm_n : int
            Number of closest strikes to include
        
        Returns
        -------
        dict with:
            'call_wall': Weighted average call strike (resistance)
            'put_wall': Weighted average put strike (support)
            'magnet': Combined weighted average (consensus)
            'spread': call_wall - put_wall
            'call_wall_strength': Concentration measure (0-1)
            'put_wall_strength': Concentration measure (0-1)
            'directional_bias': Negative = bearish, Positive = bullish
            'bias_magnitude': Strength of directional bias (0-1)
        """
        import pandas as pd
        
        df = option_chain.copy()
        
        # Filter to ATM region
        strikes = df['strike'].unique()
        closest = sorted(strikes, key=lambda x: abs(x - S0))[:atm_n]
        df = df[df['strike'].isin(closest)]
        
        # Split calls and puts
        calls = df[df['type'] == 'Call'].copy()
        puts = df[df['type'] == 'Put'].copy()
        
        if calls.empty or puts.empty or weight_col not in df.columns:
            return {
                'call_wall': np.nan, 'put_wall': np.nan, 'magnet': np.nan,
                'spread': np.nan, 'call_wall_strength': 0.0, 'put_wall_strength': 0.0,
                'directional_bias': 0.0, 'bias_magnitude': 0.0
            }
        
        # Get weights (absolute value for exposures that can be negative)
        call_strikes = calls['strike'].values
        call_weights = np.abs(calls[weight_col].values).astype(float)
        put_strikes = puts['strike'].values
        put_weights = np.abs(puts[weight_col].values).astype(float)
        
        # Compute walls
        call_total = np.sum(call_weights)
        put_total = np.sum(put_weights)
        
        if call_total > 1e-10:
            call_wall = np.sum(call_strikes * call_weights) / call_total
        else:
            call_wall = np.nan
        
        if put_total > 1e-10:
            put_wall = np.sum(put_strikes * put_weights) / put_total
        else:
            put_wall = np.nan
        
        # Magnet
        total_weight = call_total + put_total
        if total_weight > 1e-10:
            magnet = (np.sum(call_strikes * call_weights) + 
                     np.sum(put_strikes * put_weights)) / total_weight
        else:
            magnet = np.nan
        
        # WALL STRENGTH: How concentrated is OI around the wall?
        # Using coefficient of variation inverse: lower spread = stronger wall
        def compute_wall_strength(strikes, weights, wall):
            if np.sum(weights) < 1e-10 or np.isnan(wall):
                return 0.0
            # Weighted standard deviation
            w_norm = weights / np.sum(weights)
            variance = np.sum(w_norm * (strikes - wall)**2)
            std = np.sqrt(variance)
            # Normalize by strike range
            strike_range = np.max(strikes) - np.min(strikes)
            if strike_range < 1e-10:
                return 1.0
            # Lower relative std = stronger wall (more concentrated)
            relative_std = std / strike_range
            strength = 1.0 - np.clip(relative_std, 0, 1)
            return float(strength)
        
        call_wall_strength = compute_wall_strength(call_strikes, call_weights, call_wall)
        put_wall_strength = compute_wall_strength(put_strikes, put_weights, put_wall)
        
        # DIRECTIONAL BIAS: Where is magnet relative to spot?
        # Also consider total call vs put weight
        if not np.isnan(magnet):
            # Position bias: magnet above spot = bullish
            position_bias = (magnet - S0) / S0  # Normalized distance
            
            # Weight bias: more put OI than call = bearish (hedging)
            if total_weight > 1e-10:
                weight_ratio = (call_total - put_total) / total_weight
            else:
                weight_ratio = 0.0
            
            # Combined: position matters more
            directional_bias = position_bias * 0.7 + weight_ratio * 0.3
            bias_magnitude = min(abs(directional_bias) * 5, 1.0)  # Scale to 0-1
        else:
            directional_bias = 0.0
            bias_magnitude = 0.0
        
        return {
            'call_wall': call_wall,
            'put_wall': put_wall,
            'magnet': magnet,
            'spread': call_wall - put_wall if not (np.isnan(call_wall) or np.isnan(put_wall)) else np.nan,
            'call_wall_strength': call_wall_strength,
            'put_wall_strength': put_wall_strength,
            'directional_bias': directional_bias,  # Negative=bearish, Positive=bullish
            'bias_magnitude': bias_magnitude,       # 0-1 strength of bias
            'call_oi_total': call_total,
            'put_oi_total': put_total,
            'put_call_ratio': put_total / call_total if call_total > 1e-10 else np.nan
        }
    
    def adjust_pdf_with_walls(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        walls_info: Dict[str, Any],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Adjust PDF using wall/magnet framework with data-derived strengths.
        
        Parameters
        ----------
        strikes : np.ndarray
            Strike grid for PDF
        pdf : np.ndarray
            Base PDF values
        S0 : float
            Current spot price
        walls_info : dict
            Output from compute_walls_and_magnet() containing walls and strengths
        normalize : bool
            Whether to renormalize to integrate to 1
        
        Returns
        -------
        pdf_adjusted : np.ndarray
            Wall/magnet-adjusted PDF
        """
        pdf_adj = pdf.copy()
        
        call_wall = walls_info.get('call_wall', np.nan)
        put_wall = walls_info.get('put_wall', np.nan)
        magnet = walls_info.get('magnet', np.nan)
        call_strength = walls_info.get('call_wall_strength', 0.0)
        put_strength = walls_info.get('put_wall_strength', 0.0)
        bias_mag = walls_info.get('bias_magnitude', 0.0)
        
        # 1. Mode shift toward magnet (strength based on bias magnitude)
        if not np.isnan(magnet) and bias_mag > 0.05:
            dist_from_magnet = np.abs(strikes - magnet) / S0
            # Stronger bias = stronger pull toward magnet
            magnet_boost = np.exp(-dist_from_magnet / 0.05) * bias_mag * 0.5
            pdf_adj = pdf_adj * (1 + magnet_boost)
        
        # 2. Compress right tail at call wall (using computed strength)
        if not np.isnan(call_wall) and call_strength > 0.1:
            above_wall = strikes > call_wall
            if np.any(above_wall):
                overshoot = (strikes[above_wall] - call_wall) / S0
                # Stronger wall = faster decay
                decay_rate = call_strength * 15
                compression = np.exp(-overshoot * decay_rate)
                pdf_adj[above_wall] *= compression
        
        # 3. Compress left tail at put wall (using computed strength)
        if not np.isnan(put_wall) and put_strength > 0.1:
            below_wall = strikes < put_wall
            if np.any(below_wall):
                undershoot = (put_wall - strikes[below_wall]) / S0
                decay_rate = put_strength * 15
                compression = np.exp(-undershoot * decay_rate)
                pdf_adj[below_wall] *= compression
        
        # Ensure non-negative
        pdf_adj = np.maximum(pdf_adj, 0)
        
        # Renormalize
        if normalize:
            integral = np.trapezoid(pdf_adj, strikes)
            if integral > 1e-10:
                pdf_adj = pdf_adj / integral
        
        return pdf_adj
    
    def full_flow_adjustment(
        self,
        strikes: np.ndarray,
        pdf: np.ndarray,
        S0: float,
        option_chain: 'pd.DataFrame',
        tau: float = 0.0,
        weight_col: str = 'openinterest',
        use_walls: bool = True,
        use_exposures: bool = True,
        atm_n: int = 25
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combined flow adjustment using both exposure-based and wall/magnet methods.
        
        This is the recommended method for full flow-aware PDF adjustment.
        Takes the option chain directly for simplicity.
        
        Parameters
        ----------
        strikes : np.ndarray
            Strike grid for PDF
        pdf : np.ndarray
            Base PDF (e.g., from Heston FFT)
        S0 : float
            Current spot price
        option_chain : pd.DataFrame
            Option chain with columns: strike, type, openinterest, 
            and optionally gexp, vexp, cexp, oi_chg for exposures
        tau : float
            Time to expiry (for charm scaling)
        weight_col : str
            Column to use for wall weights (openinterest, volume, gexp, etc.)
        use_walls : bool
            Whether to apply wall/magnet adjustments
        use_exposures : bool
            Whether to apply GEX/vanna/charm adjustments
        atm_n : int
            Number of ATM strikes to consider
        
        Returns
        -------
        pdf_final : np.ndarray
            Fully adjusted PDF
        info : dict
            Diagnostic information including walls, strengths, adjustments
        """
        import pandas as pd
        
        info = {}
        pdf_current = pdf.copy()
        
        # Step 1: Compute walls and magnet (with strengths)
        walls = self.compute_walls_and_magnet(option_chain, S0, weight_col, atm_n)
        info['walls'] = walls
        
        # Step 2: Build FlowData from option chain if exposure columns exist
        if use_exposures:
            chain_strikes = option_chain['strike'].unique()
            chain_strikes = np.sort(chain_strikes)
            
            # Aggregate exposures by strike (sum calls and puts)
            agg_cols = ['strike']
            exp_cols = ['gexp', 'vexp', 'cexp', 'oi_chg', 'volume', 'openinterest']
            available_cols = [c for c in exp_cols if c in option_chain.columns]
            
            if available_cols:
                agg_df = option_chain.groupby('strike')[available_cols].sum().reset_index()
                agg_df = agg_df.sort_values('strike')
                
                flows = FlowData(
                    strikes=agg_df['strike'].values,
                    gex=agg_df['gexp'].values if 'gexp' in agg_df.columns else None,
                    vex=agg_df['vexp'].values if 'vexp' in agg_df.columns else None,
                    cex=agg_df['cexp'].values if 'cexp' in agg_df.columns else None,
                    oi_chg=agg_df['oi_chg'].values if 'oi_chg' in agg_df.columns else None,
                    volume=agg_df['volume'].values if 'volume' in agg_df.columns else None
                )
                
                # Apply exposure adjustments
                pdf_current, components = self.adjust_pdf(strikes, pdf_current, S0, flows, tau)
                info['exposure_adjustment'] = components
                info['key_strikes'] = self.identify_key_strikes(flows.strikes, flows, S0)
        
        # Step 3: Apply wall/magnet adjustments
        if use_walls and not np.isnan(walls.get('magnet', np.nan)):
            pdf_current = self.adjust_pdf_with_walls(strikes, pdf_current, S0, walls)
        
        return pdf_current, info
    
    def compute_multi_weight_walls(
        self,
        option_chain: 'pd.DataFrame',
        S0: float,
        weight_cols: list = None,
        atm_n: int = 25
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute walls using multiple weighting schemes for robustness.
        
        Parameters
        ----------
        option_chain : pd.DataFrame
            Option chain data
        S0 : float
            Current spot price
        weight_cols : list
            List of columns to use as weights. Default: ['openinterest', 'volume', 'gexp']
        atm_n : int
            Number of ATM strikes
        
        Returns
        -------
        dict mapping weight_col -> walls_info
        
        Also includes 'consensus' key with median values across all schemes.
        """
        if weight_cols is None:
            # Default: use available columns
            available = ['openinterest', 'volume', 'gexp', 'oi_chg']
            weight_cols = [c for c in available if c in option_chain.columns]
        
        results = {}
        for col in weight_cols:
            try:
                walls = self.compute_walls_and_magnet(option_chain, S0, col, atm_n)
                results[col] = walls
            except Exception:
                continue
        
        if not results:
            return {'consensus': {
                'call_wall': np.nan, 'put_wall': np.nan, 'magnet': np.nan,
                'spread': np.nan, 'call_wall_strength': 0.0, 'put_wall_strength': 0.0,
                'directional_bias': 0.0, 'bias_magnitude': 0.0
            }}
        
        # Compute consensus (median across schemes)
        def median_safe(key):
            vals = [r[key] for r in results.values() if not np.isnan(r.get(key, np.nan))]
            return float(np.median(vals)) if vals else np.nan
        
        results['consensus'] = {
            'call_wall': median_safe('call_wall'),
            'put_wall': median_safe('put_wall'),
            'magnet': median_safe('magnet'),
            'spread': median_safe('spread'),
            'call_wall_strength': median_safe('call_wall_strength'),
            'put_wall_strength': median_safe('put_wall_strength'),
            'directional_bias': median_safe('directional_bias'),
            'bias_magnitude': median_safe('bias_magnitude'),
        }
        
        return results
