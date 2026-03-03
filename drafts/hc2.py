##### WORKING VERSION DO NOT CHANGE #####

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.integrate import trapz
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Step 1: Heston Characteristic Function (vectorized over phi) - Stable "little trap" Heston Characteristic Function
def heston_charfunc(phi, v0, kappa, theta, xi, rho, lambd, tau, r, q=0.0):
    """
    phi: shape (N,) or (N,1) — frequency vector
    Returns complex array same shape as phi
    """
    phi = np.asarray(phi, dtype=complex)          # ensure complex
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]                  # (N,1) for safety

    i = 1j
    a = kappa * theta
    u = i * phi + phi**2                          # (N,1)
    b = kappa + lambd
    rspi = rho * xi * phi * i                     # (N,1)

    tmp = (rspi - b)**2 + xi**2 * u
    d = np.sqrt(tmp)                              # complex sqrt, shape (N,1)
    d = np.where(np.isnan(d) | np.isinf(d.real) | np.isinf(d.imag), 1.0 + 0j, d)  # fallback to mild value

    # Little-trap formulation (recommended for stability)
    g = (b - rspi + d) / (b - rspi - d)           # (N,1)
    g = np.clip(g.real, -10, 10) + 1j * np.clip(g.imag, -10, 10)

    g_exp = g * np.exp(-d * tau)                  # (N,1)
    g_exp = np.where(np.isnan(g_exp) | np.isinf(g_exp), 0.0 + 0j, g_exp)

    term1 = 1 - g_exp
    term2 = 1 - g
    pow_term = (term1 / term2) ** (-2 * a / xi**2)
    pow_term = np.where(np.isnan(pow_term) | np.isinf(pow_term), 1.0, pow_term)

    exp_arg = (a / xi**2) * (b - rspi + d) * tau + \
              (v0 / xi**2) * (b - rspi + d) * (1 - np.exp(-d * tau)) / (1 - g_exp)
    exp_arg = np.clip(exp_arg.real, -50, 50) + 1j * np.clip(exp_arg.imag, -50, 50)
    char = np.exp(r * phi * i * tau) * pow_term * np.exp(exp_arg)

    # Clean any NaN/inf from numerics
    # char = np.nan_to_num(char, nan=0j, posinf=0j, neginf=0j)
    char = np.where(np.isnan(char) | np.isinf(char.real) | np.isinf(char.imag), 0j, char)
    return char.squeeze()   # back to (N,) if input was (N,)

# Stable "little trap" Heston Characteristic Function with Bates Jump Extension
def heston_bates_charfunc(phi, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q=0.0, lambda_j=0.0, mu_j=0.0, sigma_j=0.0):
    """
    phi: shape (N,) or (N,1) — frequency vector
    Returns complex array same shape as phi
    """
    phi = np.asarray(phi, dtype=complex)          # ensure complex
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]                  # (N,1) for safety

    i = 1j
    a = kappa * theta
    u = i * phi + phi**2                          # (N,1)
    b = kappa + lambd_sv
    rspi = rho * xi * phi * i                     # (N,1)

    tmp = (rspi - b)**2 + xi**2 * u
    d = np.sqrt(tmp)                              # complex sqrt, shape (N,1)

    # Little-trap formulation (recommended for stability)
    denom = (b - rspi - d)
    denom = np.where(np.abs(denom) < 1e-14, 1e-14 * (1 + 1j), denom)
    g = (b - rspi + d) / denom

    g_exp = g * np.exp(-d * tau)                  # (N,1)
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

    # Clean any NaN/inf from numerics
    char = np.where(np.isnan(char) | np.isinf(char.real) | np.isinf(char.imag), 0j, char)
    return char.squeeze()   # back to (N,) if input was (N,)


# Step 2: Vectorized BS Call Price (for IV inverter)
def bs_price(S0, K, tau, r, sigma, q=0.0, is_call=True):
    """
    Vectorized Black-Scholes price — is_call can be scalar or array
    """
    S0 = np.asarray(S0)
    K = np.asarray(K)
    tau = np.asarray(tau)
    r = np.asarray(r)
    sigma = np.asarray(sigma)
    q = np.asarray(q)
    is_call = np.asarray(is_call)  # allow array or scalar

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = S0 * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put_price  = K * np.exp(-r * tau) * norm.cdf(-d2) - S0 * np.exp(-q * tau) * norm.cdf(-d1)

    # Use np.where to select per element if is_call is array
    return np.where(is_call, call_price, put_price)


# Step 3: Vectorized BS IV Inverter (Newton-Raphson)
def vectorized_bs_iv(prices, S0, K, tau, r, q=0.0, is_call=True, max_iv=5.0, tol=1e-6, max_iter=50):
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
        delta_sigma = error / vega
        delta_sigma = np.clip(delta_sigma, -0.5, 0.5)
        sigma -= delta_sigma

        if np.all(np.abs(delta_sigma) < tol):
            break

    sigma = np.clip(sigma, 1e-6, max_iv)
    sigma[vega < 1e-4] = np.nan
    return sigma

def compute_ivs(prices, S0, strikes, tau, r, q):
    ivs = np.full_like(strikes, np.nan, dtype=float)
    
    # OTM calls (K >= S0) — invert as calls
    mask_otm_call = strikes >= S0
    if np.any(mask_otm_call):
        ivs[mask_otm_call] = vectorized_bs_iv(
            prices[mask_otm_call],
            S0, strikes[mask_otm_call], tau, r, q,
            is_call=True
        )
    
    # OTM puts (K < S0) — convert to put prices, then invert as puts
    mask_otm_put = strikes < S0
    if np.any(mask_otm_put):
        put_prices = prices[mask_otm_put] - S0 * np.exp(-q * tau) + strikes[mask_otm_put] * np.exp(-r * tau)
        put_prices = np.maximum(put_prices, 0)  # avoid negative prices from noise
        ivs[mask_otm_put] = vectorized_bs_iv(
            put_prices,
            S0, strikes[mask_otm_put], tau, r, q,
            is_call=False
        )
    
    return ivs

# Step 4: Damped P1/P2 Heston Pricing (vectorized, stable for short tau and includes jump params)
def heston_call_price_damped(S0, K, v0, kappa, theta, xi, rho, lambd_sv, tau, r,
                             q=0.0, alpha=0.05, N=8192, umax=1000, lambda_j=0.0, mu_j=0.0, sigma_j=0.0):
    K = np.atleast_1d(K)
    n_strikes = len(K)
    
    phi = np.linspace(1e-5, umax, N)          # (N,)
    
    # P1 integrand
    char_p1 = heston_bates_charfunc(phi - (alpha + 1)*1j, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q, lambda_j, mu_j, sigma_j)
    logK = np.log(K)                          # (n_strikes,)
    exp_part_p1 = np.exp(-1j * phi[:, np.newaxis] * logK[np.newaxis, :])  # (N, n_strikes)
    numer_p1 = exp_part_p1 * char_p1[:, np.newaxis]                       # (N, n_strikes)
    denom_p1 = alpha**2 + alpha - phi**2 + 1j * phi * (2*alpha + 1)       # (N,)
    denom_p1 = denom_p1[:, np.newaxis]                                    # (N, 1) → broadcasts
    integrand_p1 = np.real(numer_p1 / denom_p1)
    integrand_p1 = np.nan_to_num(integrand_p1, nan=0.0, posinf=0.0, neginf=0.0)
    P1 = trapz(integrand_p1, phi, axis=0) / np.pi                        # (n_strikes,)
    
    # P2 similar
    char_p2 = heston_bates_charfunc(phi - alpha*1j, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q, lambda_j, mu_j, sigma_j)
    exp_part_p2 = np.exp(-1j * phi[:, np.newaxis] * logK[np.newaxis, :])
    numer_p2 = exp_part_p2 * char_p2[:, np.newaxis]
    denom_p2 = alpha**2 + alpha - phi**2 + 1j * phi
    denom_p2 = denom_p2[:, np.newaxis]
    integrand_p2 = np.real(numer_p2 / denom_p2)
    integrand_p2 = np.nan_to_num(integrand_p2, nan=0.0, posinf=0.0, neginf=0.0)
    P2 = trapz(integrand_p2, phi, axis=0) / np.pi
    
    # Call price
    call = (S0 * np.exp(-q * tau) * (0.5 + P1) -
            K * np.exp(-r * tau) * (0.5 + P2))
    
    # Undamp (note: sign might be -alpha for some conventions; test with known data)
    call *= np.exp(-alpha * np.log(K))  
    
    return np.maximum(call, 0)


# Step 5: Calibration Function (IV-based + adds jump params)
def calibrate_heston_to_chain(strikes, market_prices, S0, tau, r, q=0.0, lambd_sv=0.0, alpha=1.5, N=2048, umax=100, weight_type='atm'):
    bounds = [
        (0.05, 5.0),    # kappa
        (0.01, 0.15),   # theta
        (0.01, 0.1),    # v0
        (-0.95, -0.4),  # rho
        (0.5, 2.0),     # xi
        (0.1, 10.0),    # lambda_j (jump intensity)
        (-0.1, 0.0),    # mu_j (mean jump size, negative for downside skew)
        (0.05, 0.2)     # sigma_j (jump vol)
    ]

    # Pre-compute market IVs (fixed)
    market_ivs = np.full_like(strikes, np.nan)
    mask_call = strikes >= S0
    market_ivs[mask_call] = vectorized_bs_iv(market_prices[mask_call], S0, strikes[mask_call], tau, r, q, is_call=True)
    mask_put = strikes < S0
    if np.any(mask_put):
        put_prices_market = market_prices[mask_put] - S0 * np.exp(-q * tau) + strikes[mask_put] * np.exp(-r * tau)
        market_ivs[mask_put] = vectorized_bs_iv(put_prices_market, S0, strikes[mask_put], tau, r, q, is_call=False)

    def objective(p):
        kappa, theta, v0, rho, xi, lambda_j, mu_j, sigma_j = np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds])

        # Hard constraints: positive params (Feller 2*kappa*theta >= xi^2 not required for pricing)
        if v0 <= 0 or theta <= 0 or xi <= 0 or kappa <= 0:
            return 1e12

        try:
            model_prices = heston_call_price_damped(S0, strikes, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q=q, alpha=alpha, N=N, umax=umax,
                                                   lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
        except Exception as e:
            logger.debug("heston_call_price_damped failed: %s", e)
            return 1e12

        if not np.any(np.isfinite(model_prices)) or np.all(model_prices <= 0):
            logger.debug("model_prices non-finite or all non-positive")
            return 1e12

        model_ivs = compute_ivs(model_prices, S0, strikes, tau, r, q)
        valid = np.isfinite(model_ivs) & np.isfinite(market_ivs)
        if not np.any(valid):
            n_model = np.sum(np.isfinite(model_ivs))
            n_mkt = np.sum(np.isfinite(market_ivs))
            logger.debug("no valid IVs: n_finite(model_ivs)=%d, n_finite(market_ivs)=%d", n_model, n_mkt)
            return 1e12

        errors = (model_ivs[valid] - market_ivs[valid]) ** 2

        dist = np.abs(strikes[valid] - S0) / S0
        if weight_type == 'atm':
            weights = 1 / (dist + 0.01) ** 0.5
        elif weight_type == 'tail':
            weights = np.exp(dist * 3.0) + 0.3
        else:  # uniform
            weights = np.ones_like(dist)

        weights = weights / weights.max()
        weighted_errors = weights * errors
        weighted_rmse = np.sqrt(np.sum(weighted_errors) / np.sum(weights))

        # print(f"Weighted IV RMSE ({weight_type}): {weighted_rmse:.4f} | Params: {p.round(4)}")  # Progress
        return weighted_rmse

    best_rmse = 1e12
    best_p = None
    last_res = None

    # Include inits with smaller xi for numerical stability of the characteristic function
    initials = [
        [2.0, 0.08, 0.05, -0.85, 0.8, 1.0, -0.05, 0.1],   # xi=0.8 stable
        [2.0, 0.08, 0.05, -0.85, 2.2, 1.0, -0.05, 0.1],
        [4.0, 0.06, 0.04, -0.92, 0.9, 1.5, -0.05, 0.1],   # xi=0.9 stable
        [1.5, 0.10, 0.07, -0.78, 1.8, 0.5, -0.03, 0.08],
        [3.0, 0.09, 0.06, -0.88, 1.2, 1.0, -0.05, 0.1],   # xi=1.2
    ]

    iter_count = [0]  # mutable so callback can update

    def progress_callback(xk):
        iter_count[0] += 1
        # logger.info("  iter %d  params=%s", iter_count[0], np.round(xk, 4).tolist())

    for i, init_p in enumerate(initials):
        iter_count[0] = 0
        # logger.info("Starting multi-start %d/%d from init %s", i + 1, len(initials), [round(x, 4) for x in init_p])
        res = minimize(
            objective, init_p, bounds=bounds, method='L-BFGS-B',
            options={'maxiter': 1000}, callback=progress_callback
        )
        last_res = res
        if res.fun < best_rmse:
            best_rmse = res.fun
            best_p = res.x.copy()
            logger.info("Improved: RMSE=%.6f from init %s", best_rmse, [round(x, 4) for x in init_p])

    if best_p is not None:
        out_p, out_rmse = best_p, best_rmse
    elif last_res is not None:
        out_p, out_rmse = last_res.x, last_res.fun
    else:
        raise RuntimeError("calibrate_heston_to_chain: no initials provided or all minimizations failed")
    logger.info("Calibration done. Best RMSE=%.6f  params=%s", out_rmse, np.round(out_p, 4).tolist())
    return out_p, out_rmse


# Step 6: PDF Extraction Function
from scipy.interpolate import UnivariateSpline

def extract_pdf(strikes, prices, r, tau, s = 0.05, k = 4):
    # Sort strikes/prices (just in case)
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    prices = prices[sort_idx]

    # Smoother spline with lower s for more curvature
    spl = UnivariateSpline(strikes, prices, s=s, k=k)  # quartic, less smoothing

    # Finer grid for deriv to avoid artifacts
    fine_strikes = np.linspace(strikes.min(), strikes.max(), 1000)
    second_deriv = spl.derivative(n=2)(fine_strikes)

    # PDF
    pdf = np.exp(r * tau) * second_deriv
    pdf = np.maximum(pdf, 0)

    # Normalize
    integral = trapz(pdf, fine_strikes)
    if integral > 0:
        pdf /= integral

    return fine_strikes, pdf


# Updated PDF via FFT for Bates (same as Heston since CF is extended)
def heston_pdf_via_fft(S0, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q=0.0, N=4096, L=0.025, lambda_j=0.0, mu_j=0.0, sigma_j=0.0):
    eta = 2 * np.pi / (N * L)
    b = N * L / 2
    u = np.arange(0, N) * eta
    ku = -b + L * np.arange(0, N)
    
    psi_u = heston_bates_charfunc(u - 0.5j, v0, kappa, theta, xi, rho, lambd_sv, tau, r, q, lambda_j, mu_j, sigma_j)
    psi_u = np.where(np.isnan(psi_u) | np.isinf(psi_u.real) | np.isinf(psi_u.imag), 0j, psi_u)
    psi_u[np.abs(u) < 1e-6] = 1.0 + 0j
    
    psi = psi_u / (1j * u + 0.25 + 1e-12)
    fft_input = np.exp(1j * b * u) * psi * eta
    fft_input = np.nan_to_num(fft_input, nan=0.0, posinf=0.0, neginf=0.0)
    fft_output = np.real(np.fft.fft(fft_input))
    
    adj = np.log(S0) + (r - q - 0.5 * v0) * tau
    log_strikes = ku + adj
    strikes = np.exp(np.clip(log_strikes, -50, 50))  # Clip to avoid exp inf
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    pdf_log = np.maximum(fft_output / np.pi, 0)[sort_idx]
    pdf = pdf_log / strikes
    
    valid = np.isfinite(pdf) & (pdf > 0)
    if np.any(valid):
        integral = trapz(pdf[valid], strikes[valid])
        if integral > 0:
            pdf /= integral
    
    return strikes, pdf


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


def agg_by_strike(option_chain, x = 'strike', y = 'lastprice'):
    gcdf = option_chain.groupby(['timevalue',x]).agg({
        'gexp': 'sum', 
        'cexp': 'sum', 
        'vexp': 'sum',
        'vega': 'mean',
        'impliedvolatility': 'mean',
        'volume': 'sum',
        'openinterest': 'sum',
        'oi_chg': 'sum', ## This will be net oi change per strike, so if one strike goes down, the other goes up, the net oi change will be 0.

         y: 'mean',
        'stk_price': 'last'})
    return gcdf.reset_index()

def get_front_month_chain(option_chain, expiry = None):
    expiry_dates = sorted(option_chain.expiry.unique())
    if expiry is not None:
        if type(expiry) == str:
            expiry = pd.to_datetime(expiry)
        if expiry in expiry_dates:
            print(f"Using specified expiry: {expiry}")
            return option_chain[option_chain.expiry == expiry]
        if type(expiry) == int:
            if 0 <= expiry < len(expiry_dates):
                print(f"Using expiry at index {expiry}: {expiry_dates[expiry]}")
                return option_chain[option_chain.expiry == expiry_dates[expiry]]
            else:
                raise ValueError(f"Expiry index {expiry} out of range.")
        else:
            current_datetime = pd.Timestamp.now()
            if current_datetime.hour > 15: 
                expiry = expiry_dates[1]
                print(f"Current time is after 3 PM. Using next expiry: {expiry}")
            else:
                expiry = expiry_dates[0]
            
            return option_chain[option_chain.expiry == expiry]

def filter_otm_options(cdf):
    # keep otm options only
    otm_calls = cdf[(cdf.strike > cdf.stk_price) & (cdf.type == 'Call')]
    otm_puts = cdf[(cdf.strike < cdf.stk_price) & (cdf.type == 'Put')]

    cdf = pd.concat([otm_calls, otm_puts])
    return cdf




if __name__ == "__main__":
    import sys 
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from bin.dbm import DBManager
    from bin.main import Manager
    from bin.options.bsm.bs import bs_df

    manager = Manager()
    stock = manager.get_stock_data('spy')
    option_chain = stock.options.option_chain_on(chain_date = '2026-02-20', use_cache = True)
    # option_chain = stock.options.atm_chain_df
    option_chain = bs_df(option_chain, fit_iv = False)
    option_chain.to_csv('density/option_chain.csv')
    option_chain = pd.read_csv('density/option_chain.csv', parse_dates = ['expiry', 'gatherdate'])
    front_month = get_front_month_chain(option_chain, expiry = 10)
    gcdf = agg_by_strike(front_month)
    print('Strike range:',gcdf.strike.min(),'--->', gcdf.strike.max())

    # Example Usage (replace with your data)
    # Assume gcdf loaded
    S0 = stock.price_data.current_price
    tau = gcdf['timevalue'].mean()
    r = 0.0405
    q = 0.00
    strikes = gcdf['strike'].values
    market_prices = gcdf['lastprice'].values
    actual_ivs = gcdf['impliedvolatility'].values

    # Calibrate
    alpha = .75
    N = 1024
    umax = 500
    weight_type = 'tail'
    lambd = 0.0
    s = 0.05
    k = 4

    print(f"Calibrating Heston model to chain with alpha: {alpha}, N: {N}, umax: {umax}, weight_type: {weight_type}, lambd: {lambd}, s: {s}, k: {k}")
    print('res = differential_evolution(objective, bounds=bounds, maxiter=50, popsize=20, tol=1e-5, disp=True)')

    # Compute model prices with opt_params
    opt_params, rmse = calibrate_heston_to_chain(strikes, market_prices, S0, tau, r, q, lambd_sv = lambd, alpha = alpha, N = N, umax = umax, weight_type = weight_type)
    # kappa, theta, v0, rho, xi = opt_params
    kappa, theta, v0, rho, xi, lambda_j, mu_j, sigma_j = opt_params

    # model_prices = heston_call_price_damped(S0, strikes, v0, kappa, theta, xi, rho, lambd, tau, r, q=q, alpha = alpha, N = N, umax = umax)
    model_prices = heston_call_price_damped(S0, strikes, v0, kappa, theta, xi, rho, lambd, tau, r, q=q, alpha=alpha, N=N, umax=umax,
                                       lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)

    # Extract PDF from model prices (or market if preferred)
    strikes_pdf, pdf_values = extract_pdf(strikes, model_prices, r, tau, s = s, k = k)

    ### IV RMSE Check
    model_ivs = vectorized_bs_iv(model_prices, S0, strikes, tau, r, q, is_call=(strikes >= S0))
    market_ivs = vectorized_bs_iv(market_prices, S0, strikes, tau, r, q, is_call=(strikes >= S0))
    iv_rmse = np.sqrt(np.nanmean((model_ivs - market_ivs)**2))


    # Extract Pure Market PDF, without model assumptions.
    market_pdf = extract_pdf(strikes, market_prices, r, tau, s = s, k = k)
    # hpdf = heston_pdf_via_fft(S0 = S0, v0 = v0, kappa = kappa, theta = theta, xi = xi, rho = rho, lambd = lambd, tau = tau, r = r, q = q, N = N, L = 0.025)
    hpdf = heston_pdf_via_fft(S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, lambd_sv=lambd, tau=tau, r=r, q=q, N=N, L=0.025,
                          lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    fig, ax = plt.subplots(1,2,figsize=(15, 5))

    ax[0].plot(strikes_pdf, pdf_values, label='Heston PDF', color = 'blue')
    # twiny = ax[0].twiny()
    # twiny.plot(hpdf[0], hpdf[1], label='Heston PDF via FFT', color = 'orange')
    ax[0].set_xlabel('Strike')
    ax[0].set_ylabel('Density')
    ax[0].set_title('Risk-Neutral PDF ' +f'Tau: {tau:.4f}, DTE: {tau * 365:.4f}')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(strikes, model_ivs, label='Model IV', color='blue')
    ax[1].plot(strikes, market_ivs, label='Market IV', color='red')
    ax[1].plot(strikes, actual_ivs, label='Actual IV', color='green')
    ax[1].set_xlabel('Strike')
    ax[1].set_ylabel('Implied Volatility')
    ax[1].set_title(f'IV RMSE: {iv_rmse:.4f}')
    ax[1].grid()
    ax[1].legend()
    plt.show()