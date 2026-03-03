# andreasen_huge_pdf.py
# Andreasen–Huge arbitrage-free spline for option prices → smooth PDF
# 2025 implementation – simple version for calls (can be extended to puts)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapz
from scipy.optimize import minimize
from scipy.stats import norm

def andreasen_huge_fit(strikes, call_prices, S0, r, tau, lambda_penalty=1e-3, max_iter=300, verbose=False):
    """
    Andreasen–Huge style arbitrage-free spline fit to call prices.
    """
    strikes = np.asarray(strikes)
    call_prices = np.asarray(call_prices)
    
    if len(strikes) != len(call_prices):
        raise ValueError("strikes and call_prices must have the same length")
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    call_prices = call_prices[sort_idx]
    
    n = len(strikes)
    
    # Initial spline guess
    initial_spline = UnivariateSpline(strikes, call_prices, s=1.0, k=4)
    
    # Fine grid for smoothness penalty
    K_fine = np.linspace(strikes.min() * 0.92, strikes.max() * 1.08, 2000)
    
    def objective(coeffs):
        spl = UnivariateSpline(strikes, coeffs, k=4, s=0, ext=3)
        
        # Data fit error (can later weight by vega if desired)
        fitted = spl(strikes)
        data_error = np.sum((fitted - call_prices)**2)
        
        # Smoothness penalty: ∫ (d²C/dK²)² dK
        second_deriv = spl.derivative(n=2)(K_fine)
        smoothness = trapz(second_deriv**2, K_fine)
        
        # Penalize negative convexity (arbitrage indicator)
        convexity_penalty = np.sum(np.maximum(-second_deriv, 0)**2) * 5.0
        
        total = data_error + lambda_penalty * smoothness + convexity_penalty
        
        if verbose:
            print(f"Data err: {data_error:.4f} | Smooth: {smoothness:.4e} | Conv pen: {convexity_penalty:.4f} | Total: {total:.4f}")
        
        return total
    
    # Bounds per knot: intrinsic lower bound and upper bound (call <= S0)
    lower = np.maximum(0.0, S0 - strikes * np.exp(-r * tau))
    upper = S0 * np.ones(n)
    
    bounds_list = list(zip(lower, upper))
    
    # Optimize the y-values at the knots
    res = minimize(
        objective,
        initial_spline(strikes),  # better initial: evaluate initial spline at strikes
        bounds=bounds_list,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': verbose}
    )
    
    if not res.success:
        print("Warning: optimizer did not fully converge (may still be usable)")
        print(res.message)
    
    # Build final spline
    fitted_spline = UnivariateSpline(strikes, res.x, k=4, s=0, ext=3)
    
    # PDF via Breeden-Litzenberger
    K_pdf = np.linspace(strikes.min() * 0.95, strikes.max() * 1.05, 1500)
    d2C_dK2 = fitted_spline.derivative(n=2)(K_pdf)
    pdf_raw = np.exp(r * tau) * d2C_dK2
    pdf = np.maximum(pdf_raw, 0.0)
    
    # Normalize
    integral = trapz(pdf, K_pdf)
    if integral > 1e-10:
        pdf /= integral
    else:
        print("Warning: PDF integral near zero — check data/smoothing")
    
    fitted_prices = fitted_spline(K_pdf)
    
    return fitted_spline, K_pdf, pdf, fitted_prices


def ah_fit_vega(strikes, call_prices, S0, r, tau, lambda_penalty=5e-4, vega=None, max_iter=300, verbose=False):
    """
    Andreasen–Huge style arbitrage-free spline fit to call prices with vega weighting.
    
    - Vega weighting in misfit to prioritize liquid strikes
    - Tail extrapolation: linear in log-moneyness for far OTM/ITM
    """
    strikes = np.asarray(strikes)
    call_prices = np.asarray(call_prices)
    
    if len(strikes) != len(call_prices):
        raise ValueError("strikes and call_prices must have the same length")
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    call_prices = call_prices[sort_idx]
    
    n = len(strikes)
    
    # Approximate vega for weighting (use rough ATM IV ~0.25; refine if needed)
    if vega is None:
        approx_iv = 0.25
        moneyness = np.log(S0 / strikes)
        d1_approx = (moneyness + (r + 0.5 * approx_iv**2) * tau) / (approx_iv * np.sqrt(tau))
        vega = S0 * np.sqrt(tau) * norm.pdf(d1_approx)
        vega = np.maximum(vega, 1e-6)  # avoid zero
    
    # Initial spline guess
    initial_spline = UnivariateSpline(strikes, call_prices, s=0.8, k=4)
    
    # Fine grid for smoothness penalty
    K_fine = np.linspace(strikes.min() * 0.92, strikes.max() * 1.08, 2000)
    
    def objective(coeffs):
        spl = UnivariateSpline(strikes, coeffs, k=4, s=0, ext=3)
        
        # Vega-weighted data fit error
        fitted = spl(strikes)
        residuals = (fitted - call_prices)
        data_error = np.sum((residuals**2) * vega)  # or / vega for inverse weighting; test both
        
        # Smoothness penalty: ∫ (d²C/dK²)² dK
        second_deriv = spl.derivative(n=2)(K_fine)
        smoothness = trapz(second_deriv**2, K_fine)
        
        # Penalize negative convexity
        convexity_penalty = np.sum(np.maximum(-second_deriv, 0)**2) * 5.0
        
        total = data_error + lambda_penalty * smoothness + convexity_penalty
        
        if verbose:
            print(f"Data err: {data_error:.4f} | Smooth: {smoothness:.4e} | Conv pen: {convexity_penalty:.4f} | Total: {total:.4f}")
        
        return total
    
    # Bounds per knot
    lower = np.maximum(0.0, S0 - strikes * np.exp(-r * tau))
    upper = S0 * np.ones(n)
    
    bounds_list = list(zip(lower, upper))
    
    # Optimize
    res = minimize(
        objective,
        initial_spline(strikes),
        bounds=bounds_list,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': verbose}
    )
    
    if not res.success:
        print("Warning: optimizer did not fully converge")
        print(res.message)
    
    # Final spline with tail extrapolation
    # For far left (deep ITM): linear to S0 - K e^{-rτ}
    # For far right (deep OTM): linear decay in log(call) vs log(K/S0) for power-law tails
    def extrapolated_call(K):
        K = np.asarray(K)
        result = np.zeros_like(K, dtype=float)
        
        mask_mid = (K >= strikes.min()) & (K <= strikes.max())
        result[mask_mid] = UnivariateSpline(strikes, res.x, k=4, s=0, ext=3)(K[mask_mid])
        
        # Left tail (K < min): linear to intrinsic
        mask_left = K < strikes.min()
        if np.any(mask_left):
            slope_left = (res.x[1] - res.x[0]) / (strikes[1] - strikes[0])
            result[mask_left] = res.x[0] + slope_left * (K[mask_left] - strikes[0])
            result[mask_left] = np.maximum(result[mask_left], S0 - K[mask_left] * np.exp(-r * tau))
        
        # Right tail (K > max): linear in log(call) vs log(moneyness) for smooth decay
        mask_right = K > strikes.max()
        if np.any(mask_right):
            log_K_tail = np.log(K[mask_right] / S0)
            log_K_edge = np.log(strikes[-2:] / S0)
            log_call_edge = np.log(np.maximum(res.x[-2:], 1e-8))
            spl_log = UnivariateSpline(log_K_edge, log_call_edge, k=1, s=0, ext='extrapolate')
            result[mask_right] = np.exp(spl_log(log_K_tail))
            result[mask_right] = np.maximum(result[mask_right], 0)
        
        return result
    
    # PDF on fine grid
    K_pdf = np.linspace(strikes.min() * 0.95, strikes.max() * 1.05, 1500)
    # Approximate second deriv numerically (spline deriv can be noisy in tails)
    h = 1e-5 * (K_pdf[1] - K_pdf[0])  # small step
    C_plus = extrapolated_call(K_pdf + h)
    C_minus = extrapolated_call(K_pdf - h)
    C = extrapolated_call(K_pdf)
    d2C_dK2_approx = (C_plus + C_minus - 2 * C) / h**2
    pdf_raw = np.exp(r * tau) * d2C_dK2_approx
    pdf = np.maximum(pdf_raw, 0)
    
    # Normalize
    integral = trapz(pdf, K_pdf)
    if integral > 1e-10:
        pdf /= integral
    else:
        print("Warning: PDF integral near zero")
    
    fitted_prices = extrapolated_call(K_pdf)
    
    return extrapolated_call, K_pdf, pdf, fitted_prices


def andreasen_huge_fit_exposure(
    strikes,
    call_prices,
    S0,
    r,
    tau,
    lambda_penalty=5e-4,
    max_iter=300,
    verbose=False,
    exposure_gamma=None,       # np.array or pd.Series, same length as strikes
    exposure_charm=None,
    weight_type='vega'         # 'none', 'vega', 'gamma', 'charm', 'gamma_charm'
):
    """
    Andreasen–Huge arbitrage-free spline with optional exposure-based weighting.
    
    weight_type:
      - 'none':           uniform weighting
      - 'vega':           approximate BS vega
      - 'gamma':          use exposure_gamma only
      - 'charm':          use exposure_charm only
      - 'gamma_charm':    geometric mean of gamma and charm exposures
    """
    strikes = np.asarray(strikes)
    call_prices = np.asarray(call_prices)
    
    n = len(strikes)
    if len(call_prices) != n:
        raise ValueError("strikes and call_prices must have same length")
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    call_prices = call_prices[sort_idx]
    
    # ──────────────────────────────────────
    # Prepare weights
    # ──────────────────────────────────────
    weights = np.ones(n)  # default uniform
    
    if weight_type == 'vega' or weight_type == 'none':
        # Approximate vega
        approx_iv = 0.25
        moneyness = np.log(S0 / strikes)
        d1_approx = (moneyness + (r + 0.5 * approx_iv**2) * tau) / (approx_iv * np.sqrt(tau))
        vega_approx = S0 * np.sqrt(tau) * norm.pdf(d1_approx)
        weights = np.maximum(vega_approx, 1e-6)
    
    elif weight_type in ['gamma', 'charm', 'gamma_charm']:
        if weight_type in ['gamma', 'gamma_charm'] and exposure_gamma is None:
            raise ValueError("exposure_gamma required for gamma or gamma_charm weighting")
        if weight_type in ['charm', 'gamma_charm'] and exposure_charm is None:
            raise ValueError("exposure_charm required for charm or gamma_charm weighting")
        
        g_exp = np.asarray(exposure_gamma) if exposure_gamma is not None else np.ones(n)
        c_exp = np.asarray(exposure_charm) if exposure_charm is not None else np.ones(n)
        
        g_exp = g_exp[sort_idx]
        c_exp = c_exp[sort_idx]
        
        if weight_type == 'gamma':
            weights = np.maximum(g_exp, 1e-8)
        elif weight_type == 'charm':
            weights = np.maximum(c_exp, 1e-8)
        elif weight_type == 'gamma_charm':
            # Geometric mean — emphasizes strikes where both are high
            weights = np.sqrt(np.maximum(g_exp * c_exp, 1e-10))
    
    # Normalize weights so max=1
    weights /= weights.max()
    
    # Initial spline
    initial_spline = UnivariateSpline(strikes, call_prices, s=0.8, k=4)
    
    K_fine = np.linspace(strikes.min() * 0.92, strikes.max() * 1.08, 2000)
    
    def objective(coeffs):
        spl = UnivariateSpline(strikes, coeffs, k=4, s=0, ext=3)
        
        fitted = spl(strikes)
        residuals = fitted - call_prices
        
        # Weighted data error
        data_error = np.sum((residuals**2) * weights)
        
        second_deriv = spl.derivative(n=2)(K_fine)
        smoothness = trapz(second_deriv**2, K_fine)
        
        convexity_penalty = np.sum(np.maximum(-second_deriv, 0)**2) * 5.0
        
        total = data_error + lambda_penalty * smoothness + convexity_penalty
        
        if verbose:
            print(f"Data err: {data_error:.4f} | Smooth: {smoothness:.4e} | Conv pen: {convexity_penalty:.4f} | Total: {total:.4f}")
        
        return total
    
    lower = np.maximum(0.0, S0 - strikes * np.exp(-r * tau))
    upper = S0 * np.ones(n)
    
    res = minimize(
        objective,
        initial_spline(strikes),
        bounds=list(zip(lower, upper)),
        method='L-BFGS-B',
        options={
            'maxiter': 1000,           # was 300 → give it more room
            'maxfun': 5000,            # explicit limit on function evaluations
            'ftol': 1e-8,
            'gtol': 1e-8,
            'disp': verbose
        }
    )
    
    if not res.success:
        print("Warning: optimizer convergence issue:", res.message)
    
    # Final callable with tail extrapolation (same as before)
    # def extrapolated_call(K):
    #     K = np.asarray(K)
    #     result = np.zeros_like(K, dtype=float)
        
    #     mask_mid = (K >= strikes.min()) & (K <= strikes.max())
    #     result[mask_mid] = UnivariateSpline(strikes, res.x, k=4, s=0, ext=3)(K[mask_mid])
        
    #     mask_left = K < strikes.min()
    #     if np.any(mask_left):
    #         slope_left = (res.x[1] - res.x[0]) / (strikes[1] - strikes[0])
    #         result[mask_left] = res.x[0] + slope_left * (K[mask_left] - strikes[0])
    #         result[mask_left] = np.maximum(result[mask_left], S0 - K[mask_left] * np.exp(-r * tau))
        
    #     mask_right = K > strikes.max()
    #     if np.any(mask_right):
    #         log_K_tail = np.log(K[mask_right] / S0)
    #         log_K_edge = np.log(strikes[-2:] / S0)
    #         log_call_edge = np.log(np.maximum(res.x[-2:], 1e-8))
    #         spl_log = UnivariateSpline(log_K_edge, log_call_edge, k=1, s=0, ext='extrapolate')
    #         result[mask_right] = np.exp(spl_log(log_K_tail))
    #         result[mask_right] = np.maximum(result[mask_right], 0)
        
    #     return result

    def extrapolated_call(K):
        K = np.asarray(K)
        result = np.zeros_like(K, dtype=float)
        
        min_K, max_K = strikes.min(), strikes.max()
        
        # Mid region: exact spline
        mask_mid = (K >= min_K) & (K <= max_K)
        result[mask_mid] = UnivariateSpline(strikes, res.x, k=4, s=0, ext=3)(K[mask_mid])
        
        # Left tail (deep ITM calls): constant to intrinsic bound
        mask_left = K < min_K
        intrinsic_left = np.maximum(S0 - K[mask_left] * np.exp(-r * tau), 0)
        result[mask_left] = result[mask_mid][0] if np.any(mask_mid) else intrinsic_left  # hold last value or intrinsic
        
        # Right tail (deep OTM calls): exponential decay toward 0
        mask_right = K > max_K
        if np.any(mask_right):
            # Simple exponential decay from last point
            last_price = result[mask_mid][-1] if np.any(mask_mid) else 1e-6
            decay_rate = 0.05  # tune: higher = faster decay
            dist = K[mask_right] - max_K
            result[mask_right] = last_price * np.exp(-decay_rate * dist)
            result[mask_right] = np.maximum(result[mask_right], 0)
        
        return result
    
    # PDF
    K_pdf = np.linspace(strikes.min() * 0.95, strikes.max() * 1.05, 1500)
    h = 1e-5 * (K_pdf[1] - K_pdf[0])
    C_plus = extrapolated_call(K_pdf + h)
    C_minus = extrapolated_call(K_pdf - h)
    C = extrapolated_call(K_pdf)
    d2C_dK2 = (C_plus + C_minus - 2 * C) / h**2
    pdf_raw = np.exp(r * tau) * d2C_dK2
    pdf = np.maximum(pdf_raw, 0)
    
    integral = trapz(pdf, K_pdf)
    if integral > 1e-10:
        pdf /= integral
    
    fitted_prices = extrapolated_call(K_pdf)
    
    return extrapolated_call, K_pdf, pdf, fitted_prices, weights


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


def agg_by_strike(option_chain, x = 'strike', y = 'lastprice'):
    gcdf = option_chain.groupby(['timevalue',x]).agg({
        'gexp': 'sum', 
        'cexp': 'sum', 
        'vega': 'mean',
        'impliedvolatility': 'mean',
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

# ────────────────────────────────────────────────
#  Example usage + plotting
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys 
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from bin.dbm import DBManager
    from bin.main import Manager
    from bin.options.bsm.bs import bs_df

    manager = Manager()
    stock = manager.get_stock_data('spy')
    # option_chain = stock.options.option_chain_on(chain_date = '2026-02-20', use_cache = True)
    option_chain = stock.options.atm_chain_df
    option_chain = bs_df(option_chain, fit_iv = False)
    # option_chain.to_csv('density/option_chain.csv')
    # option_chain = pd.read_csv('density/option_chain.csv', parse_dates = ['expiry', 'gatherdate'])
    front_month = get_front_month_chain(option_chain, expiry = 0)
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
    print("Market call prices:", market_prices.round(3))

    # Fit
    fitted_spline, K_fine, pdf, fitted_prices = andreasen_huge_fit(
        strikes, market_prices, S0, r, tau,
        lambda_penalty=2e-3,   # tune this: higher = smoother, lower = closer fit
        max_iter=300
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: prices
    ax1.scatter(strikes, market_prices, color='red', s=80, label='Market calls', zorder=3)
    ax1.plot(K_fine, fitted_prices, color='blue', lw=2.2, label='Andreasen–Huge fit')
    ax1.axvline(S0, color='gray', ls='--', alpha=0.6, label='Spot')
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Call Price")
    ax1.set_title("Arbitrage-Free Call Price Fit")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: PDF
    ax2.plot(K_fine, pdf, color='navy', lw=2.4, label='Implied RN PDF')
    # Set xlim to 670-695
    ax2.set_xlim(670, 695)
    ax2.axvline(S0, color='gray', ls='--', alpha=0.6, label='Spot')
    ax2.fill_between(K_fine, pdf, color='navy', alpha=0.12)
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Risk-Neutral PDF (τ = {tau:.4f}, DTE ≈ {tau*365:.1f})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Optional: print integral check
    integral_check = trapz(pdf, K_fine)
    print(f"PDF integral check: {integral_check:.6f} (should be very close to 1)")