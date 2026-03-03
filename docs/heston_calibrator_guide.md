# Heston Calibrator: Technical Guide

## Overview

The `heston_calibrator.py` module is a refactored and improved version of `hc3.py`, providing:
- More robust option pricing via Gil-Pelaez inversion (no damping needed)
- Parallel Differential Evolution optimizer (faster than scipy.minimize)
- Flow-aware PDF adjustments based on dealer positioning

---

## Changes from hc3.py

### 1. Pricing Formula: Damped vs Gil-Pelaez

**hc3.py (Damped Carr-Madan)**:
```python
# Required alpha damping parameter
char_p1 = heston_bates_charfunc(phi - (alpha + 1)*1j, ...)
denom_p1 = alpha**2 + alpha - phi**2 + 1j * phi * (2*alpha + 1)
# Then "undamp" at the end
call *= np.exp(-alpha * np.log(K))
```

**heston_calibrator.py (Gil-Pelaez P1/P2)**:
```python
# No damping needed - direct probability extraction
def _heston_charfunc_pj(phi, ..., j):
    # j=1: P1 measure (asset price weighting)
    # j=2: P2 measure (risk-neutral)
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * sigma if j == 1 else kappa
    ...
```

### Why Remove Alpha?

| Aspect | Damped (hc3.py) | Gil-Pelaez (new) |
|--------|-----------------|------------------|
| **Stability** | Requires tuning α for each tau/strike | Inherently stable |
| **Complexity** | Damping + undamping steps | Direct integration |
| **Edge cases** | α too small → oscillation; too large → inaccuracy | No tuning needed |
| **Deep OTM** | Can fail without careful α | More robust |

**Bottom line**: The Gil-Pelaez formulation is mathematically equivalent but numerically more stable without requiring the damping parameter.

---

### 2. Optimizer: scipy.minimize → Parallel Differential Evolution

**hc3.py**:
```python
result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
```
- Sequential evaluation
- Local optimizer (can get stuck)
- ~2-5 minutes for typical calibration

**heston_calibrator.py**:
```python
def differential_evolution_parallel(objective, bounds, pop_size, max_iter, n_workers):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Parallel fitness evaluation across population
        ...
```
- Parallel evaluation using ThreadPoolExecutor
- Global optimizer (explores parameter space)
- ~30-60 seconds with 4+ workers

---

### 3. Code Organization

| hc3.py | heston_calibrator.py |
|--------|---------------------|
| Functions scattered | Organized into `HestonCalibrator` class |
| Parameters as tuples | `HestonParams`, `BatesParams` dataclasses |
| Mixed logging/print | Proper logging module |
| Inline test code | Separate `if __name__ == "__main__"` |

---

## Integration Parameters: N and umax

### What They Control

| Parameter | Role | Effect of Increasing |
|-----------|------|---------------------|
| **N** | Number of integration points | More accurate but slower |
| **umax** | Upper limit of frequency integration | Captures high-frequency features |

### How They Affect Results

```
Price accuracy ∝ N × umax (up to a point)
Computation time ∝ N
```

**N (Integration Points)**:
- N=512: Fast but potentially inaccurate for short tau or deep OTM
- N=1024: Good balance for most cases
- N=2048: High accuracy, recommended for final calibration
- N=4096+: Diminishing returns, mainly for PDF extraction

**umax (Frequency Limit)**:
- umax=50: May miss tail features
- umax=100: Good for tau > 30 days
- umax=150-200: Better for short-dated options (< 14 days)
- umax=300+: Rarely needed, can introduce noise

### Optimal Selection Strategy

```python
def select_integration_params(tau: float, otm_ratio: float) -> Tuple[int, float]:
    """
    Select N and umax based on option characteristics.
    
    Parameters
    ----------
    tau : float
        Time to maturity in years
    otm_ratio : float
        Max |K/S - 1| in the chain (e.g., 0.3 for 30% OTM)
    
    Returns
    -------
    N, umax : int, float
    """
    # Short-dated needs higher frequency resolution
    if tau < 7/365:      # < 1 week
        base_umax = 200
        base_N = 2048
    elif tau < 30/365:   # < 1 month
        base_umax = 150
        base_N = 1024
    else:
        base_umax = 100
        base_N = 1024
    
    # Deep OTM needs more integration points
    if otm_ratio > 0.2:
        base_N = int(base_N * 1.5)
        base_umax *= 1.2
    
    return base_N, base_umax
```

### Validation Test

To verify your N/umax are adequate:

```python
def validate_integration_params(calibrator, S0, params, tau, r, strikes):
    """Check if increasing N/umax changes prices significantly."""
    # Current settings
    prices_base = calibrator.price_options(strikes, S0, params, tau, r)
    
    # Double resolution
    calibrator_fine = HestonCalibrator(N=calibrator.N*2, umax=calibrator.umax*1.5)
    prices_fine = calibrator_fine.price_options(strikes, S0, params, tau, r)
    
    # Should be < 0.1% difference
    max_diff = np.max(np.abs(prices_fine - prices_base) / (prices_base + 1e-6))
    print(f"Max relative difference: {max_diff:.4%}")
    
    if max_diff > 0.001:
        print("⚠️ Consider increasing N or umax")
    else:
        print("✓ Integration parameters are adequate")
```

---

## Flow-Aware PDF Adjustment

### Concept

The base Heston PDF captures risk-neutral expectations from option prices. The `FlowAwarePDF` class adjusts this using **dealer positioning data** to create a more market-aware distribution.

### Flow Types and Their Effects

| Flow | Measure | PDF Effect |
|------|---------|------------|
| **GEX (Gamma)** | Dealer gamma exposure | **Mode shaping**: Positive → pinning, Negative → breakout |
| **VEX (Vanna)** | ∂Δ/∂σ sensitivity | **Tail shaping**: High vanna → fat tails (vol feedback) |
| **CEX (Charm)** | ∂Δ/∂t decay | **Short-term tails**: High charm → overnight risks |
| **OI_chg** | Position building | **Confidence scaling**: + builds signals, - dampens |

### Signed Effects (Negative Pressure)

**Yes, flows can have negative pressure:**

```python
# GEX example:
# Positive GEX at strike K → dealers long gamma → they stabilize price near K
#   → INCREASE density at K (pinning effect)

# Negative GEX at strike K → dealers short gamma → they amplify moves away from K  
#   → DECREASE mode density, INCREASE tail density (breakout risk)
```

### Implementation

```python
from density.heston_calibrator import FlowAwarePDF, FlowData

# Create flow data container
flows = FlowData(
    strikes=chain_df['strike'].values,
    gex=chain_df['gexp'].values,        # Gamma exposure
    vex=chain_df['vexp'].values,        # Vanna exposure  
    cex=chain_df['cexp'].values,        # Charm exposure
    oi_chg=chain_df['oi_chg'].values    # OI change
)

# Initialize adjuster with sensitivity parameters
flow_adj = FlowAwarePDF(
    alpha_gex=0.3,      # GEX sensitivity (mode shaping)
    alpha_vex=0.2,      # Vanna sensitivity (tail shaping)
    alpha_cex=0.15,     # Charm sensitivity (short-term)
    alpha_oi=0.1,       # OI confidence scaling
    mode_bandwidth=0.05,    # ±5% around spot = "mode region"
    tail_threshold=0.10     # Beyond ±10% = "tails"
)

# Apply adjustments to base PDF
pdf_adjusted, components = flow_adj.adjust_pdf(
    strikes=pdf_strikes,
    pdf=pdf_base,
    S0=S0,
    flows=flows,
    tau=tau
)

# Analyze components
print(f"GEX adjustment range: {components['gex_adjustment'].min():.2f} - {components['gex_adjustment'].max():.2f}")
print(f"Total adjustment range: {components['total_adjustment'].min():.2f} - {components['total_adjustment'].max():.2f}")
```

### Key Strike Identification

```python
key = flow_adj.identify_key_strikes(flows.strikes, flows, S0, n_strikes=5)

print("Pinning zones (stabilizing):", key['pinning'])
print("Breakout risks (destabilizing):", key['breakout'])
print("High activity strikes:", key['high_activity'])
```

### Tuning Alpha Parameters

The `alpha_*` parameters control sensitivity to each flow type:

| Parameter | Default | Increase If | Decrease If |
|-----------|---------|-------------|-------------|
| `alpha_gex` | 0.3 | GEX historically predictive | GEX noisy or stale |
| `alpha_vex` | 0.2 | High vol regime | Low vol, stable regime |
| `alpha_cex` | 0.15 | 0-3 DTE options | Longer dated (>7 DTE) |
| `alpha_oi` | 0.1 | OI changes reliable | OI data has artifacts |

**Backtesting approach**:
```python
# Compare adjusted PDF predictions vs realized returns
def backtest_flow_alphas(historical_data, alpha_grid):
    results = []
    for alpha_gex in alpha_grid:
        flow_adj = FlowAwarePDF(alpha_gex=alpha_gex, ...)
        # ... compute adjusted PDFs and compare to realized
        score = evaluate_pdf_accuracy(...)
        results.append((alpha_gex, score))
    return optimal_alpha
```

---

## Quick Reference

### Basic Calibration

```python
from density.heston_calibrator import HestonCalibrator, BatesParams

calibrator = HestonCalibrator(N=2048, umax=150)
result = calibrator.calibrate(
    strikes=strikes,
    market_prices=prices,
    S0=spot,
    tau=time_to_expiry,
    r=risk_free_rate,
    pop_size=30,
    max_iter=100
)

if result.success:
    print(f"RMSE: {result.rmse:.4f}")
    print(f"v0={result.params.v0:.4f}, rho={result.params.rho:.4f}")
```

### PDF Extraction

```python
# FFT-based (faster, good for smooth PDFs)
strikes_fft, pdf_fft = calibrator.extract_pdf_fft(S0, result.params, tau, r)

# Price-based Breeden-Litzenberger (more robust)
strikes_bl, pdf_bl = calibrator.extract_pdf_from_prices(S0, result.params, tau, r)
```

### Flow-Adjusted PDF

```python
from density.heston_calibrator import FlowAwarePDF, FlowData

flows = FlowData(strikes=K, gex=gex, vex=vex, cex=cex, oi_chg=oi_chg)
adjuster = FlowAwarePDF(alpha_gex=0.3, alpha_vex=0.2)
pdf_adjusted, _ = adjuster.adjust_pdf(strikes_fft, pdf_fft, S0, flows, tau)
```

---

## Integration with OI Predictors

The `FlowAwarePDF` class integrates with the existing work in `bin/models/density/oi_predictors.py` (Bhuyan and Chaudhury, 2005 methodology).

### Wall/Magnet Framework

| Concept | Formula | Interpretation |
|---------|---------|---------------|
| **COP (Call Wall)** | Σ(K × OI_call) / Σ(OI_call) | Resistance level |
| **POP (Put Wall)** | Σ(K × OI_put) / Σ(OI_put) | Support level |
| **CWOP (Magnet)** | Combined weighted average | Consensus price |

### Using with Option Chain Data

```python
from density.heston_calibrator import FlowAwarePDF, FlowData
from bin.models.density.oi_predictors import get_oi_predictors

# Get walls/magnet from existing module
predictors = get_oi_predictors(option_chain, single_out=True)
call_wall = predictors[0]['CallWall']
put_wall = predictors[0]['PutWall']
magnet = predictors[0]['Magnet']

# Or compute directly with FlowAwarePDF
adjuster = FlowAwarePDF()
walls = adjuster.compute_walls_and_magnet(
    call_strikes, call_oi, put_strikes, put_oi
)

# Full adjustment pipeline
pdf_adjusted, info = adjuster.full_flow_adjustment(
    strikes=pdf_strikes,
    pdf=pdf_base,
    S0=S0,
    flows=flows,
    call_strikes=call_strikes,
    call_oi=call_oi,
    put_strikes=put_strikes,
    put_oi=put_oi,
    tau=tau
)

print(f"Call Wall: {info['walls']['call_wall']:.2f}")
print(f"Put Wall: {info['walls']['put_wall']:.2f}")  
print(f"Magnet: {info['walls']['magnet']:.2f}")
```

### Weighting Schemes

The OI predictors module supports multiple weighting schemes:

| Column | Flag | Use Case |
|--------|------|----------|
| `openinterest` | OI | Baseline positioning |
| `oi_chg` | OI_CHG | Recent positioning shifts |
| `volume` | VOL | Trading activity |
| `gexp` | gamma_exp | Gamma exposure walls |
| `vexp` | vega_exp | Vega exposure |
| `cexp` | charm_exp | Charm exposure |

---

## Numba Acceleration (Experimental)

The module includes experimental Numba JIT-compiled functions for ~10x speedup. However, **Numba is disabled by default** due to potential segfaults on some systems.

To enable (at your own risk):

```python
# At top of your script, before importing
import density.heston_calibrator as hc
hc.NUMBA_ENABLED = True

# Then reimport to pick up Numba
from importlib import reload
reload(hc)

# Use Numba-accelerated calibration
result = calibrator.calibrate_numba(...)
```

The standard `calibrate()` method using ThreadPoolExecutor is reliable and still provides good parallelization.

---

## Summary of Key Differences

| Feature | hc3.py | heston_calibrator.py |
|---------|--------|---------------------|
| Pricing | Damped Carr-Madan (needs α) | Gil-Pelaez P1/P2 (no α) |
| Optimizer | scipy L-BFGS-B (slow) | Parallel DE (fast) |
| Structure | Functions | Class-based |
| Flow weights | Absolute values only | Signed + interactions |
| PDF adjustment | Not supported | FlowAwarePDF class |
| Wall/Magnet | Separate module | Integrated |
