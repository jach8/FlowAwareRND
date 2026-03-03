import numpy as np
from scipy.stats import lognorm, skew, kurtosis, kstest, entropy as scipy_entropy
from scipy.signal import find_peaks
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz

def score_pdf_quality(
    K_pdf: np.ndarray,
    pdf: np.ndarray,
    spot: float,
    target_skew: float = -1.5,
    target_kurt: float = -1.5,
    peak_height_thresh: float = 0.08,
    extreme_move_pct: float = 0.10,      # 10% move threshold
    verbose: bool = False
) -> tuple[float, dict]:
    """
    Scores PDF quality for automatic parameter selection (λ in Andreasen–Huge, α in Heston).
    
    Lower score = better PDF (clean, realistic, unimodal, good tails).
    
    Returns: (score, diagnostics_dict)
    """
    diagnostics = {}

    # 1. Integral must be close to 1
    integral = trapz(pdf, K_pdf)
    integral_dev = abs(1 - integral)
    diagnostics['integral_dev'] = integral_dev
    if integral_dev > 0.08:
        if verbose:
            print("Rejected: bad integral")
        return 1e6, {**diagnostics, 'reject': 'bad_integral'}

    # 2. Mode & peak count (unimodal preferred)
    mode_idx = np.argmax(pdf)
    mode = K_pdf[mode_idx]
    mode_dev = abs(mode - spot) / spot
    diagnostics['mode_dev'] = mode_dev
    diagnostics['mode'] = mode

    # Find significant peaks
    peaks, props = find_peaks(
        pdf,
        height=peak_height_thresh * pdf.max(),
        distance=len(K_pdf) // 15
    )
    peak_count = len(peaks)
    diagnostics['peak_count'] = peak_count
    peak_count_penalty = max(0, peak_count - 1) * 5.0  # heavy penalty for multimodal

    # 3. Skew & kurtosis realism
    pdf_skew = skew(pdf)
    pdf_kurt = kurtosis(pdf)
    skew_dev = abs(pdf_skew - target_skew)
    kurt_dev = abs(pdf_kurt - target_kurt)
    diagnostics['skew'] = pdf_skew
    diagnostics['kurt'] = pdf_kurt

    # 4. Tails & imbalance
    left_tail_prob = trapz(pdf[K_pdf < spot * 0.95], K_pdf[K_pdf < spot * 0.95])
    right_tail_prob = trapz(pdf[K_pdf > spot * 1.05], K_pdf[K_pdf > spot * 1.05])
    tail_imb = abs(left_tail_prob - right_tail_prob * 1.2)  # allow ~20% more left
    diagnostics['left_tail_prob'] = left_tail_prob
    diagnostics['right_tail_prob'] = right_tail_prob
    diagnostics['tail_imb'] = tail_imb

    # Extreme tails penalty (should be small for short DTE)
    extreme_left = trapz(pdf[K_pdf < spot * (1 - extreme_move_pct)], K_pdf[K_pdf < spot * (1 - extreme_move_pct)])
    extreme_right = trapz(pdf[K_pdf > spot * (1 + extreme_move_pct)], K_pdf[K_pdf > spot * (1 + extreme_move_pct)])
    extreme_tail = (extreme_left + extreme_right) * 15.0
    diagnostics['extreme_tail'] = extreme_tail

    # 5. Entropy (concentration) & peak density ratio
    pdf_pmf = pdf / (pdf.sum() + 1e-12)
    ent = scipy_entropy(pdf_pmf)
    entropy_penalty = abs(ent - np.log(len(K_pdf) / 8.0)) * 0.8  # target moderate concentration
    diagnostics['entropy'] = ent

    max_ratio = pdf.max() / (pdf.mean() + 1e-10)
    ratio_penalty = abs(max_ratio - 9.0) * 0.6  # typical 6–12 for good bell
    diagnostics['max_density_ratio'] = max_ratio

    # 6. Lognormal fit quality (KS statistic)
    spacing = np.mean(np.diff(K_pdf))
    weights = np.round(pdf * spacing * len(K_pdf)).astype(int)
    pseudo_samples = np.repeat(K_pdf, weights)
    if len(pseudo_samples) < 30:
        fit_penalty = 1.0
    else:
        try:
            shape, loc, scale = lognorm.fit(pseudo_samples, floc=0)
            _, pval = kstest(pseudo_samples, 'lognorm', args=(shape, loc, scale))
            fit_penalty = 1 - pval  # 0 = perfect, 1 = poor
        except:
            fit_penalty = 1.0
    diagnostics['fit_ks_penalty'] = fit_penalty

    # ────────────────────────────────────────────────
    # Composite score (lower = better)
    # ────────────────────────────────────────────────
    score = (
        6.0 * integral_dev +              # critical
        7.0 * peak_count_penalty +        # unimodal priority
        5.0 * fit_penalty +               # distribution-likeness
        3.0 * skew_dev +                  # skew realism
        2.0 * kurt_dev +                  # kurtosis mild
        3.0 * mode_dev +                  # mode near spot
        2.5 * tail_imb +                  # reasonable asymmetry
        3.0 * extreme_tail +              # no huge crash/upside pricing
        1.5 * entropy_penalty +           # concentration
        1.0 * ratio_penalty               # peak height realism
    )

    if verbose:
        print(f"Score: {score:.4f}")
        print(diagnostics)

    return score, diagnostics