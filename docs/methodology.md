### Creating a Flow Aware PDF 



Extending the parameter tuning from [[Heston Calibration]]
We might also want to look at the various exposure open interest and volume levels within our calculation.
### 1. Definitions and Individual Roles

- **Gamma Exposure (GEX/gexp)**: The aggregate gamma (second derivative of option price to underlying price) across all options at a strike, often netted between calls and puts. Positive gexp means dealers are long gamma (hedge by buying low/selling high, stabilizing spot); negative means short gamma (hedge amplifies moves, volatile pinning).
    - Role in PDF: High |gexp| indicates pinning potential—weights these strikes more to shape the central mode/tails, as gamma clusters "pull" density toward them.
- **Vanna Exposure (vexp)**: Sensitivity of delta (first derivative) to volatility changes (∂delta/∂vol). High vanna means small vol shifts cause large delta adjustments, leading to vol-correlated flows (e.g., vol spike → dealers buy/sell underlying). Often netted like gamma.
    - Role in PDF: Captures vol-tail interactions—weight strikes with high |vexp| to emphasize skew/tails, as vanna drives convexity in the distribution (e.g., fat tails if vanna is high in OTM regions).
- **Charm Exposure (cexp)**: Sensitivity of delta to time passage (∂delta/∂time, or delta decay). High charm means delta changes rapidly as expiration nears, especially overnight/weekend in short-dated options. Positive/negative charm can flip hedging direction over time.
    - Role in PDF: Highlights time-decay risks—weight high |cexp| strikes to adjust short-term tails, as charm persistence signals overnight pinning or tail fattening (e.g., decaying calls → sell pressure).
- **Change in Open Interest (oi_chg)**: Net change in OI at a strike (positive = new positions opened, negative = positions closed/unwound). Not an exposure per se, but a flow indicator: positive oi_chg = building conviction (e.g., new hedges/bets), negative = exiting (profit-taking or sentiment shift).
    - Role in PDF: Signals dynamic flows—positive oi_chg boosts weight for emerging tails (e.g., new OTM put OI → fatter left tail), negative reduces weight for fading strikes.

### 2. Key Relationships and Interactions

These aren't isolated— they interact in ways that amplify or offset each other, especially in short-dated options where time decay and vol are key. Here's a mapped overview:

- **Gamma ↔ Vanna**:
    - Gamma is price-driven hedging; vanna is vol-driven delta change. High gamma strikes often have high vanna because gamma peaks near ATM, where vol sensitivity is high.
    - Interaction: Short gamma + high vanna = explosive vol feedback (price drop → vol up → vanna flips delta → more selling). In PDF weighting: combine |gexp| + |vexp| to heavily emphasize strikes where vol-price feedback could fatten tails (e.g., downside skew amplification).
- **Gamma ↔ Charm**:
    - Gamma hedging is instantaneous; charm erodes delta over time, effectively "decaying" gamma's impact. High gamma + high charm = time-sensitive pinning (charm flips hedging direction near expiry).
    - Interaction: Positive charm on high-gamma calls = bullish decay (delta becomes less positive over time → less buying support). In weighting: |gexp| * |cexp| for short-DTE, to weight strikes where time decay could sharpen or shift the mode/tails.
- **Vanna ↔ Charm**:
    - Both are second-order Greeks: vanna ties delta to vol, charm to time. They interact in "vol-time" space (e.g., overnight vol spike + charm decay = rapid delta shift).
    - Interaction: High vanna + charm = volatile overnight risks (common in 0DTE). Weighting: |vexp| + |cexp| for tail emphasis, as persistence signals fat/extreme tails.
- **OI_Chg ↔ All Exposures**:
    - oi_chg is the "flow lens" — positive oi_chg on high gexp/vexp/cexp strikes = building exposure (reinforces weighting), negative = unwinding (could reduce weight or signal reversal).
    - Interaction: Positive oi_chg * high gexp = strong pinning build-up (boost mode weight); negative oi_chg on high vanna = sentiment shift in vol-sensitive strikes (dampen tail weight). In weighting: multiply exposures by (1 + oi_chg / max_oi_chg) to scale by flow direction (positive boosts, negative dampens).

Overall: These relationships form a "hedging feedback loop" in short-dated options — gamma pins spot, vanna/charm modulate it over vol/time, oi_chg shows if it's intensifying or fading. In PDF context, weighting with them makes the distribution more "dealer-aware" (tails reflect hedging risks, mode reflects pinning).