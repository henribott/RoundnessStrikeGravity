# FX_RoundnessStrikeGravity
Strike Gravity: A Unified Framework for Roundness, Gamma Exposure, and Reflexive Volatility at Psychological Price Levels

[IN PROGRESS]

Note: This project is an unpretentious attempt to give colour to an idea I had at the desk, while my main objectives are to learn and explore the potential of Gamma sizing round numbers and the psychological patterns around them, in an attempt to generate alpha in Forex Options markets.
> (This will probably be the theme for my Undergraduate Thesis paper too lol)

This project is based on the conceptual property of roundness as not only the property of a number (e.g. The number 1.5 is theoretically more "round" than 1.68375), but also as the property of how the payoff surface curves around that number. The key is that roundness creates a discrete discontinuity in the distribution of strikes, which shows as a local maximum in the gamma surface, given in (THEORY) in a continuous FX Market, gamma is smooth. Near a round number with high OI(open interest), I am using the backtests to observe and explore whether gamma is a spike, a curvature anomaly, and the spike being the roundness in geometric terms. 

The catch I initially sensed, from observing the markets, is that since gamma proximity is itself a function of implied volatility, IV near round numbers is distorted by the roundness itself, so there is a circularity to it.
The true gamma at a round-number strike is larger than BS predicts from a flat vol surface, because the Open Interest concentration creates a local volatility smile kink(at first, a weird, unstable hedging behaviour).

Roundness(k) = local excess curvature of the Implied Volatility surface at K (∂²IV/∂K² |_K)
> IF the roundness is real, the local IV at K is elevated (σ_K > σ_ATM), and a higher local vol at K means BS gamma at K is lower (vol spreads the gamma bell curve, reducing the peak)
FIX 1: For the gravity formula: **Ψ(k) × Ω × Φ** I was using flat IV instead of local vol at K when computing Φ



The backtest tests FOUR claims from the theory, each independently:
 
  CLAIM 1 — Roundness predicts excess realized vol
            At bars where spot is within N pips of a high-R strike, RV(next M bars) > RV(random bars at same distance from non-round strikes).
 
  CLAIM 2 — Edge ratio (RV/IV at K) is positive near round strikes
            The vol risk premium (IV - RV) should be SMALLER or NEGATIVE near high-R strikes, meaning gamma is cheap there.
 
  CLAIM 3 — Gamma cascade: GEX integral predicts squeeze magnitude
            The cumulative gamma exposure integral from S to K predicts the actual spot move over the next window better than distance alone.
 
  CLAIM 4 — Strategy selection by R generates positive expected value
            Systematic entry using R-thresholds (strangle / straddle / ratio) with BS-priced options produces positive P&L after theta.
