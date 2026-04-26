# FX_RoundnessStrikeGravity
##Strike Gravity: A Unified Framework for Roundness, Gamma Exposure, and Reflexive Volatility at Psychological Price Levels

[IN PROGRESS]

Note: This project is an unpretentious attempt to give colour to an idea I had at the desk, while my main objectives are to learn and explore the potential of Gamma sizing round numbers and the psychological patterns around them, in an attempt to generate alpha in Forex Options markets.
> _(This will probably be the theme for my Undergraduate Thesis paper too lol)_

This project is based on the conceptual property of roundness as not only the property of a number (e.g. The number 1.5 is theoretically more "round" than 1.68375), but also as the property of how the payoff surface curves around that number. The key is that roundness creates a discrete discontinuity in the distribution of strikes, which shows as a local maximum in the gamma surface, given in (THEORY), in a continuous FX Market, gamma is smooth. Near a round number with high OI(open interest), I am using the backtests to observe and explore whether gamma is a spike, a curvature anomaly, and the spike being the roundness in geometric terms. 

The catch I initially sensed, from observing the markets, is that since gamma proximity is itself a function of implied volatility, IV near round numbers is distorted by the roundness itself, so there is a circularity to it.
The true gamma at a round-number strike is larger than BS predicts from a flat vol surface, because the Open Interest concentration creates a local volatility smile kink(at first, a weird, unstable hedging behaviour).

**A round number is a focal point for stop-losses, option barriers, and institutional hedging mandates.**

Roundness(k) = local excess curvature of the Implied Volatility surface at K (∂²IV/∂K² |_K)

###Realized vol > implied vol at round numbers:
>This claim is what I am empirically testing, and is the heart of the strategy. It can be divided in three mechanisms:
1. **Dealer delta hedging creates realised vol** - They short gamma at Kbuy spot as price falls toward K and sell as it rises. This adds realised vol at the range of K, and the vol added is proportional to the dealer's net short gamma position, reflecting OI.
2. **Reflexivity via pain thresholds** - When a spot approaches a round number, directional traders with stop-losses at that level execute. This creates a clustering of order flow at the levels where dealer gamma hedging is also concentrated. Both of these behavious amplify each other, and this is the reflexive loop. Soro's reflexivity applied to FX options microstructure.
3. **Inflation of put-call partity at the strike** - Put-call parity says C - P = S - Ke^{-rT}. Near a round number with high OI, the observed C - P relationship develops a local distortion because the demand for puts and calls at that specific strike is driven by hedging flow, not just directional views. The bid-ask spread at K is tighter (more liquidity) but the mid-price has a local vol premium (the market is implicitly charging more for the right to participate in the squeeze).

###Back to the gravity formula:
> IF the roundness is real, the local IV at K is elevated (σ_K > σ_ATM), and a higher local vol at K means BS gamma at K is lower (vol spreads the gamma bell curve, reducing the peak)

FIX(for the vol smile kink): For the gravity formula: **Ψ(k) × Ω × Φ** I was using flat IV instead of local vol at K when computing gamma proximity(Φ), so I started using local vol. 
Here, a higher local vol at K means BS gamma at K is lower, but actual hedging flow and realised vol are higher. 
So the formula is slightly inaccurate, because the number is too low in terms of actual squeeze intensity(because σ_K inflates realised vol), and too high in terms of convexity you can buy cheaply(because σ_K means options at K are more expensive)

Solution: Seek the only number that tells you wether the squeeze is worth buying (edge ratio): 
```
R_squeeze  = Ψ × Ω × Γ_local(σ_RV)   ← use realized vol for squeeze sizing
R_cost     = Ψ × Ω × Γ_local(σ_IV)   ← use implied vol for what you pay
Edge       = R_squeeze / R_cost - 1    ← positive edge only if RV > IV at K
```

For exponential bidding and the gamma cascade, the correct model is not linear GEX, but the cumulative ***GEX integral***:
>GEX_cumulative(S→K) = ∫_S^K  Γ(x, K, t, σ) × OI(K) dx
This shows the total dealer hedging obligation as spot traverses the area under the gamma curve between current spot and strike(from S to K).
A higher integral here means a stronger casascade, so I will use this to replace the point estimate of Φ in the formula.

###Pain threshold and inflation pressure layer
This is where everything became incredibly hard, but utterly interesting.
If:
>A round number is a focal point for stop-losses, option barriers, and institutional hedging mandates.

Then as a spot approaches, there is a discrete increase in the number of market participants who must transact, not because of rational valuation, but because their risk systems are triggered; This is the pain threshold behind my entire strategy.
For example,  for USD/JPY 150.00, the pain threshold included the Bank of Japan itself.
There is a nonlinear acceleration that neither the GEX model nor the roundness formula alone captures, but their product R × GEX_integral does, because the exponential bidding where each pip toward K costs more and more, is the pain threshold clustering on top of the dealer gamma cascade.

###Python Backtest
The backtest tests FOUR claims from the theory, each independently:
 
  **CLAIM 1 — Roundness predicts excess realised vol**
            At bars where spot is within N pips of a high-R strike, RV(next M bars) > RV(random bars at same distance from non-round strikes).
 
  **CLAIM 2 — Edge ratio (RV/IV at K) is positive near round strikes**
            The vol risk premium (IV - RV) should be SMALLER or NEGATIVE near high-R strikes, meaning gamma is cheap there.
 
  **CLAIM 3 — Gamma cascade: GEX integral predicts squeeze magnitude**
            The cumulative gamma exposure integral from S to K predicts the actual spot move over the next window better than distance alone.
 
  **CLAIM 4 — Strategy selection by R generates positive expected value**
            Systematic entry using R-thresholds (strangle / straddle / ratio) with BS-priced options produces positive P&L after theta.
