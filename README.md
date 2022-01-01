# binomial-pricing

Lattice pricing of vanilla options.

Contents:
- CRR calibration and recombining trees
- European and American exercise
- Continuous dividend yield
- Greeks by finite differences
- Convergence study against Black-Scholes
- Self checks: python pricer.py --check

Example:
    python pricer.py --spot 100 --strike 100 --maturity 1.0 \
        --volatility 0.2 --rate 0.03 --dividend 0.01 --steps 500
