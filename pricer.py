"""Binomial option pricing library.

Lattice pricing of vanilla options: CRR calibration, European and
American exercise, continuous dividend yields, Greeks by finite
differences, and a convergence study against Black-Scholes.

CLI usage:
    python pricer.py --spot 100 --strike 100 --maturity 1.0 \
        --volatility 0.2 --rate 0.03 --steps 500
Internal checks:
    python pricer.py --check
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Literal

OptionType = Literal["call", "put"]
Exercise = Literal["european", "american"]


@dataclass(frozen=True)
class OptionContract:
    """Vanilla option specification.

    spot and strike in currency units, maturity in years,
    volatility annualised, rate and dividend continuously
