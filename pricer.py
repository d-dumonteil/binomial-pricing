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
    compounded.
    """

    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float = 0.0
    dividend: float = 0.0
    option_type: OptionType = "call"
    exercise: Exercise = "european"

    def validate(self) -> None:
        """Reject degenerate inputs early."""
        if self.spot <= 0.0 or self.strike <= 0.0:
            raise ValueError("spot and strike must be positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.volatility < 0.0:
            raise ValueError("volatility must be non-negative")

    def intrinsic(self, spot: float) -> float:
        """Immediate exercise value at the given spot level."""
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)


def _single_period(opt: OptionContract, spot: float) -> tuple[float, float, float]:
    """One-step risk-neutral valuation: the base case of the lattice.

    Returns (u, d, discounted option price).
    """
    growth = math.exp((opt.rate - opt.dividend) * opt.maturity)
    u = math.exp(opt.volatility * math.sqrt(opt.maturity))
    d = 1.0 / u
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("risk-neutral probability outside [0, 1]")
    disc = math.exp(-opt.rate * opt.maturity)
    price = disc * (
        p * opt.intrinsic(spot * u) + (1.0 - p) * opt.intrinsic(spot * d)
    )
    return u, d, price


@dataclass(frozen=True)
class CRRParams:
    """Cox-Ross-Rubinstein lattice parameters for a step count.

    u = 1/d = exp(sigma sqrt(dt)) keeps the tree recombining and
    the terminal layer log-centered around the forward.
    """

    steps: int
    dt: float
    u: float
    d: float
    p: float
    disc: float

    @classmethod
    def build(cls, opt: OptionContract, steps: int) -> "CRRParams":
        """Calibrate the lattice for the given contract."""
        if steps < 1:
            raise ValueError("steps must be at least 1")
        dt = opt.maturity / steps
        u = math.exp(opt.volatility * math.sqrt(dt))
        d = 1.0 / u
        growth = math.exp((opt.rate - opt.dividend) * dt)
        p = (growth - d) / (u - d)
        if not 0.0 < p < 1.0:
            raise ValueError("arbitrage condition violated for these inputs")
        disc = math.exp(-opt.rate * dt)
        return cls(steps, dt, u, d, p, disc)


def _terminal_values(opt: OptionContract, par: CRRParams) -> list[float]:
    """Terminal node spots S0 u^(n-j) d^j with their payoffs."""
    return [
        opt.intrinsic(opt.spot * (par.u ** (par.steps - j)) * (par.d ** j))
        for j in range(par.steps + 1)
    ]


def price_european(opt: OptionContract, par: CRRParams) -> float:
    """Backward induction on the terminal payoff vector.

    O(n) space, one convex combination per parent node.
    """
    values = _terminal_values(opt, par)
    q = 1.0 - par.p
    for _ in range(par.steps):
        values = [
            par.disc * (par.p * values[j] + q * values[j + 1])
            for j in range(len(values) - 1)
        ]
    return values[0]


def price_american(opt: OptionContract, par: CRRParams) -> float:
    """Backward induction with early exercise at every interior node.

    At each node the holder compares the continuation value with
    the intrinsic value; the scheme resolves the free boundary.
    """
    values = _terminal_values(opt, par)
    q = 1.0 - par.p
    for k in range(par.steps - 1, -1, -1):
        values = [
            max(
                par.disc * (par.p * values[j] + q * values[j + 1]),
                opt.intrinsic(opt.spot * (par.u ** (k - j)) * (par.d ** j)),
            )
            for j in range(k + 1)
        ]
    return values[0]


def forward_price(opt: OptionContract) -> float:
    """Risk-neutral forward S0 exp((r - q) T)."""
    return opt.spot * math.exp((opt.rate - opt.dividend) * opt.maturity)


def price(opt: OptionContract, steps: int) -> float:
    """Dispatcher honouring the exercise style."""
    opt = replace(opt, dividend=max(opt.dividend, 0.0))
    par = CRRParams.build(opt, steps)
    if opt.exercise == "american":
        return price_american(opt, par)
    return price_european(opt, par)


def parity_gap(opt: OptionContract, steps: int) -> float:
    """|C - P - (S e^{-qT} - K e^{-rT})| for a European pair."""
    call = replace(opt, option_type="call", exercise="european")
    put = replace(opt, option_type="put", exercise="european")
    lhs = price(call, steps) - price(put, steps)
    rhs = opt.spot * math.exp(-opt.dividend * opt.maturity)
    rhs -= opt.strike * math.exp(-opt.rate * opt.maturity)
    return abs(lhs - rhs)


def sanity_checks(opt: OptionContract, steps: int) -> bool:
    """Cheap invariants: dominance relations and parity."""
    call = replace(opt, option_type="call")
    put = replace(opt, option_type="put")
    if price(replace(call, exercise="american"), steps) < price(
        replace(call, exercise="european"), steps
    ) - 1e-12:
        return False
    if price(replace(put, exercise="american"), steps) < price(
        replace(put, exercise="european"), steps
    ) - 1e-12:
        return False
    return parity_gap(opt, steps) < 1e-6


def _spot_bump_prices(opt: OptionContract, steps: int, rel: float = 1e-4):
    """Prices at S0-h, S0, S0+h for central differences."""
    h = rel * opt.spot
    up = price(replace(opt, spot=opt.spot + h), steps)
    mid = price(opt, steps)
    down = price(replace(opt, spot=opt.spot - h), steps)
    return down, mid, up, h


