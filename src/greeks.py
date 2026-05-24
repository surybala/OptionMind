"""
greeks.py
=========

Black-Scholes Greeks calculator for option positions.

Used by PositionMonitor to compute a dynamic risk score that detects when a
short-premium position is becoming dangerously close to expiring in-the-money,
allowing the agent to lock in remaining premium before it erodes further.

Risk-score logic
----------------
For short-premium strategies (CSP, PCS, CCS, IC, IFLY, STRANGLE, CC) we collect
premium upfront and want it to expire worthless. The main Greek risks are:

  Gamma  — negative for short options: large moves hurt us more than BS predicts
  Theta  — positive for short options: time decay earns us money each day

The **gamma/theta ratio** measures how much gamma risk we carry per dollar of
daily theta income.  A rising ratio means the position is becoming more
dangerous relative to the premium we're still earning.

  gamma_theta_ratio = |net_gamma| / |net_theta_per_day|

Additional signal: **short delta** (absolute value of the net delta of the short
legs). A CSP short put with delta approaching -0.30 is no longer "safely OTM".

Combined risk score
-------------------
  risk_score = gamma_theta_ratio * (1 + short_delta_penalty)

where:
  short_delta_penalty = max(0, |short_delta| - delta_neutral_zone) / delta_neutral_zone

All values are per-share (standard Black-Scholes, not per-contract).

Public API
----------
  bs_greeks(spot, strike, iv, dte_days, option_type) -> dict
      Returns {'delta', 'gamma', 'theta', 'vega'}.
      Uses risk-free rate r=0 (conservative; standard for short-dated equity options).

  position_risk_score(spot, legs, dte_days) -> dict
      legs: list of dicts with keys:
          'strike', 'iv', 'option_type' ('put'|'call'),
          'position' ('short'|'long')
      Returns:
          {
            'risk_score':        float,   # composite risk score (higher = riskier)
            'gamma_theta_ratio': float,   # |net_gamma| / |net_theta|
            'net_short_delta':   float,   # net delta of short legs only
            'net_gamma':         float,   # portfolio gamma (negative for net-short)
            'net_theta':         float,   # portfolio theta per day (positive = earn)
            'net_vega':          float,   # portfolio vega (negative for net-short vol)
          }
"""
from __future__ import annotations

import math
from typing import Optional


# ── Internal helpers ──────────────────────────────────────────────────────────

def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ── Public API ────────────────────────────────────────────────────────────────

def bs_greeks(
    spot: float,
    strike: float,
    iv: float,
    dte_days: float,
    option_type: str,          # 'put' | 'call'
    r: float = 0.0,            # risk-free rate (annualised)
) -> dict:
    """
    Compute Black-Scholes delta, gamma, theta and vega for a single option leg.

    Parameters
    ----------
    spot        : current underlying price
    strike      : option strike price
    iv          : implied volatility (annualised, e.g. 0.30 for 30%)
    dte_days    : calendar days to expiration (may be fractional)
    option_type : 'put' or 'call' (case-insensitive)
    r           : risk-free rate; defaults to 0

    Returns
    -------
    dict with keys 'delta', 'gamma', 'theta', 'vega'.
    All values are per-share. Theta is per calendar day (negative for long).
    Vega is per 1.00 change in IV, so a +1 vol-point shock uses vega * 0.01.

    Edge cases
    ----------
    - dte_days <= 0  → delta is 1 (call ITM) / 0 (call OTM) style at expiry;
      gamma and theta are 0.
    - iv <= 0        → same intrinsic-value snap; gamma and theta are 0.
    """
    opt = option_type.lower()

    # Guard against degenerate inputs
    if dte_days <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        # At expiry, intrinsic value delta only
        if opt == 'call':
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

    T = dte_days / 365.0          # time in years
    sqrtT = math.sqrt(T)

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * sqrtT)
        d2 = d1 - iv * sqrtT

        phi_d1 = _phi(d1)
        cdf_d1 = _cdf(d1)
        cdf_d2 = _cdf(d2)

        # Gamma is the same for puts and calls
        gamma = phi_d1 / (spot * iv * sqrtT)
        vega = spot * phi_d1 * sqrtT

        if opt == 'call':
            delta = math.exp(-r * T) * cdf_d1          # ≈ cdf_d1 at r=0
            # Theta per year (negative = time decay costs the holder)
            theta_annual = (
                -(spot * phi_d1 * iv) / (2.0 * sqrtT)
                - r * strike * math.exp(-r * T) * cdf_d2
            )
        else:  # put
            delta = math.exp(-r * T) * (cdf_d1 - 1.0)  # negative
            theta_annual = (
                -(spot * phi_d1 * iv) / (2.0 * sqrtT)
                + r * strike * math.exp(-r * T) * (1.0 - cdf_d2)
            )

        theta_per_day = theta_annual / 365.0   # negative for long, positive sign convention below

        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 8),
            'theta': round(theta_per_day, 6),   # per calendar day, holder's perspective
            'vega': round(vega, 6),
        }

    except (ValueError, ZeroDivisionError):
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}


def _risk_score_from_net_greeks(
    net_delta: float,
    net_gamma: float,
    net_theta: float,
    net_vega: float,
    net_short_delta: float,
) -> dict:
    """
    Compute the composite risk score from already-accumulated portfolio Greeks.

    Shared by ``position_risk_score()`` and ``position_risk_score_from_greeks()``;
    neither function should duplicate this logic.

    Parameters
    ----------
    net_delta       : portfolio delta across all legs
    net_gamma       : portfolio gamma (negative for a net-short position)
    net_theta       : portfolio theta per day (positive = we earn time decay)
    net_vega        : portfolio vega
    net_short_delta : sum of raw deltas for short legs only (sign preserved)

    Returns
    -------
    dict with keys: risk_score, gamma_theta_ratio, net_short_delta,
                    net_gamma, net_theta
    """
    abs_gamma = abs(net_gamma)
    abs_theta = abs(net_theta)

    # gamma/theta ratio: how much gamma risk per dollar of daily theta income.
    # When theta ≈ 0 (near expiry or far OTM) use abs_gamma × 1000 as a
    # large-but-finite sentinel so downstream comparisons still work.
    if abs_theta < 1e-8:
        gamma_theta_ratio = abs_gamma * 1000.0 if abs_gamma > 0 else 0.0
    else:
        gamma_theta_ratio = abs_gamma / abs_theta

    # Delta penalty: penalise short legs moving in-the-money.
    abs_short_delta    = abs(net_short_delta)
    delta_neutral_zone = 0.15   # below this we consider "safely OTM"
    if abs_short_delta > delta_neutral_zone:
        delta_penalty = (abs_short_delta - delta_neutral_zone) / delta_neutral_zone
    else:
        delta_penalty = 0.0

    risk_score = gamma_theta_ratio * (1.0 + delta_penalty)

    return {
        'risk_score':        round(risk_score,        4),
        'gamma_theta_ratio': round(gamma_theta_ratio, 4),
        'net_delta':         round(net_delta,          6),
        'net_short_delta':   round(net_short_delta,   6),
        'net_gamma':         round(net_gamma,          8),
        'net_theta':         round(net_theta,          6),
        'net_vega':          round(net_vega,           6),
    }


_ZERO_RISK = {
    'risk_score':        0.0,
    'gamma_theta_ratio': 0.0,
    'net_delta':         0.0,
    'net_short_delta':   0.0,
    'net_gamma':         0.0,
    'net_theta':         0.0,
    'net_vega':          0.0,
}


def position_risk_score_from_greeks(legs_with_greeks: list[dict]) -> dict:
    """
    Compute a composite risk score for a multi-leg position using
    **broker-supplied** Greeks (e.g. from Alpaca's OptionsSnapshot).

    This is the HFT-mode equivalent of ``position_risk_score()``: same
    maths, but accepts pre-fetched delta/gamma/theta values instead of
    running Black-Scholes internally.  Use this when the broker already
    provides accurate, surface-aware Greeks rather than flat B-S estimates.

    Parameters
    ----------
    legs_with_greeks : list of dicts, each with:
        'delta'    : float  — option delta from broker snapshot
        'gamma'    : float  — option gamma from broker snapshot
        'theta'    : float  — option theta (per day) from broker snapshot
        'position' : 'short' | 'long'

    Any leg missing delta/gamma/theta is skipped.

    Returns
    -------
    Same dict shape as ``position_risk_score()``:
        risk_score, gamma_theta_ratio, net_short_delta, net_gamma, net_theta
    """
    if not legs_with_greeks:
        return dict(_ZERO_RISK)

    net_delta = net_gamma = net_theta = net_vega = net_short_delta = 0.0

    for leg in legs_with_greeks:
        delta = leg.get('delta')
        gamma = leg.get('gamma')
        theta = leg.get('theta')
        vega = leg.get('vega', 0.0)
        pos   = str(leg.get('position', 'short')).lower()

        if delta is None or gamma is None or theta is None:
            continue  # broker didn't supply greeks for this leg — skip

        sign       = -1.0 if pos == 'short' else 1.0
        net_delta += sign * float(delta)
        net_gamma += sign * float(gamma)
        net_theta += sign * float(theta)
        net_vega += sign * float(vega or 0.0)

        if pos == 'short':
            net_short_delta += float(delta)

    return _risk_score_from_net_greeks(
        net_delta, net_gamma, net_theta, net_vega, net_short_delta
    )


def position_risk_score(
    spot: float,
    legs: list[dict],
    dte_days: float,
) -> dict:
    """
    Compute a composite risk score for a multi-leg option position.

    Parameters
    ----------
    spot      : current underlying price
    dte_days  : calendar days to expiration
    legs      : list of leg dicts, each with:
                  'strike'      : float
                  'iv'          : float  (annualised)
                  'option_type' : 'put' | 'call'
                  'position'    : 'short' | 'long'

    Returns
    -------
    dict:
        risk_score        — composite, higher is riskier (0 = no risk data)
        gamma_theta_ratio — |net_gamma| / |net_theta_per_day|;
                            higher = gamma outpacing theta income
        net_short_delta   — sum of deltas for *short* legs (sign preserved)
        net_gamma         — total portfolio gamma (negative = net-short gamma)
        net_theta         — total portfolio theta/day from holder's perspective
                            (positive for net-short positions = we earn)
    """
    if not legs or spot <= 0 or dte_days <= 0:
        return dict(_ZERO_RISK)

    net_delta = net_gamma = net_theta = net_vega = net_short_delta = 0.0

    for leg in legs:
        strike = float(leg.get('strike', 0) or 0)
        iv     = float(leg.get('iv', 0)     or 0)
        opt    = str(leg.get('option_type', 'put')).lower()
        pos    = str(leg.get('position',   'short')).lower()

        if strike <= 0 or iv <= 0:
            continue

        g    = bs_greeks(spot, strike, iv, dte_days, opt)
        sign = -1.0 if pos == 'short' else 1.0

        net_delta += sign * g['delta']
        net_gamma += sign * g['gamma']
        net_theta += sign * g['theta']
        net_vega += sign * g['vega']

        if pos == 'short':
            net_short_delta += g['delta']

    return _risk_score_from_net_greeks(
        net_delta, net_gamma, net_theta, net_vega, net_short_delta
    )
