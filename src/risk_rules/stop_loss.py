import logging
from typing import Optional
from .base import CloseSignal

_log = logging.getLogger('optionwheel')


def _width_from_legs(pos: dict) -> Optional[float]:
    """Derive spread width in dollars from the position's stored leg strikes.

    Returns None for strategies without a defined width (CSP, CC, STRANGLE).
    """
    legs  = pos.get('legs') or {}
    strat = pos.get('type', '')
    try:
        if strat in ('PCS', 'CCS'):
            ss = float(legs.get('short_strike') or 0)
            ls = float(legs.get('long_strike')  or 0)
            if ss > 0 and ls > 0:
                return abs(ss - ls)
        elif strat in ('IC', 'IFLY'):
            sp = float(legs.get('short_put')  or 0)
            lp = float(legs.get('long_put')   or 0)
            sc = float(legs.get('short_call') or 0)
            lc = float(legs.get('long_call')  or 0)
            put_w  = abs(sp - lp) if sp > 0 and lp > 0 else 0.0
            call_w = abs(sc - lc) if sc > 0 and lc > 0 else 0.0
            w = max(put_w, call_w)
            if w > 0:
                return w
    except (TypeError, ValueError):
        pass
    return None


class StopLossRule:
    """Close a position when its unrealised loss exceeds a threshold.

    Two complementary guards — whichever fires first wins:

    1. **Premium-multiplier** (legacy): fires when
       ``current_mark > (1 + multiplier) × entry_premium``.

    2. **Width-relative** (primary for spreads): fires when the spread has
       lost ``max_loss_pct × max_loss`` dollars, where
       ``max_loss = spread_width - entry_premium``.  This prevents
       hair-trigger exits on low-premium / far-OTM spreads whose 2× premium
       threshold sits just a few cents above entry noise.

    When ``max_loss_pct`` is None the width-relative guard is disabled and
    only the premium-multiplier applies (original behaviour).
    """
    name = 'STOP_LOSS'

    def __init__(self, stop_loss_multiplier: float,
                 max_loss_pct: Optional[float] = None):
        self._multiplier   = stop_loss_multiplier
        self._max_loss_pct = max_loss_pct

    def evaluate(self, entry_premium: float, current_mark: float,
                 pnl_per_share: float, spot: Optional[float],
                 pos: dict, **kwargs) -> Optional[CloseSignal]:
        loss = current_mark - entry_premium

        # ── Width-relative guard (primary for spreads) ────────────────────────
        # When max_loss_pct is configured and the position is a spread, this
        # guard *replaces* the premium-multiplier.  Firing at a fixed % of max
        # possible loss gives far-OTM spreads room proportional to their actual
        # capital at risk rather than a hair-trigger 2× the (tiny) premium.
        if self._max_loss_pct is not None:
            width = _width_from_legs(pos)
            if width is not None and width > entry_premium:
                max_loss_dollars = width - entry_premium
                width_trig = self._max_loss_pct * max_loss_dollars
                triggered  = loss >= width_trig
                trig_label = (
                    f"{self._max_loss_pct:.0%} of max-loss "
                    f"(width={width:.0f}, trig={width_trig:.2f})"
                )
                if not triggered:
                    return None
                reason_str = (f"current_mark={current_mark:.2f} > "
                              f"entry={entry_premium:.2f} + {trig_label}")
                extras_str = f"mark={current_mark:.2f}, entry={entry_premium:.2f}"
                return CloseSignal(
                    reason_tag='STOP_LOSS',
                    reason_str=reason_str,
                    extras_str=extras_str,
                    metrics={'pnl_per_share': pnl_per_share},
                )

        # ── Premium-multiplier guard (fallback: non-spread positions or legacy) ─
        mult_trig  = self._multiplier * entry_premium
        triggered  = loss >= mult_trig
        trig_label = f"{self._multiplier}× premium ({mult_trig:.2f})"

        if not triggered:
            return None

        reason_str = (f"current_mark={current_mark:.2f} > "
                      f"entry={entry_premium:.2f} + {trig_label}")
        extras_str = f"mark={current_mark:.2f}, entry={entry_premium:.2f}"
        return CloseSignal(
            reason_tag='STOP_LOSS',
            reason_str=reason_str,
            extras_str=extras_str,
            metrics={'pnl_per_share': pnl_per_share},
        )
