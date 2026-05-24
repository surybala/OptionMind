_RISK_ORDER = ('SAFE', 'WATCH', 'CAUTION', 'CRITICAL')


def classify_risk_level(pos: dict, stop_loss_multiplier: float) -> str:
    """
    Classify a position's risk into 'SAFE', 'WATCH', 'CAUTION', or 'CRITICAL'.

    Three independent axes — overall level = highest across all three:

    Stop proximity  (current_mark / stop_threshold):
      >= 90 %  -> CRITICAL  |  75-90 %  -> CAUTION  |  50-75 %  -> WATCH
    Profit captured  ((entry - mark) / entry):
      < 0 %   -> CRITICAL  |  0-25 %   -> CAUTION  |  25-50 %  -> WATCH
    Gamma/theta ratio:
      >= 1.5   -> CRITICAL  |  1.2-1.5  -> CAUTION  |  0.8-1.2  -> WATCH

    If *none* of the three axes has data (e.g. chain fetch failed or entry
    premium is missing/negative) the position is classified as 'WATCH' rather
    than 'SAFE' to signal that the risk assessment is incomplete.
    """
    def _bump(current: str, candidate: str) -> str:
        return (candidate
                if _RISK_ORDER.index(candidate) > _RISK_ORDER.index(current)
                else current)

    level    = 'SAFE'
    axes_hit = 0          # count of axes that actually contributed data

    prox = pos.get('stop_proximity_pct')
    if prox is not None:
        axes_hit += 1
        f = prox / 100.0
        if f >= 0.90:
            level = _bump(level, 'CRITICAL')
        elif f >= 0.75:
            level = _bump(level, 'CAUTION')
        elif f >= 0.50:
            level = _bump(level, 'WATCH')

    profit = pos.get('profit_captured_pct')
    if profit is not None:
        axes_hit += 1
        if profit < 0:
            level = _bump(level, 'CRITICAL')
        elif profit < 25:
            level = _bump(level, 'CAUTION')

    gr = pos.get('gamma_theta_ratio')
    if gr is not None:
        axes_hit += 1
        if gr >= 1.5:
            level = _bump(level, 'CRITICAL')
        elif gr >= 1.2:
            level = _bump(level, 'CAUTION')
        elif gr >= 0.8:
            level = _bump(level, 'WATCH')

    # No axis had data — can't assess risk at all; don't report as SAFE.
    if axes_hit == 0:
        return 'WATCH'

    return level
