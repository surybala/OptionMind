from .base import StrategyScanner


class CoveredCallScanner(StrategyScanner):
    """Covered Call scanner."""

    def scan(self, symbol, current_price, expiry, days, calls, atr=0.0):
        """
        Covered Call: sell an OTM call against an assumed long stock position.

        Generates income from existing holdings; upside is capped at the short
        strike.  max_loss depends on the holder's cost basis and is left as None.
        Score = premium x prob_OTM².
        """
        picks = []
        params      = self._params
        min_premium = params.get('min_premium', 0.10)

        for _, row in calls.iterrows():
            strike = row['strike']
            if strike <= current_price: continue   # must be OTM

            # Minimum % OTM distance from spot
            if self._min_otm_call > 0:
                if strike < current_price * (1.0 + self._min_otm_call):
                    continue

            # ATR-based distance guard
            if self._atr_enabled and atr > 0:
                if abs(current_price - strike) < self._atr_multiplier * atr:
                    continue

            prob_otm = self._prob_otm(row, current_price, strike, days, 'call')

            premium = row.get('bid', 0) or row.get('lastPrice', 0)

            if prob_otm > self._min_prob and premium >= min_premium:
                score = premium * (prob_otm ** 2)  # prob_win² weights safety over premium
                _oi, _vol = self._row_oi_vol(row)
                picks.append({
                    'strategy':     'CC',
                    'symbol':       symbol,
                    'expiry':       expiry,
                    'current_price': round(current_price, 2),
                    'short_strike': strike,
                    'long_strike':  None,
                    'premium':      premium,
                    'max_loss':     None,   # depends on holder's cost basis
                    'prob_win':     round(prob_otm, 4),
                    'roi':          round(premium / current_price, 6) if current_price > 0 else 0,
                    'score':        round(score, 4),
                    'short_oi':     _oi,
                    'short_volume': _vol,
                })
        return picks
