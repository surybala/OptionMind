from .base import StrategyScanner


class CspScanner(StrategyScanner):
    """Cash Secured Put scanner."""

    def scan(self, symbol, current_price, expiry, days, puts, atr=0.0):
        picks = []
        min_premium = self._params.get('min_premium', 0.10)

        for _, row in puts.iterrows():
            strike = row['strike']
            # Basic filter: OTM puts
            if strike >= current_price: continue

            # Minimum % OTM distance from spot
            if self._min_otm_put > 0:
                if strike > current_price * (1.0 - self._min_otm_put):
                    continue

            # ATR-based distance guard
            if self._atr_enabled and atr > 0:
                if abs(current_price - strike) < self._atr_multiplier * atr:
                    continue

            iv = row['impliedVolatility']
            prob_otm = self._prob_otm(row, current_price, strike, days, 'put')

            # Use bid price for short option (what we actually collect at entry)
            bid_price = row.get('bid', 0) or row.get('lastPrice', 0)
            if prob_otm > self._min_prob and bid_price >= min_premium:
                score = bid_price * (prob_otm ** 2)  # prob_win² weights safety over premium
                _oi, _vol = self._row_oi_vol(row)
                raw_roi = (bid_price / strike) if strike > 0 else 0
                ann_roi = raw_roi * (365 / max(1, days))  # annualized for DTE comparison
                picks.append({
                    'strategy': 'CSP',
                    'symbol': symbol,
                    'expiry': expiry,
                    'current_price': round(current_price, 2),
                    'short_strike': strike,
                    'long_strike': None,
                    'premium': bid_price,
                    'max_loss': strike * 100, # theoretically for CSP
                    'prob_win': prob_otm,
                    'roi': round(raw_roi, 4),
                    'annualized_roi': round(ann_roi, 4),
                    'score': round(score, 4),
                    'short_oi': _oi,
                    'short_volume': _vol,
                })
        return picks
