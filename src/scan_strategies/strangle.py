from .base import StrategyScanner


class StrangleScanner(StrategyScanner):
    """Short Strangle scanner."""

    def scan(self, symbol, current_price, expiry, days, puts, calls, atr=0.0):
        """
        Short Strangle: sell naked OTM put + naked OTM call.

        Collects the highest absolute premium of all strategies but carries
        undefined downside risk.  max_loss is approximated as put_strike x 100
        (worst-case stock-goes-to-zero scenario) for scoring/comparison only.
        Score = total_credit x min(prob_put_OTM, prob_call_OTM)².
        """
        picks = []
        params     = self._params
        min_credit = params.get('min_net_credit', 0.50)
        min_prob   = params.get('min_prob_profit', 0.75)
        max_delta  = params.get('max_delta_short_leg', 0.20)

        put_map  = {row['strike']: row for _, row in puts.iterrows()}
        call_map = {row['strike']: row for _, row in calls.iterrows()}

        # ── Valid naked put legs ──────────────────────────────────────────────
        valid_puts = []
        for put_strike, put_opt in put_map.items():
            if put_strike >= current_price: continue
            if self._min_otm_put > 0:
                if put_strike > current_price * (1.0 - self._min_otm_put): continue
            if self._atr_enabled and atr > 0:
                if abs(current_price - put_strike) < self._atr_multiplier * atr: continue
            prob_put = self._prob_otm(put_opt, current_price, put_strike, days, 'put')
            if (1.0 - prob_put) > max_delta or prob_put < min_prob: continue
            put_premium = put_opt.get('bid', 0) or put_opt.get('lastPrice', 0)
            if put_premium <= 0: continue
            _poi, _pvol = self._row_oi_vol(put_opt)
            valid_puts.append({
                'put_strike':  put_strike,
                'put_premium': put_premium,
                'prob_put':    prob_put,
                'put_oi':      _poi,
                'put_volume':  _pvol,
            })

        # ── Valid naked call legs ─────────────────────────────────────────────
        valid_calls = []
        for call_strike, call_opt in call_map.items():
            if call_strike <= current_price: continue
            if self._min_otm_call > 0:
                if call_strike < current_price * (1.0 + self._min_otm_call): continue
            if self._atr_enabled and atr > 0:
                if abs(current_price - call_strike) < self._atr_multiplier * atr: continue
            prob_call = self._prob_otm(call_opt, current_price, call_strike, days, 'call')
            if (1.0 - prob_call) > max_delta or prob_call < min_prob: continue
            call_premium = call_opt.get('bid', 0) or call_opt.get('lastPrice', 0)
            if call_premium <= 0: continue
            _coi, _cvol = self._row_oi_vol(call_opt)
            valid_calls.append({
                'call_strike':  call_strike,
                'call_premium': call_premium,
                'prob_call':    prob_call,
                'call_oi':      _coi,
                'call_volume':  _cvol,
            })

        # ── Combine ───────────────────────────────────────────────────────────
        for pc in valid_puts:
            for cc in valid_calls:
                if pc['put_strike'] >= cc['call_strike']: continue

                total_credit   = pc['put_premium'] + cc['call_premium']
                if total_credit < min_credit: continue

                # Proxy max loss: downside if stock collapses to 0
                max_loss_proxy = pc['put_strike'] * 100
                prob_win       = min(pc['prob_put'], cc['prob_call'])
                roi            = (total_credit / max_loss_proxy) if max_loss_proxy > 0 else 0
                score          = total_credit * (prob_win ** 2)  # prob_win² weights safety over premium

                _st_oi  = (min(pc['put_oi'], cc['call_oi'])
                           if pc['put_oi'] is not None and cc['call_oi'] is not None
                           else (pc['put_oi'] or cc['call_oi']))
                _st_vol = min(pc['put_volume'], cc['call_volume'])
                picks.append({
                    'strategy':    'STRANGLE',
                    'symbol':      symbol,
                    'expiry':      expiry,
                    'current_price': round(current_price, 2),
                    'short_put':   pc['put_strike'],
                    'short_call':  cc['call_strike'],
                    'premium':     round(total_credit, 2),
                    'max_loss':    round(max_loss_proxy, 2),
                    'prob_win':    round(prob_win, 4),
                    'prob_put':    round(pc['prob_put'], 4),
                    'prob_call':   round(cc['prob_call'], 4),
                    'roi':         round(roi, 6),
                    'score':       round(score, 4),
                    'short_oi':     _st_oi,
                    'short_volume': _st_vol,
                })

        return picks
