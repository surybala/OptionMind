from .base import StrategyScanner


class IronCondorScanner(StrategyScanner):
    """Iron Condor scanner."""

    def scan(self, symbol, current_price, expiry, days, puts, calls,
             put_delta_override=None, call_delta_override=None, sentiment=None,
             put_width_override=None, call_width_override=None, atr=0.0):
        """
        Iron Condor = Put Credit Spread + Call Credit Spread on the same expiry.

        Collects premium on both sides simultaneously.  Because only one side
        can expire ITM at expiration, max loss = max(put_width, call_width) - total_credit.
        Score = total_credit x min(prob_put_OTM, prob_call_OTM).

        put_delta_override / call_delta_override: sentiment-adjusted delta ceilings
        applied independently to each leg (BULL lowers call ceiling, BEAR lowers put ceiling).

        put_width_override / call_width_override: override wing widths for high-IV names.
        When set, these replace the put_strike_width / call_strike_width from config.
        """
        picks = []
        params   = self._params
        default_put_w  = self._width_for_price(current_price, params.get('put_strike_width',  5))
        default_call_w = self._width_for_price(current_price, params.get('call_strike_width', 5))
        put_width  = put_width_override  if put_width_override  is not None else default_put_w
        call_width = call_width_override if call_width_override is not None else default_call_w
        min_credit = params.get('min_net_credit', 0.40)
        min_prob   = params.get('min_prob_profit', 0.70)
        min_oi     = params.get('min_open_interest', 0)
        base_delta = params.get('max_delta_short_leg', 0.25)
        # Use per-leg sentiment-adjusted deltas, fall back to shared base
        put_max_delta  = put_delta_override  if put_delta_override  is not None else base_delta
        call_max_delta = call_delta_override if call_delta_override is not None else base_delta

        # Build strike -> row lookup maps for puts and calls
        puts_sorted  = puts.sort_values('strike', ascending=True)
        calls_sorted = calls.sort_values('strike', ascending=True)
        put_map  = {row['strike']: row for _, row in puts_sorted.iterrows()}
        call_map = {row['strike']: row for _, row in calls_sorted.iterrows()}

        # ── Valid put legs ────────────────────────────────────────────────────
        valid_put_legs = []
        for short_put, put_opt in put_map.items():
            if short_put >= current_price: continue           # must be OTM
            if short_put < current_price * 0.30: continue    # sanity: >70% below spot -> price error
            if self._min_otm_put > 0:
                if short_put > current_price * (1.0 - self._min_otm_put): continue
            if self._atr_enabled and atr > 0:
                if abs(current_price - short_put) < self._atr_multiplier * atr: continue
            iv = put_opt['impliedVolatility']
            prob_put = self._prob_otm(put_opt, current_price, short_put, days, 'put')
            if (1.0 - prob_put) > put_max_delta or prob_put < min_prob: continue

            long_put = short_put - put_width
            if long_put not in put_map: continue

            # OI liquidity filter (-1 = unknown from Alpaca, skip filter)
            if min_oi > 0:
                short_put_oi = int(put_opt.get('openInterest') or 0)
                long_put_oi  = int(put_map[long_put].get('openInterest') or 0)
                if short_put_oi != -1 and long_put_oi != -1:
                    if short_put_oi < min_oi or long_put_oi < min_oi: continue

            short_bid = put_opt.get('bid', 0) or put_opt.get('lastPrice', 0)
            long_ask  = put_map[long_put].get('ask', 0) or put_map[long_put].get('lastPrice', 0)
            put_credit = short_bid - long_ask
            if put_credit <= 0: continue

            _poi, _pvol = self._row_oi_vol(put_opt)
            valid_put_legs.append({
                'short_put': short_put,
                'long_put': long_put,
                'put_credit': put_credit,
                'short_put_iv': float(put_opt.get('impliedVolatility') or 0),
                'long_put_iv': float(put_map[long_put].get('impliedVolatility') or 0),
                'prob_put': prob_put,
                'short_put_oi': _poi,
                'short_put_volume': _pvol,
            })

        # ── Valid call legs ───────────────────────────────────────────────────
        valid_call_legs = []
        for short_call, call_opt in call_map.items():
            if short_call <= current_price: continue          # must be OTM
            if short_call > current_price * 2.00: continue   # sanity: >200% of spot -> price error
            if self._min_otm_call > 0:
                if short_call < current_price * (1.0 + self._min_otm_call): continue
            if self._atr_enabled and atr > 0:
                if abs(current_price - short_call) < self._atr_multiplier * atr: continue
            prob_call = self._prob_otm(call_opt, current_price, short_call, days, 'call')
            if (1.0 - prob_call) > call_max_delta or prob_call < min_prob: continue

            long_call = short_call + call_width
            if long_call not in call_map: continue

            # OI liquidity filter (-1 = unknown from Alpaca, skip filter)
            if min_oi > 0:
                short_call_oi = int(call_opt.get('openInterest') or 0)
                long_call_oi  = int(call_map[long_call].get('openInterest') or 0)
                if short_call_oi != -1 and long_call_oi != -1:
                    if short_call_oi < min_oi or long_call_oi < min_oi: continue

            short_bid = call_opt.get('bid', 0) or call_opt.get('lastPrice', 0)
            long_ask  = call_map[long_call].get('ask', 0) or call_map[long_call].get('lastPrice', 0)
            call_credit = short_bid - long_ask
            if call_credit <= 0: continue

            _coi, _cvol = self._row_oi_vol(call_opt)
            valid_call_legs.append({
                'short_call': short_call,
                'long_call': long_call,
                'call_credit': call_credit,
                'short_call_iv': float(call_opt.get('impliedVolatility') or 0),
                'long_call_iv': float(call_map[long_call].get('impliedVolatility') or 0),
                'prob_call': prob_call,
                'short_call_oi': _coi,
                'short_call_volume': _cvol,
            })

        # ── Combine: every valid put leg x every valid call leg ───────────────
        for pl in valid_put_legs:
            for cl in valid_call_legs:
                # Short put must be below short call (non-overlapping wings)
                if pl['short_put'] >= cl['short_call']: continue

                total_credit = pl['put_credit'] + cl['call_credit']
                if total_credit < min_credit: continue

                max_loss = max(put_width, call_width) - total_credit
                if max_loss <= 0: max_loss = 0.01

                prob_win     = min(pl['prob_put'], cl['prob_call'])
                roi          = total_credit / max_loss
                annualized_roi = roi * (365 / max(1, days))  # annualized for DTE comparison
                # Yield-normalised: (credit / max_wing_width) × prob²
                score    = self._score(total_credit, prob_win, max(put_width, call_width))

                # short_oi / short_volume: weakest-link of the two short legs
                _ic_oi  = (min(pl['short_put_oi'],     cl['short_call_oi'])
                           if pl['short_put_oi'] is not None and cl['short_call_oi'] is not None
                           else (pl['short_put_oi'] or cl['short_call_oi']))
                _ic_vol = min(pl['short_put_volume'], cl['short_call_volume'])
                ic_pick = {
                    'strategy':   'IC',
                    'symbol':     symbol,
                    'expiry':     expiry,
                    'current_price': round(current_price, 2),
                    'short_put':  pl['short_put'],
                    'long_put':   pl['long_put'],
                    'short_call': cl['short_call'],
                    'long_call':  cl['long_call'],
                    'put_width':  put_width,
                    'call_width': call_width,
                    'premium':    round(total_credit, 2),
                    'max_loss':   round(max_loss, 2),
                    'short_put_iv':  round(pl['short_put_iv'], 4),
                    'long_put_iv':   round(pl['long_put_iv'], 4),
                    'short_call_iv': round(cl['short_call_iv'], 4),
                    'long_call_iv':  round(cl['long_call_iv'], 4),
                    'prob_win':   round(prob_win, 4),
                    'prob_put':   round(pl['prob_put'], 4),
                    'prob_call':  round(cl['prob_call'], 4),
                    'roi':        round(roi, 4),
                    'annualized_roi': round(annualized_roi, 4),
                    'score':      round(score, 4),
                    'short_oi':     _ic_oi,
                    'short_volume': _ic_vol,
                }
                if sentiment:
                    ic_pick['sentiment']          = sentiment.get('sentiment', 'NEUTRAL')
                    ic_pick['sentiment_strength']  = sentiment.get('strength', 0.0)
                    ic_pick['put_delta_used']      = round(put_max_delta, 4)
                    ic_pick['call_delta_used']     = round(call_max_delta, 4)
                picks.append(ic_pick)

        return picks
