from .base import StrategyScanner


class IronButterflyScanner(StrategyScanner):
    """Iron Butterfly scanner."""

    def scan(self, symbol, current_price, expiry, days, puts, calls, atr=0.0):
        """
        Iron Butterfly: sell ATM put + sell ATM call at the same strike,
        protected by an OTM long put (lower wing) and OTM long call (upper wing).

        Why low-capital?
        ----------------
        Like an Iron Condor the maximum loss is capped at wing_width x 100 minus the
        net credit received — the broker holds that spread width as collateral rather
        than the full stock price.  For a $100 stock with $10 wings the capital
        requirement is $1,000/contract versus $10,000 for a Covered Call.

        vs Iron Condor
        --------------
        Short strikes are ATM (higher premium) instead of OTM (lower premium).  The
        tradeoff is a narrower profit window: max profit is realised only when the
        stock closes exactly at the short strike, but partial profit is kept whenever
        the stock closes between the breakeven points
            lower BE = short_strike - net_credit
            upper BE = short_strike + net_credit

        Scoring
        -------
        prob_win = min(P(S > long_put), P(S < long_call)), i.e. the probability that
        neither wing is breached at expiry (no maximum-loss scenario).  This is higher
        than the probability of full profit, but is the appropriate risk measure for
        position sizing.

        Capital = max(put_wing_width, call_wing_width) x 100
        """
        picks       = []
        params      = self._params
        put_wing    = params.get('put_wing_width',    10)
        call_wing   = params.get('call_wing_width',   10)
        min_credit  = params.get('min_net_credit',   1.50)
        min_prob    = params.get('min_prob_profit',  0.60)
        atm_tol_pct = params.get('atm_pct_tolerance', 0.025)  # ±2.5% of spot

        put_map  = {row['strike']: row for _, row in puts.sort_values('strike').iterrows()}
        call_map = {row['strike']: row for _, row in calls.sort_values('strike').iterrows()}

        atm_band = current_price * atm_tol_pct

        for short_strike, put_opt in put_map.items():
            # Short strike must be near ATM
            if abs(short_strike - current_price) > atm_band:
                continue
            # Matching call option at the same strike required
            if short_strike not in call_map:
                continue

            long_put_strike  = short_strike - put_wing
            long_call_strike = short_strike + call_wing

            if long_put_strike  not in put_map:  continue
            if long_call_strike not in call_map: continue

            call_opt       = call_map[short_strike]
            long_put_opt   = put_map[long_put_strike]
            long_call_opt  = call_map[long_call_strike]

            short_put_bid  = put_opt.get('bid',  0) or put_opt.get('lastPrice',  0)
            short_call_bid = call_opt.get('bid', 0) or call_opt.get('lastPrice', 0)
            long_put_ask   = long_put_opt.get('ask',  0) or long_put_opt.get('lastPrice',  0)
            long_call_ask  = long_call_opt.get('ask', 0) or long_call_opt.get('lastPrice', 0)

            net_credit = short_put_bid + short_call_bid - long_put_ask - long_call_ask
            if net_credit < min_credit:
                continue

            max_loss = max(put_wing, call_wing) - net_credit
            if max_loss <= 0:
                max_loss = 0.01

            # Probability measure: both wings stay OTM (no max-loss scenario)
            prob_put_wing  = self._prob_otm(long_put_opt,  current_price, long_put_strike,  days, 'put')
            prob_call_wing = self._prob_otm(long_call_opt, current_price, long_call_strike, days, 'call')
            prob_win = min(prob_put_wing, prob_call_wing)

            if prob_win < min_prob:
                continue

            roi   = net_credit / max_loss
            score = net_credit * (prob_win ** 2)  # prob_win² weights safety over premium

            _ifly_oi_p, _ifly_vol_p = self._row_oi_vol(put_opt)
            _ifly_oi_c, _ifly_vol_c = self._row_oi_vol(call_opt)
            _ifly_oi  = (min(_ifly_oi_p, _ifly_oi_c)
                         if _ifly_oi_p is not None and _ifly_oi_c is not None
                         else (_ifly_oi_p or _ifly_oi_c))
            _ifly_vol = min(_ifly_vol_p, _ifly_vol_c)
            picks.append({
                'strategy':        'IFLY',
                'symbol':          symbol,
                'expiry':          expiry,
                'current_price':   round(current_price, 2),
                'short_put':       short_strike,
                'short_call':      short_strike,   # same ATM strike
                'long_put':        long_put_strike,
                'long_call':       long_call_strike,
                'put_wing':        put_wing,
                'call_wing':       call_wing,
                'premium':         round(net_credit, 2),
                'max_loss':        round(max_loss, 2),
                'prob_win':        round(prob_win, 4),
                'prob_put_wing':   round(prob_put_wing, 4),
                'prob_call_wing':  round(prob_call_wing, 4),
                'roi':             round(roi, 4),
                'score':           round(score, 4),
                'short_oi':        _ifly_oi,
                'short_volume':    _ifly_vol,
            })

        return picks
