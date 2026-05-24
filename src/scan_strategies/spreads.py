from .base import StrategyScanner


class SpreadsScanner(StrategyScanner):
    """Put Credit Spread and Call Credit Spread scanner."""

    def __init__(self, pcs_params: dict, ccs_params: dict, **common):
        # Use pcs_params as the primary params; store ccs_params separately
        super().__init__(pcs_params, **common)
        self._pcs_params = pcs_params
        self._ccs_params = ccs_params

    def scan(self, symbol, current_price, expiry, days, chain_df, option_type,
             delta_override=None, sentiment=None, atr=0.0):
        picks = []
        is_put = (option_type == 'put')
        params = self._pcs_params if is_put else self._ccs_params

        width = self._width_for_price(current_price, params.get('strike_width', 5))
        min_credit = params.get('min_net_credit', 0.20)
        min_prob = params.get('min_prob_profit', 0.70)
        min_oi = params.get('min_open_interest', 0)
        # Use sentiment-adjusted delta if provided, else fall back to config value
        max_delta = delta_override if delta_override is not None \
                    else params.get('max_delta_short_leg', 0.30)

        # Sort by strike to easily find pairs
        chain_df = chain_df.sort_values('strike', ascending=True)
        strikes = chain_df['strike'].tolist()

        # Create a lookup for price/iv
        data_map = {row['strike']: row for _, row in chain_df.iterrows()}

        for short_strike in strikes:
            # Filter Short Leg for OTM
            if is_put and short_strike >= current_price: continue # Short Put must be OTM (below price)
            if not is_put and short_strike <= current_price: continue # Short Call must be OTM (above price)

            # Sanity guard: reject strikes that are implausibly far from spot.
            # A CCS short call >200% of spot (or PCS short put <30% of spot)
            # signals either a data error (wrong current_price) or a zero-premium
            # far-OTM option that will never meet the credit threshold anyway.
            if is_put  and short_strike < current_price * 0.30: continue
            if not is_put and short_strike > current_price * 2.00: continue

            # Minimum % OTM distance from spot
            if is_put and self._min_otm_put > 0:
                if short_strike > current_price * (1.0 - self._min_otm_put):
                    continue
            if not is_put and self._min_otm_call > 0:
                if short_strike < current_price * (1.0 + self._min_otm_call):
                    continue

            # ATR-based distance guard
            if self._atr_enabled and atr > 0:
                if abs(current_price - short_strike) < self._atr_multiplier * atr:
                    continue

            short_opt = data_map[short_strike]
            iv = short_opt['impliedVolatility']
            prob_short_otm = self._prob_otm(short_opt, current_price, short_strike, days, option_type)

            # Check Delta / Prob requirement
            if (1.0 - prob_short_otm) > max_delta:
                continue
            if prob_short_otm < min_prob:
                continue

            # Find Long Leg (Exact width match)
            target_long_strike = short_strike - width if is_put else short_strike + width

            if target_long_strike in data_map:
                long_opt = data_map[target_long_strike]

                # Open-interest liquidity filter: skip if either leg has
                # insufficient OI (configurable; 0 = no filter).
                # OI == -1 means "unknown" (Alpaca OptionsSnapshot omits OI) —
                # treat as passing so Alpaca picks are never incorrectly filtered.
                if min_oi > 0:
                    short_oi = int(short_opt.get('openInterest') or 0)
                    long_oi  = int(long_opt.get('openInterest')  or 0)
                    if short_oi != -1 and long_oi != -1:
                        if short_oi < min_oi or long_oi < min_oi:
                            continue

                # Prices: Sell Short (Bid), Buy Long (Ask)
                short_bid = short_opt.get('bid', 0) or short_opt.get('lastPrice', 0)
                long_ask = long_opt.get('ask', 0) or long_opt.get('lastPrice', 0)

                net_credit = short_bid - long_ask

                if net_credit >= min_credit:
                    max_loss = width - net_credit
                    if max_loss <= 0: max_loss = 0.01

                    roi = net_credit / max_loss
                    annualized_roi = roi * (365 / max(1, days))  # annualized for DTE comparison
                    estimated_delta = 1.0 - prob_short_otm

                    # Yield-normalised score: (credit/width) × prob²
                    # Dividing by width removes bias toward near-the-money spreads
                    # that collect more absolute dollars at the same strike interval.
                    score = self._score(net_credit, prob_short_otm, width)

                    _oi, _vol = self._row_oi_vol(short_opt)
                    pick = {
                        'strategy': 'PCS' if is_put else 'CCS',
                        'symbol': symbol,
                        'expiry': expiry,
                        'current_price': round(current_price, 2),
                        'short_strike': short_strike,
                        'long_strike': target_long_strike,
                        'width': width,
                        'premium': round(net_credit, 2),
                        'max_loss': round(max_loss, 2),
                        'short_iv': round(float(short_opt.get('impliedVolatility') or 0), 4),
                        'long_iv': round(float(long_opt.get('impliedVolatility') or 0), 4),
                        'prob_win': round(prob_short_otm, 4),
                        'roi': round(roi, 4),
                        'annualized_roi': round(annualized_roi, 4),
                        'estimated_delta': round(estimated_delta, 4),
                        'score': round(score, 4),
                        'short_oi': _oi,
                        'short_volume': _vol,
                    }
                    # Attach sentiment metadata if available
                    if sentiment:
                        pick['sentiment']         = sentiment.get('sentiment', 'NEUTRAL')
                        pick['sentiment_strength'] = sentiment.get('strength', 0.0)
                        pick['delta_used']         = round(max_delta, 4)
                    picks.append(pick)

        return picks
