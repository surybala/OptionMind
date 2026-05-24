import logging
import pandas as pd

_log = logging.getLogger('optionwheel')


def apply_liquidity_filter(df: 'pd.DataFrame', min_bid: float,
                           min_oi: int, max_spread_pct: float) -> 'pd.DataFrame':
    """Apply bid/OI/spread filters to an option chain DataFrame."""
    if df is None or df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    # 1. Minimum bid
    if min_bid > 0 and 'bid' in df.columns:
        mask &= df['bid'] >= min_bid

    # 2. Minimum open interest (-1 sentinel = unknown from Alpaca -> always pass)
    if min_oi > 0 and 'openInterest' in df.columns:
        oi = df['openInterest'].fillna(0)
        mask &= (oi < 0) | (oi >= min_oi)

    # 3. Maximum bid-ask spread percentage
    if max_spread_pct < 1.0 and 'bid' in df.columns and 'ask' in df.columns:
        mid = (df['bid'] + df['ask']) / 2.0
        # Only apply where mid > 0 to avoid division-by-zero on zero-bid rows
        # (those are already caught by the min_bid check above)
        valid_mid = mid > 0
        spread_pct = pd.Series(0.0, index=df.index)
        spread_pct[valid_mid] = (df['ask'] - df['bid'])[valid_mid] / mid[valid_mid]
        mask &= ~valid_mid | (spread_pct <= max_spread_pct)

    return df[mask].reset_index(drop=True)
