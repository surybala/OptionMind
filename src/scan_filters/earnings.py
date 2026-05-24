import datetime
from typing import Set


def should_skip_expiry(expiry_dt: datetime.datetime,
                       earnings_dates: Set,
                       buffer_days: int,
                       today: datetime.datetime) -> bool:
    """Return True if this expiry should be skipped due to earnings proximity."""
    if not earnings_dates:
        return False
    expiry_date = expiry_dt.date()
    buf = datetime.timedelta(days=buffer_days)
    return any(today.date() <= ed <= expiry_date + buf for ed in earnings_dates)
