from __future__ import annotations


def normalize_order_status(status) -> str:
    """
    Return a broker order status as a lowercase Alpaca-style token.

    alpaca-py may expose statuses as enum instances whose string form looks
    like ``OrderStatus.FILLED``.  The trading code expects plain values like
    ``filled``.
    """
    if status is None:
        return ""

    value = getattr(status, "value", None)
    if value is not None:
        status = value

    text = str(status).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text
