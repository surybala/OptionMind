import logging
from typing import Optional

_log = logging.getLogger('optionwheel')


def compute_mark_from_maps(strat: str, legs: dict, pos: dict,
                           put_map: dict, call_map: dict,
                           conservative: bool = False) -> Optional[float]:
    """Compute net mark-to-market cost-to-close from pre-fetched chain maps."""

    def mid(row) -> float:
        b  = float(row.get('bid',       0) or 0)
        a  = float(row.get('ask',       0) or 0)
        lp = float(row.get('lastPrice', 0) or 0)
        return (b + a) / 2.0 if b > 0 and a > 0 else lp

    def mark(strike, opt_type: str, buying: bool = True) -> Optional[float]:
        """
        buying=True  → leg is bought to close (short leg) — ask when conservative.
        buying=False → leg is sold  to close (long  leg) — bid when conservative.
        """
        m   = put_map if opt_type == 'put' else call_map
        row = m.get(float(strike))
        if row is None:
            return None
        if not conservative:
            return mid(row)
        b  = float(row.get('bid',       0) or 0)
        a  = float(row.get('ask',       0) or 0)
        lp = float(row.get('lastPrice', 0) or 0)
        if buying:
            return a if a > 0 else (lp if lp > 0 else b)
        else:
            return b if b > 0 else (lp if lp > 0 else a)

    try:
        if strat == 'PCS':
            ss = legs.get('short_strike') or legs.get('short_put')  or pos.get('strike')
            ls = legs.get('long_strike')  or legs.get('long_put')
            if ss is None or ls is None:
                return None
            sm = mark(ss, 'put', buying=True)   # buy back short put
            lm = mark(ls, 'put', buying=False)  # sell long put
            return sm - lm if sm is not None and lm is not None else None

        if strat == 'CCS':
            ss = legs.get('short_strike') or legs.get('short_call') or pos.get('strike')
            ls = legs.get('long_strike')  or legs.get('long_call')
            if ss is None or ls is None:
                return None
            sm = mark(ss, 'call', buying=True)   # buy back short call
            lm = mark(ls, 'call', buying=False)  # sell long call
            return sm - lm if sm is not None and lm is not None else None

        if strat in ('IC', 'IFLY'):
            sp = legs.get('short_put');  lp = legs.get('long_put')
            sc = legs.get('short_call'); lc = legs.get('long_call')
            if any(x is None for x in (sp, lp, sc, lc)):
                return None
            sm_p = mark(sp, 'put',  buying=True)   # buy back short put
            lm_p = mark(lp, 'put',  buying=False)  # sell long put
            sm_c = mark(sc, 'call', buying=True)   # buy back short call
            lm_c = mark(lc, 'call', buying=False)  # sell long call
            if any(x is None for x in (sm_p, lm_p, sm_c, lm_c)):
                return None
            return (sm_p - lm_p) + (sm_c - lm_c)

        if strat == 'CSP':
            ss = legs.get('short_strike') or pos.get('strike')
            return mark(ss, 'put', buying=True) if ss is not None else None

        if strat == 'CC':
            ss = legs.get('short_strike') or pos.get('strike')
            return mark(ss, 'call', buying=True) if ss is not None else None

        if strat == 'STRANGLE':
            sp = legs.get('short_put')  or pos.get('strike')
            sc = legs.get('short_call')
            if sp is None or sc is None:
                return None
            sm_p = mark(sp, 'put',  buying=True)  # buy back short put
            sm_c = mark(sc, 'call', buying=True)  # buy back short call
            if sm_p is None or sm_c is None:
                return None
            return sm_p + sm_c

    except Exception:
        pass

    return None
