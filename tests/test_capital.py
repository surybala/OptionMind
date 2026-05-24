from src.capital import capital_by_strategy, capital_for_position


def test_spread_capital_scales_by_contracts():
    pos = {
        'type': 'PCS',
        'status': 'EXECUTED',
        'strike': 150.0,
        'legs': {'short_strike': 150.0, 'long_strike': 145.0},
        'contracts': 3,
    }

    assert capital_for_position(pos) == 1500.0


def test_pending_close_counts_as_deployed_capital():
    pos = {
        'type': 'CSP',
        'status': 'PENDING_CLOSE',
        'strike': 50.0,
        'legs': {'short_strike': 50.0},
        'contracts': 2,
    }

    assert capital_for_position(pos) == 10000.0


def test_dry_run_does_not_count_as_deployed_capital():
    pos = {
        'type': 'PCS',
        'status': 'DRY_RUN',
        'strike': 150.0,
        'legs': {'short_strike': 150.0, 'long_strike': 145.0},
        'contracts': 10,
    }

    assert capital_for_position(pos) == 0.0


def test_capital_by_strategy_groups_totals():
    rows = capital_by_strategy([
        {
            'type': 'PCS',
            'status': 'EXECUTED',
            'strike': 150.0,
            'legs': {'short_strike': 150.0, 'long_strike': 145.0},
            'contracts': 2,
        },
        {
            'type': 'CSP',
            'status': 'EXECUTED',
            'strike': 50.0,
            'legs': {'short_strike': 50.0},
            'contracts': 1,
        },
    ])

    by_strategy = {row['strategy']: row for row in rows}
    assert by_strategy['PCS']['capital_deployed'] == 1000.0
    assert by_strategy['CSP']['capital_deployed'] == 5000.0
