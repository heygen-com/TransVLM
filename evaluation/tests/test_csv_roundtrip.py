"""Pin the CSV shape: column order, and None becoming an empty cell.

The committed reference values are read back through `read_metrics_csv`, so a change in either
convention would silently break every parity comparison.
"""

from __future__ import annotations

from transvlm_eval.aggregate import (
    AGGREGATE_METRIC_NAMES,
    read_metrics_csv,
    write_metrics_csv,
)


def test_column_order_is_the_papers_order():
    assert AGGREGATE_METRIC_NAMES == (
        'cl_tp',
        'cl_fp',
        'cl_fn',
        'cl_precision',
        'cl_recall',
        'cl_f1',
        'tl_tp',
        'tl_fp',
        'tl_fn',
        'tl_precision',
        'tl_recall',
        'tl_f1',
        'abe_sum',
        'abe_count',
        'abe_mabe',
        'fps',
    )


def test_round_trip_preserves_values_and_none(tmp_path):
    row = {name: float(i) for i, name in enumerate(AGGREGATE_METRIC_NAMES)}
    row['fps'] = None
    path = tmp_path / 'x.csv'
    write_metrics_csv(str(path), {'m': row})

    header = path.read_text().splitlines()[0]
    assert header == 'method_name,' + ','.join(AGGREGATE_METRIC_NAMES)
    assert path.read_text().splitlines()[1].endswith(','), 'None must be written as an empty cell'

    back = read_metrics_csv(str(path))['m']
    assert back['fps'] is None
    assert back['cl_tp'] == row['cl_tp']


def test_rows_are_written_sorted_by_method_name(tmp_path):
    blank = {name: 0.0 for name in AGGREGATE_METRIC_NAMES}
    path = tmp_path / 'y.csv'
    write_metrics_csv(str(path), {'zeta': dict(blank), 'alpha': dict(blank)})
    names = [line.split(',')[0] for line in path.read_text().splitlines()[1:] if line.strip()]
    assert names == ['alpha', 'zeta']
