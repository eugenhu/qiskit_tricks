import itertools
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
from qiskit.providers import JobV1 as Job
from qiskit.result import Result
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    QobjExperimentHeader,
)

from qiskit_tricks.result import (
    ensure_result,
    extract_metadata,
    find_metadata_keys,
    resultdf,
    tabulate_classified_result,
    tabulate_kerneled_result,
    tabulate_many_results,
    tabulate_result,
)


def test_tabulate_classified_result():
    counts = {hex(0b00_001): 10
             ,hex(0b01_001): 40
             ,hex(0b11_101): 30
             ,hex(0b01_000): 20}
    shots = sum(counts.values())

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.CLASSIFIED,
        data=ExperimentResultData(counts=counts),
        header=QobjExperimentHeader(
            name='my-circuit',
            metadata={'thing': 123, 'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
        ),
    )

    df = tabulate_classified_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=False)

    index = pd.MultiIndex.from_product(
        [[123], [10], (int(x, 16) for x in counts)],
        names=('thing', 'fizz', 'state')
    )
    expect = pd.DataFrame(
        {'count': counts.values()
        ,'c0': [0b001, 0b001, 0b101, 0b000]
        ,'c1': [0b00, 0b01, 0b11, 0b01]},
        index,
    )

    assert_frame_equal(df, expect)


def test_tabulate_classified_result_drop_extraneous():
    counts = {hex(0b00001): 100}
    shots = sum(counts.values())

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.CLASSIFIED,
        data=ExperimentResultData(counts=counts),
        header=QobjExperimentHeader(
            name='my-circuit',
            metadata={'thing': 123, 'fizz': 10},
            creg_sizes=[['c0', 5]],
            clbit_labels=[
                *[['c0', i] for i in range(5)],
            ],
        ),
    )

    df = tabulate_classified_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=True)

    index = pd.MultiIndex.from_product(
        [[123], [10], (int(x, 16) for x in counts)],
        names=('thing', 'fizz', 'state'),
    )
    expect = pd.DataFrame(
        {'count': counts.values()},
        index,
    )

    assert_frame_equal(df, expect)


def test_tabulate_kerneled_result_avg():
    shots = 1024
    memory = np.random.rand(5, 2)

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'thing': 123, 'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
            memory_slots=5,
        ),
    )

    series = tabulate_kerneled_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=False)

    index = pd.MultiIndex.from_tuples(
        [(123, 10, 'c0', 0)
        ,(123, 10, 'c0', 1)
        ,(123, 10, 'c0', 2)
        ,(123, 10, 'c1', 0)
        ,(123, 10, 'c1', 1)],
        names=['thing', 'fizz', 'creg', 'bit'],
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    assert_series_equal(series, expect)


def test_tabulate_kerneled_result_single():
    shots = 128
    memory = np.random.rand(shots, 5, 2)

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.SINGLE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'thing': 123, 'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
            memory_slots=5,
        ),
    )

    series = tabulate_kerneled_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=False)

    index = [(123, 10, 'c0', 0)
            ,(123, 10, 'c0', 1)
            ,(123, 10, 'c0', 2)
            ,(123, 10, 'c1', 0)
            ,(123, 10, 'c1', 1)]
    index = pd.MultiIndex.from_tuples(
        list((n, *tail) for n, tail in itertools.product(range(shots), index)),
        names=['shot', 'thing', 'fizz', 'creg', 'bit'],
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    assert_series_equal(series, expect)


def test_tabulate_kerneled_result_drop_extraneous_bit():
    shots = 128
    memory = np.random.rand(shots, 2, 2)

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.SINGLE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'thing': 123, 'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 1], ['c1', 1]],
            clbit_labels=[['c0', 1], ['c1', 1]],
            memory_slots=2,
        ),
    )

    series = tabulate_kerneled_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=True)

    index = [(123, 10, 'c0'), (123, 10, 'c1')]
    index = pd.MultiIndex.from_tuples(
        list((n, *tail) for n, tail in itertools.product(range(shots), index)),
        names=['shot', 'thing', 'fizz', 'creg'],
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    assert_series_equal(series, expect)


def test_tabulate_kerneled_result_drop_extraneous_creg():
    shots = 1024
    memory = np.random.rand(5, 2)

    result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'thing': 123, 'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 5]],
            clbit_labels=[
                *[['c0', i] for i in range(5)],
            ],
            memory_slots=5,
        ),
    )

    series = tabulate_kerneled_result(result, metadata_keys=('thing', 'fizz'), drop_extraneous=True)

    index = pd.MultiIndex.from_tuples(
        [(123, 10, 0)
        ,(123, 10, 1)
        ,(123, 10, 2)
        ,(123, 10, 3)
        ,(123, 10, 4)],
        names=['thing', 'fizz', 'bit'],
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    assert_series_equal(series, expect)


def test_tabulate_result():
    import qiskit_tricks.result

    mock1 = Mock()
    mock2 = Mock()
    mock_result = Mock()

    with patch.multiple(
            qiskit_tricks.result,
            tabulate_classified_result=mock1,
            tabulate_kerneled_result=mock2,
    ):
        expect = mock1.return_value
        got = tabulate_result((MeasLevel.CLASSIFIED, MeasReturnType.AVERAGE), mock_result, a=1, b=10)
        mock1.assert_called_once_with(mock_result, a=1, b=10)
        assert got == expect

        expect = mock2.return_value
        got = tabulate_result((MeasLevel.KERNELED, MeasReturnType.SINGLE), mock_result, a=2, b=20)
        mock2.assert_called_once_with(mock_result, a=2, b=20)
        assert got == expect

        mock2.reset_mock()

        expect = mock2.return_value
        got = tabulate_result((MeasLevel.KERNELED, MeasReturnType.AVERAGE), mock_result, a=3, b=30)
        mock2.assert_called_once_with(mock_result, a=3, b=30)
        assert got == expect


def test_tabulate_many_results():
    import qiskit_tricks.result

    mock_tabulate = Mock(return_value=pd.DataFrame({'x': [1], 'y': [2]}))
    mock_meas_type = (Mock(), Mock())
    mock_results = [Mock() for i in range(10)]

    for i, m in enumerate(mock_results):
        m.header.name = f"circuit-{i}"
        m.meas_level = mock_meas_type[0]
        m.meas_return = mock_meas_type[1]

    # Monkey patch tabulate_result()
    with patch.multiple(qiskit_tricks.result, tabulate_result=mock_tabulate):
        got = tabulate_many_results(mock_meas_type, mock_results, a=1, b=2)

    expect = pd.concat(
        [mock_tabulate.return_value]*len(mock_results),
        keys=(m.header.name for m in mock_results),
        names=['circuit']
    )

    assert_frame_equal(got, expect)

    from unittest.mock import call

    assert mock_tabulate.call_args_list == [
        call(mock_meas_type, m, a=1, b=2)
        for m in mock_results
    ]


def test_extract_metadata():
    metadata = {
        'thing': 123,
        'fizz': [10, [20]],
        'buzz': 1,
    }

    result = ExperimentResult(
        success=True,
        shots=1024,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=np.random.rand(5, 2)),
        header=QobjExperimentHeader(
            name='circuit',
            metadata=metadata,
            creg_sizes=[['c', 5]],
            clbit_labels=[['c', i] for i in range(5)],
            memory_slots=5,
        ),
    )

    got = extract_metadata(result, keys=['thing', 'fizz'])
    expect = (123, (10, (20,)))

    assert got == expect


def test_extract_metadata_when_metadata_is_None():
    result = ExperimentResult(
        success=True,
        shots=1024,
        data=ExperimentResultData(),
        header=QobjExperimentHeader(
            metadata=None,
        ),
    )
    assert extract_metadata(result, keys=['thing', 'fizz']) == (None, None)


def test_find_metadata_keys():
    metadata = {
        'thing': 123,
        'fizz': [10, [20]],
        'buzz': 1,
        'invalid': object(),
    }

    result = ExperimentResult(
        success=True,
        shots=1024,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=np.random.rand(5, 2)),
        header=QobjExperimentHeader(
            name='circuit',
            metadata=metadata,
            creg_sizes=[['c', 5]],
            clbit_labels=[['c', i] for i in range(5)],
            memory_slots=5,
        ),
    )

    got = find_metadata_keys(result)
    expect = ['thing', 'fizz', 'buzz']

    assert set(got) == set(expect)


def test_find_metadata_keys_with_subcircuits():
    subcircuits = [
        {'name': 'circuit-0'
        ,'metadata': {'thing': 123}
        ,'creg_map': {'c0': 'c'}},
        {'name': 'circuit-1'
        ,'metadata': {'thing': 456}
        ,'creg_map': {'c1': 'c'}},
    ]

    result = ExperimentResult(
        success=True,
        shots=1024,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=np.random.rand(5, 2)),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'subcircuits': subcircuits, 'some': 88},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
            memory_slots=5,
        ),
    )

    got = find_metadata_keys(result, subcircuits=False)
    expect = ['some']
    assert set(got) == set(expect)

    got = find_metadata_keys(result, subcircuits=True)
    expect = ['some', 'thing']
    assert set(got) == set(expect)


def test_find_metadata_keys_when_metadata_is_None():
    result = ExperimentResult(
        success=True,
        shots=1024,
        data=ExperimentResultData(),
        header=QobjExperimentHeader(
            metadata=None,
        ),
    )
    assert find_metadata_keys(result) == []


def test_ensure_result_with_job():
    job = MagicMock(spec=Job)
    assert ensure_result(job) == job.result()
    assert ensure_result([job] * 3) == [job.result()] * 3
    assert ensure_result(*([job] * 3)) == [job.result()] * 3


def test_ensure_result_with_result():
    result = MagicMock(spec=Result)
    assert ensure_result(result) == result
    assert ensure_result([result] * 3) == [result] * 3
    assert ensure_result(*([result] * 3)) == [result] * 3


# resultdf() is kind of difficult to unit test, needs refactoring.

def test_resultdf():
    shots = 1024
    memory = np.random.rand(5, 2)

    exp_result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'fizz': 10, 'buzz': 20},
            creg_sizes=[['c0', 5]],
            clbit_labels=[
                *[['c0', i] for i in range(5)],
            ],
            memory_slots=5,
        ),
    )

    result_a = Result(
        backend_name='lorem_ipsum',
        backend_version='0.0.0',
        qobj_id='a',
        job_id='a',
        success=True,
        results=[exp_result]
    )

    result_b = Result(
        backend_name='lorem_ipsum',
        backend_version='0.0.0',
        qobj_id='a',
        job_id='b',
        success=True,
        results=[exp_result]
    )

    index_names = ('job_id', 'fizz', 'buzz', 'bit')
    index = pd.MultiIndex.from_tuples(
        [(x, 10, 20, i) for x, i in itertools.product(['a', 'b'], range(5))],
        names=index_names,
    )
    expect = pd.Series(2*list(memory.view(np.cdouble).ravel()), index)

    got = resultdf(result_a, result_b)
    # Ordering of names in index is not guaranteed when 'metadata' kw not provided.
    got = got.reorder_levels([index_names.index(x) for x in got.index.names])
    assert_series_equal(got, expect)

    got = resultdf([result_a, result_b])
    got = got.reorder_levels([index_names.index(x) for x in got.index.names])
    assert_series_equal(got, expect)


def test_resultdf_metadata_ordering():
    shots = 1024
    memory = np.random.rand(5, 2)

    exp_result = ExperimentResult(
        success=True,
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'foo': 10, 'bar': 20, 'baz': 30},
            creg_sizes=[['c0', 5]],
            clbit_labels=[
                *[['c0', i] for i in range(5)],
            ],
            memory_slots=5,
        ),
    )

    result = Result(
        backend_name='lorem_ipsum',
        backend_version='0.0.0',
        qobj_id='a',
        job_id='a',
        success=True,
        results=[exp_result]
    )

    metadata = ('baz', 'foo', 'bar')
    index = pd.MultiIndex.from_tuples(
        [('a', 30, 10, 20, i) for i in range(5)],
        names=('job_id', *metadata, 'bit'),
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    got = resultdf(result, metadata=metadata)
    assert_series_equal(got, expect)

    metadata = ('bar', 'baz', 'foo')
    index = pd.MultiIndex.from_tuples(
        [('a', 20, 30, 10, i) for i in range(5)],
        names=('job_id', *metadata, 'bit'),
    )
    expect = pd.Series(memory.view(np.cdouble).ravel(), index)

    got = resultdf(result, metadata=metadata)
    assert_series_equal(got, expect)
