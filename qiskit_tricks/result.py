from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
import itertools
from typing import Optional, Union, overload
import warnings

import numpy as np
import pandas as pd
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
from qiskit.providers import JobV1 as Job
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from .transform import has_subcircuits, uncombine_result
from .util import bit_extract, get_creg_indices


__all__ = [
    'resultdf',
    'ensure_result',
]


ResultLike = Union[Job, Result]


def resultdf(
        result: Union[ResultLike, Iterable[ResultLike]],
        *results: ResultLike,  # type: ignore
        subcircuits: Optional[bool] = None,
        metadata=True,
        drop_extraneous=True,
        **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    if not isinstance(result, Iterable):
        result = [result]

    results: list[Result] = ensure_result(result, *results)
    del result

    results, dup_results = filter_unique_results(results)
    if dup_results:
        warnings.warn("resultdf(): got duplicate job ids, ignoring.", stacklevel=2)

    experiment_results = list(itertools.chain.from_iterable(r.results for r in results))

    if subcircuits is None:
        subcircuits = any(map(has_subcircuits, experiment_results))

    if metadata is True:
        metadata_keys = tuple(set(itertools.chain.from_iterable(
            find_metadata_keys(r, subcircuits=subcircuits)
            for r in experiment_results
        )))
    elif bool(metadata) is False:
        metadata_keys = ()
    else:
        # Use dict.fromkeys() instead of set() to retain order. Dictionaries in Python 3.7 are ordered.
        metadata_keys = tuple(dict.fromkeys(metadata).keys())

    del experiment_results

    kwargs.update({
        'metadata_keys': metadata_keys,
        'drop_extraneous': drop_extraneous,
    })

    meas_type = check_meas_type(results[0].results[0])
    job_ids: list[str] = []
    tables = []
    for result in results:
        job_ids.append(result.job_id)
        if not subcircuits:
            table = tabulate_many_results(meas_type, result.results, **kwargs)
            tables.append(table)
        else:
            circuit_names: list[str] = []
            inner_tables = []
            for i, exp_result in enumerate(result.results):
                assert exp_result.success

                circuit_name = getattr(exp_result.header, 'name', f'unnamed-{i}')
                if circuit_name in circuit_names:
                    warnings.warn(f"Duplicate circuit name '{circuit_name}'", stacklevel=2)
                circuit_names.append(circuit_name)

                table = tabulate_many_results(
                    meas_type,
                    uncombine_result(exp_result),
                    **kwargs,
                )
                table.rename_axis(index={'circuit': 'subcircuit'}, inplace=True)
                inner_tables.append(table)
            table = pd.concat(inner_tables, keys=circuit_names, names=['circuit'])
            tables.append(table)

    table = pd.concat(tables, keys=job_ids, names=['job_id'])

    if drop_extraneous:
        table.index = drop_extraneous_levels(table.index, ('circuit', 'subcircuit'))

    # Include original results as an extra attribute.
    table.attrs['results'] = results

    return table


@overload
def ensure_result(obj: ResultLike) -> Result:
    ...
@overload
def ensure_result(obj: Iterable[ResultLike]) -> list[Result]:
    ...
@overload
def ensure_result(*objs: ResultLike) -> list[Result]:
    ...
def ensure_result(obj, *objs):
    if isinstance(obj, Iterable):
        obj = [*obj, *objs]
    else:
        if objs:
            obj = [obj, *objs]
        else:
            return ensure_result([obj])[0]

    results: list[Result] = []
    for o in obj:
        if isinstance(o, Job):
            results.append(o.result())  # type: ignore
        elif isinstance(o, Result):
            results.append(o)
        else:
            raise TypeError(
                "'Job' or 'Result' expected"
                f" (got '{type(o).__qualname__}')"
            )

    return results


def filter_unique_results(results: Iterable[Result]) -> tuple[list[Result], list[Result]]:
    job_ids: set[str] = set()
    unq_results: list[Result] = []
    dup_results: list[Result] = []

    for result in results:
        if result.job_id not in job_ids:
            job_ids.add(result.job_id)
            unq_results.append(result)
        else:
            dup_results.append(result)

    return unq_results, dup_results


def find_metadata_keys(result: ExperimentResult, subcircuits=False):
    keys: list[str] = []

    def all_metadata_items():
        metadata = getattr(result.header, 'metadata', {})
        yield from metadata.items()
        if subcircuits:
            for subcircuit_info in metadata.get('subcircuits', []):
                yield from subcircuit_info.get('metadata', {}).items()

    for k, v in all_metadata_items():
        # Ignore keys starting with an underscore.
        if k.startswith('_'): continue
        try:
            atomize(v)
        except TypeError:
            continue
        keys.append(k)

    return keys


def extract_metadata(result: ExperimentResult, keys: Collection[str]) -> tuple[Hashable, ...]:
    metadata_dict: dict[str, Hashable] = {k: None for k in keys}
    for k, v in getattr(result.header, 'metadata', {}).items():
        if k not in keys: continue
        metadata_dict[k] = atomize(v)
    return tuple(metadata_dict.values())


def atomize(o, _visited: Optional[set] = None) -> Hashable:
    _visited = _visited or set()

    if id(o) in _visited:
        raise ValueError("Self-referencing loop detected")

    if isinstance(o, (int, float, str, bool, type(None))):
        return o
    elif isinstance(o, (tuple, list, set)):
        _visited.add(id(o))
        return tuple(atomize(oi, _visited) for oi in o)
    else:
        raise TypeError(f"Cannot atomize unsupported type '{type(o).__qualname__}'")


def check_meas_type(
        result: ExperimentResult,
        meas_type: Optional[tuple[MeasLevel, MeasReturnType]] = None,
) -> tuple[MeasLevel, MeasReturnType]:
    from qiskit.compiler.assembler import MeasLevel, MeasReturnType

    def throw_err(reason):
        raise ValueError(f"Heterogeneous {reason} in results")

    if meas_type is None:
        return (
            MeasLevel(result.meas_level),
            MeasReturnType(getattr(result, 'meas_return', 'avg'))
        )

    if result.meas_level != meas_type[0]:
        throw_err('meas_level')
    if getattr(result, 'meas_return', meas_type[1]) != meas_type[1]:
        throw_err('meas_return')

    return meas_type


def tabulate_many_results(
        meas_type: tuple[MeasLevel, MeasReturnType],
        results: Sequence[ExperimentResult],
        **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    circuit_names: list[str] = []
    tables = []

    for i, result in enumerate(results):
        assert result.success
        check_meas_type(result, meas_type)

        circuit_name = getattr(result.header, 'name', f'unnamed-{i}')
        if circuit_name in circuit_names:
            warnings.warn(f"Duplicate circuit name '{circuit_name}'", stacklevel=2)
        circuit_names.append(circuit_name)

        table = tabulate_result(
            meas_type,
            result,
            **kwargs,
        )
        tables.append(table)

    table = pd.concat(tables, keys=circuit_names, names=['circuit'])

    return table


def tabulate_result(
        meas_type: tuple[MeasLevel, MeasReturnType],
        result: ExperimentResult,
        **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    tabulators: dict[tuple[MeasLevel, MeasReturnType], Callable] = {
        (MeasLevel.KERNELED, MeasReturnType.SINGLE): tabulate_kerneled_result,
        (MeasLevel.KERNELED, MeasReturnType.AVERAGE): tabulate_kerneled_result,
        (MeasLevel.CLASSIFIED, MeasReturnType.AVERAGE): tabulate_classified_result,
    }
    if meas_type in tabulators:
        return tabulators[meas_type](result, **kwargs)
    else:
        raise ValueError(f"Unsupported (meas_level, meas_return): ({meas_type[0]!s}, {meas_type[1]!s})")


def tabulate_kerneled_result(
        result: ExperimentResult,
        *,
        metadata_keys: Sequence[str] =(),
        drop_extraneous=False,
) -> pd.Series:
    header = result.header
    meas_return = result.meas_return
    memory = np.array(result.data.memory, dtype=np.double).view(np.cdouble)
    num_memory_slots = getattr(header, 'memory_slots', memory.shape[-2])

    if num_memory_slots != memory.shape[-2]:
        raise ValueError(f"'memory_slots' attribute does not match memory shape")
    if meas_return == 'single' and result.shots != memory.shape[0]:
        raise ValueError(f"'shots' attribute does not match memory shape")

    clbit_labels: list[tuple[str, int]] = getattr(header, 'clbit_labels', [])
    if not clbit_labels:
        # Create a fallback creg with name 'c'.
        clbit_labels = [('c', i) for i in range(num_memory_slots)]

    if len(clbit_labels) != num_memory_slots:
        raise ValueError("Number of bits in 'clbit_labels' does not equal 'memory_slots'")

    metadata_values = extract_metadata(result, metadata_keys)

    index = [(*metadata_values, *label) for label in clbit_labels]
    index_names = [*metadata_keys, 'creg', 'bit']

    if meas_return == 'single':
        index = [(n, *tail) for n, tail in itertools.product(range(result.shots), index)]
        index_names = ['shot', *index_names]

    index = pd.MultiIndex.from_tuples(index, names=index_names)
    if drop_extraneous:
        index = drop_extraneous_levels(index, ('bit', 'creg'))

    series = pd.Series(memory.ravel(), index)

    return series


def tabulate_classified_result(
        result: ExperimentResult,
        *,
        metadata_keys: Sequence[str] = (),
        drop_extraneous=False,
) -> pd.DataFrame:
    header = result.header
    counts = {int(k, base=16): v for k, v in result.data.counts.items()}

    num_memory_slots = getattr(header, 'memory_slots', None)
    if num_memory_slots is None:
        num_memory_slots = max(map(int.bit_length, counts.keys()))

    clbit_labels: list[tuple[str, int]] = getattr(header, 'clbit_labels', [])
    if not clbit_labels:
        # Create a fallback creg with name 'c'.
        clbit_labels = [('c', i) for i in range(num_memory_slots)]

    creg_indices = get_creg_indices(clbit_labels)

    if drop_extraneous and len(creg_indices) < 2:
        creg_indices = {}

    metadata_values = extract_metadata(result, metadata_keys)

    index = []
    data = []
    for state, count in counts.items():
        index.append((*metadata_values, state))
        data.append([count, *(bit_extract(state, ix) for ix in creg_indices.values())])

    index = pd.MultiIndex.from_tuples(
        index,
        names=[*metadata_keys, 'state'],
    )

    df = pd.DataFrame(data, index, ['count', *creg_indices.keys()])

    return df


def drop_extraneous_levels(index: pd.Index, drop_order: Sequence) -> pd.Index:
    for level in drop_order:
        if level not in index.names: continue
        new_index = index.droplevel(level)
        if new_index.is_unique:
            index = new_index
    return index
