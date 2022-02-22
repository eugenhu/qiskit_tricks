from collections.abc import Callable, Hashable, Iterable, Sequence
import itertools
from typing import Optional, Union, overload, Tuple, List, Dict
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
    """Convert a qiskit Result (or many Results) into a pandas Series or DataFrame.

    For convenience, Jobs can also be passed. They will be converted to results via Job.result(). If multiple
    Results are passed, the returned pandas Series or DataFrame is a concatenation of each result. See the
    project README for how-to/examples.

    Args:
        result: A Result or Job, or an Iterable of Result or Jobs.
        results: More Results or Jobs.
        subcircuits: True if subcircuits should be expanded (will raise an error if no subcircuits), False
            otherwise. None if subcircuits should be expanded only if they exist.
        metadata: True if metadata should be extracted, False otherwise. If True, metadata keys will be
            automatically determined, pass a sequence of keys to manually specify instead.
        drop_extraneous: Drop 'useless' levels from the resulting series or frame for convenience.

    Returns:
        The Result or Jobs converted to a pandas Series or DataFrame.
    """
    if not isinstance(result, Iterable):
        result = [result]

    results: List[Result] = ensure_result(result, *results)
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

    job_ids: List[str] = []
    tables = []
    for result in results:
        job_ids.append(result.job_id)
        tables.append(_resultdf(result, subcircuits, **kwargs))

    table = pd.concat(tables, keys=job_ids, names=['job_id'])

    if drop_extraneous:
        table.index = drop_extraneous_levels(table.index, ('circuit', 'subcircuit'))

    # Include original results as an extra attribute.
    table.attrs['results'] = results

    return table


def _resultdf(result: Result, subcircuits: bool, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    meas_type = check_meas_type(result.results[0])

    if not subcircuits:
        table = tabulate_many_results(meas_type, result.results, **kwargs)
    else:
        circuit_names: List[str] = []
        subtables = []
        for i, exp_result in enumerate(result.results):
            assert exp_result.success

            name = getattr(exp_result.header, 'name', f'unnamed-{i}')
            if name in circuit_names:
                warnings.warn(f"Duplicate circuit name '{name}'", stacklevel=2)
            circuit_names.append(name)

            table = tabulate_many_results(
                meas_type,
                uncombine_result(exp_result),
                **kwargs,
            )
            table.rename_axis(index={'circuit': 'subcircuit'}, inplace=True)
            subtables.append(table)
        table = pd.concat(subtables, keys=circuit_names, names=['circuit'])

    return table


@overload
def ensure_result(obj: ResultLike) -> Result:
    ...
@overload
def ensure_result(obj: Iterable[ResultLike]) -> List[Result]:
    ...
@overload
def ensure_result(*objs: ResultLike) -> List[Result]:
    ...
def ensure_result(obj, *objs):
    """Convert Job or Results to Results.

    Can pass either a single Job or Result argument, multiple Job or Result arguments, or an iterable of Job
    or Results. Jobs will be waited on to retrieve their results.

    Returns:
        If a single argument is passed, a single Result is returned. If multiple arguments or an interable
        is passed, a list of Results is returned.
    """
    if isinstance(obj, Iterable):
        obj = [*obj, *objs]
    else:
        if objs:
            obj = [obj, *objs]
        else:
            return ensure_result([obj])[0]

    results: List[Result] = []
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


def filter_unique_results(results: Iterable[Result]) -> Tuple[List[Result], List[Result]]:
    job_ids: set[str] = set()
    unq_results: List[Result] = []
    dup_results: List[Result] = []

    for result in results:
        if result.job_id not in job_ids:
            job_ids.add(result.job_id)
            unq_results.append(result)
        else:
            dup_results.append(result)

    return unq_results, dup_results


def find_metadata_keys(result: ExperimentResult, subcircuits=False):
    """Find candidate metadata keys in result.

    Search in the 'metadata' header of result. If a key starts with '_', it is ignored. If the value cannot
    be atomized, it is ignored. Otherwise, the key is a valid and appended to a list of metadata keys to
    return.

    Args:
        result: The result to look for metadata.
        subcircuits: True if subcircuit metadata should be considered.

    Returns:
        Keys found.
    """
    keys: List[str] = []

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


def extract_metadata(result: ExperimentResult, keys: Sequence[str]) -> Tuple[Hashable, ...]:
    """Extract the metadata values from result corresponding to keys.

    Subcircuits are not searched.

    Args:
        result: The result to extract metadata from.
        keys: A sequence of metadata keys whose values are wanted.

    Returns:
        A sequence of metadata values corresponding to keys.
    """
    metadata_dict: Dict[str, Hashable] = {k: None for k in keys}
    for k, v in getattr(result.header, 'metadata', {}).items():
        if k not in keys: continue
        metadata_dict[k] = atomize(v)
    return tuple(metadata_dict.values())


def atomize(o: object, _visited: Optional[set] = None) -> Hashable:
    """Atomize an object (i.e. make it immutable).

    Instances of: int, float, str, bool, NoneType; are returned as is. Instances of: tuple, list, set; are
    converted to tuples.

    Args:
        o: The object to atomize.
        _visited: Internal parameter to detect cycles.

    Returns:
        The object converted to an immutable form.
    """
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
        meas_type: Optional[Tuple[MeasLevel, MeasReturnType]] = None,
) -> Tuple[MeasLevel, MeasReturnType]:
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
        meas_type: Tuple[MeasLevel, MeasReturnType],
        results: Sequence[ExperimentResult],
        **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    circuit_names: List[str] = []
    tables = []

    for i, result in enumerate(results):
        assert result.success
        check_meas_type(result, meas_type)

        name = getattr(result.header, 'name', f'unnamed-{i}')
        if name in circuit_names:
            warnings.warn(f"Duplicate circuit name '{name}'", stacklevel=2)
        circuit_names.append(name)

        table = tabulate_result(
            meas_type,
            result,
            **kwargs,
        )
        tables.append(table)

    table = pd.concat(tables, keys=circuit_names, names=['circuit'])

    return table


def tabulate_result(
        meas_type: Tuple[MeasLevel, MeasReturnType],
        result: ExperimentResult,
        **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    tabulators: Dict[Tuple[MeasLevel, MeasReturnType], Callable] = {
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
        metadata_keys: Sequence[str] = (),
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

    clbit_labels: List[Tuple[str, int]] = getattr(header, 'clbit_labels', [])
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

    clbit_labels: List[Tuple[str, int]] = getattr(header, 'clbit_labels', [])
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
    """Drop levels in order of `drop_order` from `index`.

    If a level is not present in `index`, or dropping it would make the index non-unique, it is ignored.

    Args:
        index: The index to drop levels from.
        drop_order: The levels to drop, in the order passed.

    Returns:
        The new index after dropping extraneous levels.
    """
    for level in drop_order:
        if level not in index.names: continue
        new_index = index.droplevel(level)
        if new_index.is_unique:
            index = new_index
    return index
