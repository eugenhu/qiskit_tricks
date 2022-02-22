from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import (
    Dict,
    List,
    Optional,
    SupportsInt,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from qiskit.circuit import ClassicalRegister
from qiskit.pulse import Call, ParametricPulse, Play, Schedule, ScheduleBlock


T = TypeVar('T')


def marginal_counts(
        counts: Mapping[Union[int, str], int],
        subsystem: Iterable[Collection[int]],
) -> List[Dict[int, int]]:
    """Maginalize counts.

    Counts is a dictionary like so::

        counts = {'0000': 10,
                  '0001':  7,
                  '0010':  3,
                  '0100':  5,
                  '1000':  2}

    The keys can be integers, or binary/hexadecminal string representations.

    Subsystem is a list of a collection of indices belonging to a subsystem, e.g.::

        subsystem = [{2, 3}, {0}]

    The marginalized counts are then::

        marginal_counts(counts, subsystem) \
            == [{0: 20,
                 1:  5,
                 2:  2},
                {0: 20,
                 1:  7}]

    Note that subsystem indices orders do not matter, i.e.::

        subsystem = [[2, 3], [0]]

    and::

        subsystem = [[3, 2], [0]]

    will produce the same return value. The bits of a state are not re-ordered when marginalizing.

    Args:
        counts: A dictionary mapping ints or strs to an integer count.
        subsystems: A list of a collection of subsystem indices.

    Returns:
        A list of dictionary counts corresponding to each subsystem.
    """
    subsystem_counts: List[Dict[int, int]] = []
    subsystem_from_bit_index: Dict[int, List[int]] = defaultdict(list)

    for i, bit_indices in enumerate(subsystem):
        subsystem_counts.append(defaultdict(int))
        for j in bit_indices:
            subsystem_from_bit_index[j].append(i)

    for state, count in counts.items():
        state = ensure_int_state(state)
        buffer = [0] * len(subsystem_counts)

        for i in reversed(range(state.bit_length())):
            bit = (state>>i)&1
            for j in subsystem_from_bit_index[i]:
                buffer[j] = (buffer[j]<<1) + bit

        for subcounts, substate in zip(subsystem_counts, buffer):
            subcounts[substate] += count

    return list(map(dict, subsystem_counts))


def ensure_int_state(state: Union[int, str, SupportsInt]) -> int:
    """Convert state to a Python int.

    If state is already a Python int, return it as is.

    If state is a str, first remove all spaces. If the string starts with '0x', interpret it as hexadecimal.
    If there is no prefix, interpret it as binary.

    If state is any other type (e.g. NumPy type), it is converted to an integer via int(state).

    Args:
        state: An object representing a bitstring.

    Returns:
        A Python int.
    """
    if isinstance(state, int):
        return state
    elif isinstance(state, str):
        state = state.replace(' ', '')
        # Hex and binary are two common formats used by Qiskit.
        if state.startswith('0x'):
            return int(state, base=16)
        else:
            return int(state, base=2)
    else: # A NumPy integer type perhaps.
        return int(state)


@overload
def bit_extract(source: int, indices: Sequence[int]) -> int:
    ...
@overload
def bit_extract(source: T, indices: Sequence[int]) -> T:
    # Accommodate any source type 'T', e.g. numpy arrays or pandas data structures.
    ...
def bit_extract(source, indices):
    """Return an integer formed by extracting the bits of `source` at `indices`.

    The 0-th bit of `source` is the least significant bit (LSB), and bit extracted at `indices[0]` will be
    stored in the LSB of the returned integer.
    """
    dest = 0*source  # easiest way to accomodate any 'source' type
    for i in reversed(indices):
        bit = (source>>i)&1
        dest = (dest<<1) + bit
    return dest


def get_creg_indices(clbit_labels: Sequence[Tuple[str, int]]) -> Dict[str, List[int]]:
    """Return a dictionary mapping register names to a list of indices of their corresponding bits in
    `clbit_labels`."""
    # This works for 'qubit_labels' as well, but we only need this function for cregs.
    creg_sizes = cast(Mapping[str, int], Counter(next(zip(*clbit_labels))))
    cregs = {name: ClassicalRegister(int(size), name) for name, size in creg_sizes.items()}
    clbits = [cregs[name][index] for name, index in clbit_labels]
    creg_indices = {name: [*map(clbits.index, creg)] for name, creg in cregs.items()}
    return creg_indices


def get_play_instruction(schedule: Union[Schedule, ScheduleBlock]) -> Optional[Play]:
    """Return the first Play instruction in schedule."""
    for t, inst in schedule.instructions:
        if isinstance(inst, Play):
            return inst
        elif isinstance(inst, Call):
            x = get_play_instruction(inst.subroutine)
            if x: return x
        else:
            continue

    return None


def install_parametric_pulse(name: str, pulse: Type[ParametricPulse]) -> None:
    """Monkey patch install a new parametric pulse.

    This is very hacky. It modifies the internal workings of the qiskit qobj assembler.

    Args:
        name: Name of pulse to register with the qiskit assembler.
        pulse: The parametric pulse class to add.
    """
    # XXX: This is very hacky.
    from enum import Enum
    import importlib
    from qiskit.assembler.assemble_schedules import ParametricPulseShapes

    importlib.import_module('qiskit.assembler.assemble_schedules') \
        .ParametricPulseShapes = Enum("ParametricPulseShapes", {
            **{t.name: t.value for t in ParametricPulseShapes},
            name: pulse,
        })
