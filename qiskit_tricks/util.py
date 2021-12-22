from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Optional, SupportsInt, TypeVar, Union, cast, overload

from qiskit.circuit import ClassicalRegister
from qiskit.pulse import Call, Play, Schedule, ScheduleBlock


T = TypeVar('T')


def marginal_counts(
        counts: Mapping[Union[int, str], int],
        subsystem: Iterable[Collection[int]],
) -> list[dict[int, int]]:
    subsystem_counts: list[dict[int, int]] = []
    subsystem_from_bit_index: dict[int, list[int]] = defaultdict(list)

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

    The 0-th bit of `source` is the least significant bit (LSB), and `indices[0]` will be the LSB of the
    returned integer.
    """
    dest = 0*source  # easiest way to accomodate any 'source' type
    for i in reversed(indices):
        bit = (source>>i)&1
        dest = (dest<<1) + bit
    return dest


def get_creg_indices(clbit_labels: Sequence[tuple[str, int]]) -> dict[str, list[int]]:
    """Return a dictionary mapping register names to a list of indices of their corresponding bits in
    `clbit_labels`."""
    # This works for 'qubit_labels' as well, but we only need this function for cregs.
    creg_sizes = cast(Mapping[str, int], Counter(next(zip(*clbit_labels))))
    cregs = {name: ClassicalRegister(int(size), name) for name, size in creg_sizes.items()}
    clbits = [cregs[name][index] for name, index in clbit_labels]
    creg_indices = {name: [*map(clbits.index, creg)] for name, creg in cregs.items()}
    return creg_indices


def get_play_instruction(sched: Union[Schedule, ScheduleBlock]) -> Optional[Play]:
    """Return the first Play instruction in `sched`."""
    for t, inst in sched.instructions:
        if isinstance(inst, Play):
            return inst
        elif isinstance(inst, Call):
            x = get_play_instruction(inst.subroutine)
            if x: return x
        else:
            continue

    return None


def install_parametric_pulse(name, pulse):
    """Monkey patch install a new parametric pulse."""
    # XXX: This is very hacky.
    from enum import Enum
    import importlib
    from qiskit.assembler.assemble_schedules import ParametricPulseShapes
    
    importlib.import_module('qiskit.assembler.assemble_schedules') \
        .ParametricPulseShapes = Enum("ParametricPulseShapes", {
            **{t.name: t.value for t in ParametricPulseShapes},
            name: pulse,
        })
