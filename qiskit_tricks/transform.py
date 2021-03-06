from collections.abc import Sequence
import copy
import itertools
from typing import Iterable, Literal, NamedTuple, overload, Optional, List, Tuple, Set

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.compiler.assembler import MeasLevel
import qiskit.pulse as qpulse
from qiskit.result.models import ExperimentResult
import retworkx

from .util import get_creg_indices, marginal_counts


__all__ = [
    'parallelize_circuits',
    'combine_circuits',
    'uncombine_result',
    'has_subcircuits',
    'bake_schedule',
]


@overload
def parallelize_circuits(
        circuits: Sequence[QuantumCircuit],
        coupling_map: Sequence[Sequence[int]] = (),
        *,
        return_index: Literal[False] = False,
) -> List[QuantumCircuit]:
    ...
@overload
def parallelize_circuits(
        circuits: Sequence[QuantumCircuit],
        coupling_map: Sequence[Sequence[int]] = (),
        *,
        return_index: Literal[True] = True,
) -> Tuple[List[QuantumCircuit], List[List[int]]]:
    ...
def parallelize_circuits(
        circuits,
        coupling_map=(),
        *,
        return_index=False,
):
    """Same as combine_circuits() but accepts a coupling_map parameter.

    The coupling_map parameter specifies coupled qubits. A circuit with active qubits that couple to another
    circuit's active qubits will not be combined into one circuit.

    If return_index=True is passed, a list of list of indices is also returned, representing the indices of
    the original circuits combined into each host circuit.

    Args:
        circuits: Circuits to combine.
        coupling_map: Qubits that may interfere.
        return_index: Return how the circuits were combined.

    Returns:
        A list of host circuits, containing the original circuits. If return_index=True, returns
        (circuits, partition) where circuits is the list of host circuits and partition is a list of list of
        indices of circuits included in each host circuit.
    """
    qubits = check_circuits_same_qubits(circuits)
    qubit_interference_graph = create_qubit_interference_graph(qubits, coupling_map)
    circuit_interference_graph = create_circuit_interference_graph(
        circuits,
        qubit_interference_graph,
    )

    color_lookup = retworkx.graph_greedy_color(circuit_interference_graph)
    partition = [[] for _ in set(color_lookup.values())]
    for i, color in color_lookup.items():
        partition[color].append(i)
    partition = list(map(sorted, partition))

    grouped_circuits = []
    for part in partition:
        subcircuits = [circuits[i] for i in part]
        circ = combine_circuits(subcircuits)
        grouped_circuits.append(circ)

    # Sort in ascending order of smallest circuit index in parts.
    sorter = sorted(range(len(partition)), key=lambda i: min(partition[i]))
    grouped_circuits = [grouped_circuits[j] for j in sorter]
    partition = [partition[j] for j in sorter]

    if not return_index:
        return grouped_circuits
    if return_index:
        return (grouped_circuits, partition)


def combine_circuits(circuits: Sequence[QuantumCircuit]) -> QuantumCircuit:
    """Combine circuits with the same qubits, but without overlapping active qubits, into one host circuit.

    The new host circuit will have a 'subcircuits' metadata field containing information about the subcircuits
    that uncombine_result() will use to "demultiplex" a single experiment result for a host circuit.
    Subcircuit calibrations and classical registers will be merged into the host circuit.

    Args:
        circuits: Circuits to combine.

    Returns:
        The combined circuit.
    """
    if len(circuits) == 0:
        raise ValueError("Got empty circuits.")

    check_circuits_same_qubits(circuits)
    qregs = circuits[0].qregs

    host_circ = QuantumCircuit(*qregs)
    host_circ.metadata = {'subcircuits': []}

    creg_counter = -1
    used_qubits: Set[Qubit] = set()
    for circ in circuits:
        active_qubits = get_active_qubits(circ)

        if used_qubits & active_qubits:
            raise ValueError("Circuits have overlapping active qubits.")

        used_qubits |= active_qubits

        cregs = []
        for orig_creg in circ.cregs:
            creg_counter += 1
            cregs.append(ClassicalRegister(orig_creg.size, f'c{creg_counter}'))
        host_circ.add_register(*cregs)

        # Check if conflicting calibrations exist.
        for gate_name, gate_dict in circ.calibrations.items():
            for gate_qubits, gate_params in gate_dict.keys():
                if circuit_has_calibration(host_circ, gate_name, gate_qubits, gate_params):
                    raise ValueError(
                        "Multiple circuits contain calibrations for the same (gate, qubits, params)"
                    )

        host_circ.compose(circ, clbits=list(itertools.chain(*cregs)), inplace=True)

        circ_info = {
            'name': circ.name,
            'creg_map': tuple((host_cr.name, cr.name) for host_cr, cr in zip(cregs, circ.cregs)),
        }
        if circ.metadata is not None:
            circ_info['metadata'] = circ.metadata

        host_circ.metadata['subcircuits'].append(circ_info)

    return host_circ


def check_circuits_same_qubits(circuits: Sequence[QuantumCircuit]) -> Sequence[Qubit]:
    if not circuits: return []

    qubits = circuits[0].qubits

    if not all(circ.qubits == qubits for circ in circuits[1:]):
        raise ValueError("All circuits must have the same qubits")

    return qubits


def create_qubit_interference_graph(
        qubits: Sequence[Qubit],
        coupling_map: Sequence[Sequence[int]] = (),
) -> retworkx.PyGraph:
    """Return a graph where vertices are qubits and edges exist if the qubits are coupled."""
    graph = retworkx.PyGraph(multigraph=False)
    graph.add_nodes_from(qubits)
    for i, j in coupling_map:
        graph.add_edge(i, j, None)
    return graph


def create_circuit_interference_graph(
        circuits: Sequence[QuantumCircuit],
        qubit_graph: retworkx.PyGraph,
) -> retworkx.PyGraph:
    """Return a graph where vertices are circuits and edges exist if the circuits have active qubits that
    couple."""
    graph = retworkx.PyGraph(multigraph=False)
    graph.add_nodes_from(circuits)

    if len(circuits) == 0:
        return graph

    circuit_qubits = {
        i: get_active_qubits(circ)
        for i, circ in enumerate(circuits)
    }

    for i, i_qubits in circuit_qubits.items():
        i_nodes = set(map(qubit_graph.find_node_by_weight, i_qubits))
        i_nodes -= {None}
        i_adj = set(itertools.chain(*map(qubit_graph.adj, i_nodes)))
        for j, j_qubits in circuit_qubits.items():
            j_nodes = set(map(qubit_graph.find_node_by_weight, j_qubits))
            j_nodes -= {None}
            if j == i: continue
            if not ((i_adj | i_nodes) & j_nodes): continue
            graph.add_edge(i, j, None)

    return graph


def get_active_qubits(circuit: QuantumCircuit) -> set:
    """Return the set of qubits that have at least one instruction performed on them by circuit."""
    return set().union(*[qubits for gate, qubits, clbits in circuit.data])


def circuit_has_calibration(circuit, name, qubits, params):
    """Return True if circuit has a calibration for name with qubits and params."""
    try:
        gate_dict = circuit.calibrations[name]
    except KeyError:
        return False

    return (tuple(qubits), tuple(params)) in gate_dict


def uncombine_result(result: ExperimentResult) -> List[ExperimentResult]:
    """Unpack an ExperimentResult for a circuit containing subcircuits by creating a list of ExperimentResults
    to store the results of each subcircuit as if they had been run individually and return this list."""
    if not has_subcircuits(result):
        return [result]

    header = result.header
    data = result.data
    creg_sizes = header.creg_sizes
    clbit_labels = header.clbit_labels
    creg_indices = get_creg_indices(clbit_labels)

    new_results = []

    for sinfo in header.metadata.get('subcircuits', []):
        new_exp_result = copy.deepcopy(result.to_dict())
        new_header = new_exp_result['header']

        new_header['name'] = sinfo['name']
        new_header['metadata'] = sinfo['metadata']

        creg_map = dict(sinfo['creg_map'])

        new_header['creg_sizes'] = [
            [creg_map[name], size]
            for name, size in creg_sizes
            if name in creg_map
        ]
        new_header['clbit_labels'] = [
            [creg_map[name], bit_index]
            for name, bit_index in clbit_labels
            if name in creg_map
        ]
        new_header['memory_slots'] = len(new_header['clbit_labels'])

        subcircuit_bit_indices = list(itertools.chain.from_iterable(
            indices
            for creg_name, indices in creg_indices.items()
            if creg_name in creg_map
        ))

        if result.meas_level == MeasLevel.CLASSIFIED:
            new_counts = marginal_counts(data.counts, [subcircuit_bit_indices])[0]
            new_exp_result['data'] = {'counts': {hex(s): n for s, n in new_counts.items()}}
        elif result.meas_level == MeasLevel.KERNELED:
            new_memory = np.take(data.memory, subcircuit_bit_indices, axis=-2)
            new_exp_result['data'] = {'memory': new_memory}
        else:
            raise ValueError(f"Unsupported meas_level: {MeasLevel(result.meas_level)!s}")

        new_results.append(ExperimentResult.from_dict(new_exp_result))

    return new_results


def has_subcircuits(exp_result: ExperimentResult) -> bool:
    """Return True if experiment result has a 'subcircuits' metadata field."""
    metadata = getattr(exp_result.header, 'metadata', None) or {}
    if isinstance(metadata.get('subcircuits', None), Iterable):
        return True
    else:
        return False


def bake_schedule(schedule, min_duration: Optional[int] = None):
    """Bake a schedule with multiple pulses into a schedule with a single Waveform with pre/post phase shifts.

    This is mainly used to circumvent minimum pulse length constraints. Frequency shifts are not supported.
    Multiple channels are not supported, schedule must utilise only one drive channel.
    """
    assert len(schedule.channels) == 1
    chan = schedule.channels[0]
    assert isinstance(chan, qpulse.DriveChannel)
    baked = _bake_schedule(schedule)
    with qpulse.build(default_alignment='sequential') as schedule:
        if baked.pre_phase: qpulse.shift_phase(baked.pre_phase, chan)
        if baked.samples.size > 0:
            samples = baked.samples
            if min_duration and baked.samples.size < min_duration:
                samples = np.pad(samples, (0, min_duration-samples.size), constant_values=0.0)
            qpulse.play(qpulse.Waveform(samples), chan)
        if baked.post_phase: qpulse.shift_phase(baked.post_phase, chan)
    return schedule


class BakedSchedule(NamedTuple):
    pre_phase: float
    post_phase: float
    samples: np.ndarray


def _bake_schedule(sched):
    buffer = np.zeros(sched.duration, complex)

    pre_phase = 0.0
    played = False
    phase = 0.0

    for t, inst in sched.instructions:
        if isinstance(inst, qpulse.Delay): continue

        if isinstance(inst, qpulse.ShiftPhase):
            assert not isinstance(inst.phase, ParameterExpression)
            if not played:
                pre_phase += inst.phase
            else:
                phase += inst.phase
        elif isinstance(inst, qpulse.Play):
            played = True
            if isinstance(inst.pulse, qpulse.ParametricPulse):
                samples = inst.pulse.get_waveform().samples
            elif isinstance(inst.pulse, qpulse.Waveform):
                samples = inst.pulse.samples
            else:
                raise ValueError(f"Unknown pulse type: {inst.pulse}")
            buffer[t:t+samples.size] = np.exp(1j*phase) * samples
        elif isinstance(inst, qpulse.Call):
            baked = _bake_schedule(inst.subroutine)

            if not played:
                pre_phase += baked.pre_phase
            else:
                phase += baked.pre_phase

            if baked.samples.size > 0:
                played = True
                buffer[t:t+baked.samples.size] = np.exp(1j*phase) * baked.samples

            if not played:
                pre_phase += baked.post_phase
            else:
                phase += baked.post_phase
        else:
            raise ValueError(f"Unsupported instruction: {inst}")

    return BakedSchedule(
        pre_phase,
        phase,
        buffer,
    )
