from collections.abc import Sequence
import copy
import itertools
from typing import Iterable, Literal, overload

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit
from qiskit.compiler.assembler import MeasLevel
from qiskit.result.models import ExperimentResult
import retworkx

from .util import get_creg_indices, marginal_counts


__all__ = [
    'parallelize_circuits',
    'combine_circuits',
    'uncombine_result',
    'has_subcircuits',
]


@overload
def parallelize_circuits(
        circuits: Sequence[QuantumCircuit],
        coupling_map: Sequence[Sequence[int]] = (),
        *,
        return_index: Literal[False] = False,
) -> list[QuantumCircuit]:
    ...
@overload
def parallelize_circuits(
        circuits: Sequence[QuantumCircuit],
        coupling_map: Sequence[Sequence[int]] = (),
        *,
        return_index: Literal[True] = True,
) -> tuple[list[QuantumCircuit], list[list[int]]]:
    ...
def parallelize_circuits(
        circuits,
        coupling_map=(),
        *,
        return_index=False,
):
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
    if len(circuits) == 0:
        raise ValueError("Got empty circuits.")

    check_circuits_same_qubits(circuits)
    qregs = circuits[0].qregs

    host_circ = QuantumCircuit(*qregs)
    host_circ.metadata = {'subcircuits': []}

    creg_counter = -1
    used_qubits: set[Qubit] = set()
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
    graph = retworkx.PyGraph(multigraph=False)
    graph.add_nodes_from(qubits)
    for i, j in coupling_map:
        graph.add_edge(i, j, None)
    return graph


def create_circuit_interference_graph(
        circuits: Sequence[QuantumCircuit],
        qubit_graph: retworkx.PyGraph,
) -> retworkx.PyGraph:
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
    return set().union(*[qubits for gate, qubits, clbits in circuit.data])


def circuit_has_calibration(circ, name, qubits, params):
    try:
        gate_dict = circ.calibrations[name]
    except KeyError:
        return False

    return (tuple(qubits), tuple(params)) in gate_dict


def uncombine_result(result: ExperimentResult) -> list[ExperimentResult]:
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
        new_expresult = copy.deepcopy(result.to_dict())
        new_header = new_expresult['header']

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
            new_expresult['data'] = {'counts': {hex(s): n for s, n in new_counts.items()}}
        elif result.meas_level == MeasLevel.KERNELED:
            new_memory = np.take(data.memory, subcircuit_bit_indices, axis=-2)
            new_expresult['data'] = {'memory': new_memory}
        else:
            raise ValueError(f"Unsupported meas_level: {MeasLevel(result.meas_level)!s}")

        new_results.append(ExperimentResult.from_dict(new_expresult))

    return new_results


def has_subcircuits(expresult: ExperimentResult) -> bool:
    metadata = getattr(expresult.header, 'metadata', {})
    if isinstance(metadata.get('subcircuits', None), Iterable):
        return True
    else:
        return False
