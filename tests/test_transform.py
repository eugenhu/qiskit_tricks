import copy
from typing import Mapping, Optional, Sequence, overload

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
from qiskit.pulse import (
    Call,
    DriveChannel,
    Gaussian,
    Play,
    Schedule,
    ShiftPhase,
    Waveform,
)
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    QobjExperimentHeader,
)
import retworkx

from qiskit_tricks.transform import (
    bake_schedule,
    combine_circuits,
    create_circuit_interference_graph,
    create_qubit_interference_graph,
    has_subcircuits,
    parallelize_circuits,
    uncombine_result,
)


# Default number of qubits in test circuits.
N = 7


def test_parallelize_circuits_1q():
    """
    Test parallelizing 1-qubit circuits acting on different qubits.
    Use OpenQASM in this test just as a place to demo it.
    """
    subcircs = [
        QuantumCircuit.from_qasm_str(f"""
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[{N}];
            creg c[1];
            h q[{i}];
            measure q[{i}] -> c[0];
        """)
        for i in range(3)
    ]

    hostcircs = parallelize_circuits(subcircs)

    assert len(hostcircs) == 1

    expect = QuantumCircuit.from_qasm_str(f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{N}];
    """ + ''.join(f"""
        creg c{i}[1];
        h q[{i}];
        measure q[{i}] -> c{i}[0];
    """ for i in range(3)))

    assert_circuit_eq(hostcircs[0], expect)


def test_parallelize_circuits_2q():
    """Test parallelizing 2-qubit circuits with overlapping active qubits."""
    subcircs = [
        hcircuit([i, i+1], [0, 1], cregs=[('c', 2)])
        for i in range(3)
    ]
    hostcircs = parallelize_circuits(subcircs)

    expect = [
        hcircuit(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            cregs=[('c0', 2), ('c1', 2)]),
        hcircuit(
            [1, 2],
            [0, 1],
            cregs=[('c0', 2)],
        ),
    ]

    assert_circuit_eq(hostcircs, expect)


def test_parallelize_circuits_return_index_1q():
    subcircs = [hcircuit([i], [0]) for i in range(3)]

    _, index = parallelize_circuits(subcircs, return_index=True)

    assert index == [list(range(len(subcircs)))]


def test_parallelize_circuits_1q_with_coupling_map():
    subcircs = [
        hcircuit([i], [0], cregs=[('c', 1)])
        for i in range(3)
    ]

    # Coupling map:
    #   0 — 1 — 2
    coupling_map = [[0, 1], [1, 2]]

    hostcircs, index = parallelize_circuits(subcircs, coupling_map=coupling_map, return_index=True)

    expect = [
        hcircuit([0, 2], [0, 1], cregs=[('c0', 1), ('c1', 1)]),
        hcircuit([1], [0], cregs=[('c0', 1)]),
    ]

    assert_circuit_eq(hostcircs, expect)
    assert index == [[0, 2], [1]]


def test_parallelize_circuits_2q_with_coupling_map():
    subcircs = [
        hcircuit([0, 2], cregs=[('c', 2)]),
        hcircuit([1, 3], cregs=[('c', 2)]),
        hcircuit([3, 5], cregs=[('c', 2)]),
        hcircuit([4, 6], cregs=[('c', 2)]),
    ]

    # Coupling map:
    #   0 — 1 — 2
    #       |
    #       3
    #       |
    #   4 — 5 — 6
    coupling_map = [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]

    hostcircs, index = parallelize_circuits(subcircs, coupling_map=coupling_map, return_index=True)

    expect = [
        hcircuit([0, 2, 3, 5], cregs=[('c0', 2), ('c1', 2)]),
        hcircuit([1, 3, 4, 6], cregs=[('c0', 2), ('c1', 2)]),
    ]

    assert_circuit_eq(hostcircs, expect)
    assert index == [[0, 2], [1, 3]]


def test_parallelize_circuits_metadata():
    subcircs = [hcircuit([i], [i]) for i in range(3)]

    for i, circ in enumerate(subcircs):
        circ.name = f'mycircuit-{i}'
        circ.metadata = {'abc': i}

    hostcirc = parallelize_circuits(subcircs)[0]

    assert isinstance(hostcirc.metadata, Mapping)

    for i, (sinfo, subcirc) in enumerate(zip(hostcirc.metadata['subcircuits'], subcircs)):
        assert sinfo['name'] == subcirc.name
        assert sinfo['metadata'] == subcirc.metadata
        for host_creg, circ_creg in sinfo['creg_map']:
            # Assert mapped cregs have same size.
            assert {cr.name: cr.size for cr in hostcirc.cregs}[host_creg] \
                   == {cr.name: cr.size for cr in subcirc.cregs}[circ_creg]


def test_parallelize_circuits_some_no_active_qubit():
    qregs = [('q', 1)]
    cregs = [('c', 1)]
    circ0 = quantum_circuit(qregs, cregs)
    circ1 = hcircuit(0, 0, qregs, cregs)

    hostcircs, index = parallelize_circuits([circ0, circ1], return_index=True)

    assert index == [[0, 1]]
    assert_circuit_eq(
        hostcircs,
        [hcircuit(0, 1, qregs, cregs=[('c0', 1), ('c1', 1)])],
    )


def test_parallelize_circuits_all_no_active_qubit():
    hostcircs, index = parallelize_circuits([quantum_circuit(), quantum_circuit()], return_index=True)

    assert index == [[0, 1]]
    assert_circuit_eq(
        hostcircs,
        [quantum_circuit(cregs=[('c0', N), ('c1', N)])],
    )


def test_parallelize_circuits_merges_calibrations():
    circ0 = hcircuit(0, 0)
    circ1 = hcircuit(1, 0)

    schedule0 = gaussian_schedule(0)
    schedule1 = gaussian_schedule(0)

    circ0.add_calibration("h", [0], schedule0, [.12, .45])
    circ1.add_calibration("h", [1], schedule1, [1.2, 4.5])

    hostcirc = parallelize_circuits([circ0, circ1])[0]

    h_dict = hostcirc.calibrations['h']

    for (gate_qubits, gate_params), gate_schedule in h_dict.items():
        if gate_qubits == (0,):
            assert gate_params == (.12, .45)
            assert gate_schedule is schedule0
        elif gate_qubits == (1,):
            assert gate_params == (1.2, 4.5)
            assert gate_schedule is schedule1
        else:
            assert False, gate_qubits


def test_parallelize_circuits_calibrations_conflict():
    # Ideally, this should not be a problem if circuits don't contain calibrations with qubits outside of the
    # qubits they use. Check that an exception is raised if a calibration is overwritten anyway.

    circ0 = hcircuit(0, 0)
    circ1 = hcircuit(1, 0)

    schedule0 = gaussian_schedule(0)
    schedule1 = gaussian_schedule(0)

    circ0.add_calibration("h", [0], schedule0)
    circ1.add_calibration("h", [0, 1], schedule1)

    # This is fine, calibration in circ1 won't overwrite circ0.
    parallelize_circuits([circ0, circ1])

    # This will override a calibration entry.
    circ1.add_calibration("h", [0], schedule1)

    with pytest.raises(ValueError, match="calibration"):
        parallelize_circuits([circ0, circ1])


def test_combine_circuits():
    subcircs = [hcircuit([i], [0], cregs=[('c', 1)]) for i in range(3)]

    host_circ = combine_circuits(subcircs)

    assert_circuit_eq(
        host_circ,
        hcircuit([0, 1, 2], [0, 1, 2], cregs=[('c0', 1), ('c1', 1), ('c2', 1)]),
    )


def test_combine_circuits_empty():
    with pytest.raises(ValueError, match="empty"):
        combine_circuits([])


def test_combine_circuits_with_overlapping_active_qubits():
    """Combining circuits with overlapping active qubits should fail."""
    subcircs = [
        hcircuit([i, i+1], [0, 1], cregs=[('c', 2)])
        for i in range(3)
    ]

    with pytest.raises(ValueError, match="overlap"):
        combine_circuits(subcircs)


def test_combine_circuits_metadata():
    subcircs = [hcircuit([i], [i]) for i in range(3)]

    for i, circ in enumerate(subcircs):
        circ.name = f'mycircuit-{i}'
        circ.metadata = {'abc': i}

    host_circ = combine_circuits(subcircs)

    assert isinstance(host_circ.metadata, Mapping)

    for i, (sinfo, subcirc) in enumerate(zip(host_circ.metadata['subcircuits'], subcircs)):
        assert sinfo['name'] == subcirc.name
        assert sinfo['metadata'] == subcirc.metadata
        for host_creg, circ_creg in sinfo['creg_map']:
            # Assert mapped cregs have same size.
            assert {cr.name: cr.size for cr in host_circ.cregs}[host_creg] \
                   == {cr.name: cr.size for cr in subcirc.cregs}[circ_creg]


def test_combine_circuits_some_no_active_qubit():
    qregs = [('q', 1)]
    cregs = [('c', 1)]
    circs = [
        quantum_circuit(qregs, cregs),
        hcircuit(0, 0, qregs, cregs),
    ]

    host_circ = combine_circuits(circs)

    assert_circuit_eq(
        host_circ,
        hcircuit(0, 1, qregs, cregs=[('c0', 1), ('c1', 1)]),
    )


def test_combine_circuits_all_no_active_qubit():
    host_circ = combine_circuits([quantum_circuit(), quantum_circuit()])

    assert_circuit_eq(
        host_circ,
        quantum_circuit(cregs=[('c0', N), ('c1', N)]),
    )


def test_combine_circuits_merges_calibrations():
    circ0 = hcircuit(0, 0)
    circ1 = hcircuit(1, 0)

    schedule0 = gaussian_schedule(0)
    schedule1 = gaussian_schedule(0)

    circ0.add_calibration("h", [0], schedule0, [.12, .45])
    circ1.add_calibration("h", [1], schedule1, [1.2, 4.5])

    host_circ = combine_circuits([circ0, circ1])

    h_dict = host_circ.calibrations['h']

    for (gate_qubits, gate_params), gate_schedule in h_dict.items():
        if gate_qubits == (0,):
            assert gate_params == (.12, .45)
            assert gate_schedule is schedule0
        elif gate_qubits == (1,):
            assert gate_params == (1.2, 4.5)
            assert gate_schedule is schedule1
        else:
            assert False, gate_qubits


def test_combine_circuits_calibrations_conflict():
    circ0 = hcircuit(0, 0)
    circ1 = hcircuit(1, 0)

    schedule0 = gaussian_schedule(0)
    schedule1 = gaussian_schedule(0)

    circ0.add_calibration("h", [0], schedule0)
    circ1.add_calibration("h", [0, 1], schedule1)

    # This is fine, calibration in circ1 won't overwrite circ0.
    combine_circuits([circ0, circ1])

    # This will override a calibration entry.
    circ1.add_calibration("h", [0], schedule1)

    with pytest.raises(ValueError, match="calibration"):
        combine_circuits([circ0, circ1])


def test_create_qubit_interference_graph():
    qubits = list(QuantumRegister(5))

    # Coupling map:
    # 0 — 1 — 2 — 3 — 4
    coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]

    qubit_graph = create_qubit_interference_graph(qubits, coupling_map)

    expect = retworkx.PyGraph()
    expect.add_nodes_from(qubits)
    for i, j in coupling_map:
        expect.add_edge(i, j, None)

    assert retworkx.is_isomorphic(qubit_graph, expect)


def test_create_circuit_interference_graph():
    circs = [
        hcircuit([0, 1], range(2)),
        hcircuit([2, 3], range(2)),
        hcircuit([4, 5], range(2)),
        hcircuit([0, 6], range(2)),
    ]

    # Linear connectivity:
    # 0 — 1 — 2 — 3 — 4 — 5 — 6
    qubit_graph = retworkx.PyGraph()
    qubit_graph.add_nodes_from(circs[0].qubits)
    for i in range(qubit_graph.num_nodes() - 1):
        qubit_graph.add_edge(i, i+1, None)

    circuit_graph = create_circuit_interference_graph(circs, qubit_graph)

    expect = retworkx.PyGraph()
    expect.add_nodes_from(circs)
    expect.add_edge(0, 1, None)
    expect.add_edge(1, 2, None)
    expect.add_edge(2, 3, None)
    expect.add_edge(3, 0, None)

    assert retworkx.is_isomorphic(circuit_graph, expect)


def test_create_circuit_interference_graph2():
    circs = [
        hcircuit([0, 1], range(2)),
        hcircuit([1, 2], range(2)),
    ]

    # No connectivity.
    qubit_graph = retworkx.PyGraph()
    qubit_graph.add_nodes_from(circs[0].qubits)

    circuit_graph = create_circuit_interference_graph(circs, qubit_graph)

    expect = retworkx.PyGraph()
    expect.add_nodes_from(circs)
    expect.add_edge(0, 1, None)

    assert retworkx.is_isomorphic(circuit_graph, expect)


def test_uncombine_result_kerneled_single():
    shots = 3
    memory = np.random.rand(shots, 5, 2)
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
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.SINGLE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'subcircuits': subcircuits, 'ignored_field': 88},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
            memory_slots=memory.shape[1],

            this_field=42,
        ),
        that_field=33,
    )

    unpacked = uncombine_result(result)

    expect = [
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
            data=ExperimentResultData(memory=memory[:, 0:3]),
            header=QobjExperimentHeader(
                name='circuit-0',
                metadata={'thing': 123},
                creg_sizes=[['c', 3]],
                clbit_labels=[['c', i] for i in range(3)],
                memory_slots=3,
                this_field=42,
            ),
            that_field=33,
        ),
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
            data=ExperimentResultData(memory=memory[:, 3:5]),
            header=QobjExperimentHeader(
                name='circuit-1',
                metadata={'thing': 456},
                creg_sizes=[['c', 2]],
                clbit_labels=[['c', i] for i in range(2)],
                memory_slots=2,
                this_field=42,
            ),
            that_field=33,
        ),
    ]

    assert_result_eq(unpacked, expect)


def test_uncombine_result_kerneled_avg():
    shots = 1024
    memory = np.random.rand(5, 2)
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
        shots=shots,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
        data=ExperimentResultData(memory=memory),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'subcircuits': subcircuits, 'ignored_field': 88},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],
            memory_slots=5,

            this_field=42,
        ),
        that_field=33,
    )

    unpacked = uncombine_result(result)

    expect = [
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            data=ExperimentResultData(memory=memory[0:3]),
            header=QobjExperimentHeader(
                name='circuit-0',
                metadata={'thing': 123},
                creg_sizes=[['c', 3]],
                clbit_labels=[['c', i] for i in range(3)],
                memory_slots=3,
                this_field=42,
            ),
            that_field=33,
        ),
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            data=ExperimentResultData(memory=memory[3:5]),
            header=QobjExperimentHeader(
                name='circuit-1',
                metadata={'thing': 456},
                creg_sizes=[['c', 2]],
                clbit_labels=[['c', i] for i in range(2)],
                memory_slots=2,
                this_field=42,
            ),
            that_field=33,
        ),
    ]

    assert_result_eq(unpacked, expect)


def test_uncombine_result_classified():
    shots = 100
    counts = {hex(0b00001): 10
             ,hex(0b01001): 40
             ,hex(0b11101): 30
             ,hex(0b01000): 20}
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
        shots=shots,
        meas_level=MeasLevel.CLASSIFIED,
        data=ExperimentResultData(counts=counts),
        header=QobjExperimentHeader(
            name='big circuit',
            metadata={'subcircuits': subcircuits, 'ignored_field': 88},
            creg_sizes=[['c0', 3], ['c1', 2]],
            clbit_labels=[
                *[['c0', i] for i in range(3)],
                *[['c1', i] for i in range(2)],
            ],

            this_field=42,
        ),
        that_field=33,
    )

    unpacked = uncombine_result(result)

    expect = [
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.CLASSIFIED,
            data=ExperimentResultData(counts={hex(0b000): 20, hex(0b001): 50, hex(0b101): 30}),
            header=QobjExperimentHeader(
                name='circuit-0',
                metadata={'thing': 123},
                creg_sizes=[['c', 3]],
                clbit_labels=[['c', i] for i in range(3)],
                memory_slots=3,
                this_field=42,
            ),
            that_field=33,
        ),
        ExperimentResult(
            success=True,
            shots=shots,
            meas_level=MeasLevel.CLASSIFIED,
            data=ExperimentResultData(counts={hex(0b00): 10, hex(0b01): 60, hex(0b11): 30}),
            header=QobjExperimentHeader(
                name='circuit-1',
                metadata={'thing': 456},
                creg_sizes=[['c', 2]],
                clbit_labels=[['c', i] for i in range(2)],
                memory_slots=2,
                this_field=42,
            ),
            that_field=33,
        ),
    ]

    assert_result_eq(unpacked, expect)


def test_uncombine_result_no_subcircuits():
    result = ExperimentResult(
        success=True,
        shots=1024,
        meas_level=MeasLevel.CLASSIFIED,
        data=ExperimentResultData(),
        header=QobjExperimentHeader(),
    )
    assert uncombine_result(result) == [result]


def test_has_subcircuits_when_metadata_is_None():
    result = ExperimentResult(
        success=True,
        shots=1024,
        data=ExperimentResultData(),
        header=QobjExperimentHeader(
            metadata=None,
        ),
    )
    assert has_subcircuits(result) is False


def test_bake_schedule():
    dchan = DriveChannel(3)
    sched1 = Schedule(ShiftPhase(0.1, dchan))
    sched2 = Schedule(
        ShiftPhase(0.2, dchan),
        Play(Gaussian(32, 0.5, 8), dchan),
    )

    got = bake_schedule(Schedule(
        (0, Call(sched1)),
        (0, Call(sched2)),
        (32, Play(Gaussian(64, 0.1, 12), dchan)),
        (96, Call(sched2)),
        (128, ShiftPhase(0.1, dchan)),
        (128, Call(sched2)),
    ))

    samples = np.zeros(160, complex)
    samples[  0: 32] = Gaussian(32, 0.5, 8).get_waveform().samples
    samples[ 32: 96] = Gaussian(64, 0.1, 12).get_waveform().samples
    samples[ 96:128] = np.exp(0.2j) * Gaussian(32, 0.5, 8).get_waveform().samples
    samples[128:160] = np.exp(0.5j) * Gaussian(32, 0.5, 8).get_waveform().samples

    expect = (
        (  0, ShiftPhase(0.1 + 0.2, dchan)),
        (  0, Play(Waveform(samples), dchan)),
        (160, ShiftPhase(0.5, dchan)),
    )

    assert len(got.instructions) == 3
    t0, inst0 = got.instructions[0]
    t1, inst1 = got.instructions[1]
    t2, inst2 = got.instructions[2]

    assert t0 == expect[0][0]
    assert isinstance(inst0, ShiftPhase)
    assert np.allclose(inst0.phase, expect[0][1].phase)
    assert inst0.channel == dchan

    assert t1 == expect[1][0]
    assert isinstance(inst1, Play)
    assert np.allclose(inst1.pulse.samples, expect[1][1].pulse.samples)
    assert inst1.channel == dchan

    assert t2 == expect[2][0]
    assert isinstance(inst2, ShiftPhase)
    assert np.allclose(inst2.phase, expect[2][1].phase)
    assert inst2.channel == dchan


def test_bake_schedule_with_min_duration():
    dchan = DriveChannel(0)

    got = bake_schedule(
        Schedule(
            Play(Gaussian(48, 0.1, 12), dchan),
        ),
        min_duration=100,
    )

    samples = np.zeros(100, complex)
    samples[:48] = Gaussian(48, 0.1, 12).get_waveform().samples

    expect = (
        (0, Play(Waveform(samples), dchan)),
    )

    assert len(got.instructions) == 1
    t0, inst0 = got.instructions[0]

    assert t0 == expect[0][0]
    assert isinstance(inst0, Play)
    assert np.allclose(inst0.pulse.samples, expect[0][1].pulse.samples)


def assert_circuit_eq(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            assert_circuit_eq(ai, bi)
        return

    from qiskit.converters import circuit_to_dag
    assert circuit_to_dag(a) == circuit_to_dag(b)


@overload
def assert_result_eq(
        a: ExperimentResult,
        b: ExperimentResult,
) -> None:
    ...
@overload
def assert_result_eq(
        a: Sequence[ExperimentResult],
        b: Sequence[ExperimentResult],
) -> None:
    ...
def assert_result_eq(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            assert_result_eq(ai, bi)
        return

    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    for x in (a, b):
        if hasattr(x.data, 'memory') and isinstance(x.data.memory, np.ndarray):
            x.data.memory = x.data.memory.tolist()

    assert a.to_dict() == b.to_dict()


def hcircuit(qubits, cbits=None, qregs=None, cregs=None) -> QuantumCircuit:
    """Create and return a new QuantumCircuit with registers `qregs` and `cregs` where a H gate is applied to
    `qubits` followed by a measurement on `qubits` with results stored in `cbits`."""
    circ = quantum_circuit(qregs, cregs)

    if isinstance(qubits, int):
        qubits = [qubits]

    if cbits is None:
        cbits = list(range(len(qubits)))

    if isinstance(cbits, int):
        cbits = [cbits]

    for q, c in zip(qubits, cbits):
        circ.h(q)
        circ.measure(q, c)

    return circ


def quantum_circuit(
    qregs: Optional[list[tuple[str, int]]] = None,
    cregs: Optional[list[tuple[str, int]]] = None,
) -> QuantumCircuit:
    """Create and return a new QuantumCircuit with registers `qregs`, `cregs`."""
    qregs = qregs or [('q', N)]
    cregs = cregs or [('c', N)]

    if qregs is not None:
        qregs_: list[QuantumRegister] = [QuantumRegister(size, name) for name, size in qregs]
    if cregs is not None:
        cregs_: list[ClassicalRegister] = [ClassicalRegister(size, name) for name, size in cregs]

    return QuantumCircuit(*qregs_, *cregs_)


def gaussian_schedule(ch: int, name: Optional[str] = None) -> Schedule:
    """Create and return a Schedule with a Gaussian pulse on drive channel `ch`."""
    return Schedule(
        Play(Gaussian(64, 1.0, 16),
             DriveChannel(ch)),
        name=name,
    )
