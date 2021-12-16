from collections import defaultdict
import functools
import itertools
import random
from typing import Dict, MutableSequence, Optional, Sequence

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
import qiskit.pulse as qpulse
from qiskit.quantum_info import Clifford, StabilizerTable

from qiskit_tricks.experiments import Experiment


__all__ = (
    'SXBenchmarkExperiment',
)


# Tabulate single qubit Cliffords.
CLIFFORD = list(map(Clifford, map(StabilizerTable.from_labels, [
    ['+X', '+Z'],
    ['-X', '+Z'],
    ['+X', '-Z'],
    ['-X', '-Z'],

    ['+Y', '+Z'],
    ['-Y', '+Z'],
    ['+Y', '-Z'],
    ['-Y', '-Z'],

    ['+Z', '+X'],
    ['-Z', '+X'],
    ['+Z', '-X'],
    ['-Z', '-X'],

    ['+Y', '+X'],
    ['-Y', '+X'],
    ['+Y', '-X'],
    ['-Y', '-X'],

    ['+X', '+Y'],
    ['-X', '+Y'],
    ['+X', '-Y'],
    ['-X', '-Y'],

    ['+Z', '+Y'],
    ['-Z', '+Y'],
    ['+Z', '-Y'],
    ['-Z', '-Y'],
])))


class FourGate(Gate):
    def __init__(self, theta, phi, lamb, **kwargs) -> None:
        super().__init__('four', 1, [theta, phi, lamb], **kwargs)

        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.rz(phi*np.pi - np.pi, 0)
        qc.sx(0)
        qc.rz(lamb*np.pi + np.pi, 0)
        qc.sx(0)
        qc.rz(theta*np.pi - np.pi, 0)
        qc.sx(0)
        qc.rz(-(phi+lamb+theta)*np.pi + np.pi, 0)

        self.definition = qc

    def create_schedule(self, sx_sched: qpulse.Schedule) -> qpulse.Schedule:
        assert not self.is_parameterized()
        theta, phi, lamb = self.params

        play_inst = get_play_instruction(sx_sched)
        if play_inst is None:
            raise ValueError("'sx_sched' has no Play instruction.")

        pulse = play_inst.pulse
        ch = play_inst.channel

        samples: np.ndarray
        if isinstance(pulse, qpulse.Waveform):
            samples = pulse.samples
        elif isinstance(pulse, qpulse.library.ParametricPulse):
            samples = pulse.get_waveform().samples
        else:
            raise RuntimeError

        with qpulse.build() as sched:
            qpulse.play(
                qpulse.Waveform(np.concatenate([
                    samples,
                    -np.exp(-1j*np.pi*phi)*samples,
                    np.exp(-1j*np.pi*(phi+lamb))*samples,
                    -np.exp(-1j*np.pi*(phi+lamb+theta))*samples,
                ])),
                ch
            )

        return sched


def get_play_instruction(sched: qpulse.Schedule) -> Optional[qpulse.Play]:
    for t, inst in sched.instructions:
        if isinstance(inst, qpulse.Play):
            return inst
        elif isinstance(inst, qpulse.Call):
            x = get_play_instruction(inst.subroutine)
            if x: return x
        else:
            continue

    return None


def create_sx_benchmark_sequences(
        lengths: Sequence[int],
        seed=0,
        *,
        verbose=False,
) -> Sequence[QuantumCircuit]:
    unique_lengths = set(lengths)
    if any(l%2 for l in unique_lengths):
        raise ValueError("'lengths' must be a sequence of integers divisible by 2.")

    random.seed(seed)
    cliffs = random.choices(CLIFFORD, k=max(lengths)//2-1)
    mutators = random.choices(CLIFFORD, k=max(lengths)//2)

    qreg = QuantumRegister(1, 'q')
    benchmark_circs = {}

    def append_block(dest: MutableSequence, block: QuantumCircuit):
        if dest:
            last_rz, qargs, *_ = dest.pop()
            next_rz, *_ = block.data[0]
            dest.extend([
                (RZGate(last_rz.params[0] + next_rz.params[0]), qargs, []),
                *block.data[1:],
            ])
        else:
            dest.extend(block.data)

    instructions = defaultdict(list)
    checkpoints = [l//2 for l in unique_lengths]
    buffer = []
    for count, (m, c) in enumerate(zip(mutators, cliffs), start=1):
        block = bake_cliffords(m, m.adjoint().dot(c), qreg)
        append_block(buffer, block)
        if 2*(count+1) in unique_lengths:
            if verbose: print(f"Generating benchmark sequence... (seed: {seed}, length: {2*(count+1)})\r",
                              end='')
            instructions[2*(count+1)] = buffer.copy()
            checkpoints.pop(0)

    for l in sorted(unique_lengths):
        if verbose: print(f"Assembling benchmark circuit...  (seed: {seed}, length: {l})\r", end='')
        benchmark_circs[l] = circ = QuantumCircuit(qreg)
        if l == 0: continue
        circ.data += instructions[l]
        recovery = functools.reduce(Clifford.compose, cliffs[:l//2-1], Clifford.from_label('I')).adjoint()
        m = mutators[l//2-1]
        block = bake_cliffords(m, m.adjoint().dot(recovery), qreg)
        append_block(circ.data, block)

    if verbose: print()

    return [benchmark_circs[l] for l in lengths]


def bake_cliffords(a: Clifford, b: Clifford, qreg: Optional[QuantumRegister] = None) -> QuantumCircuit:
    v = u3_from_clifford(a)
    w = u3_from_clifford(b)
    qc = QuantumCircuit(qreg or 1)
    qc.rz(w[2]*np.pi, 0)
    four = FourGate(v[0], w[0], (v[2]+w[1])%2)
    qc.append(four, [0])
    qc.rz((v[0]+v[1]+v[2]+w[0]+w[1])*np.pi, 0)
    return qc


_clifford_u3 = {}
def u3_from_clifford(C):
    """Convert a Clifford to a float triple corresponding to the angles in its
    U3 decomposition."""
    global _clifford_u3
    key = (*C.table.array.ravel(), *C.table.phase)
    if key not in _clifford_u3:
        _clifford_u3[key] = _u3_from_clifford(C)
    return _clifford_u3[key]


def _u3_from_clifford(C):
    mat = C.to_matrix()

    if np.isclose(abs(mat[0, 0]), 1.0):
        theta = 0
        phi = np.angle(mat[1, 1]) - np.angle(mat[0, 0])
        lamb = 0
    elif np.isclose(abs(mat[0, 0]), 0.0):
        theta = np.pi
        phi = np.angle(mat[1, 0])
        lamb = np.angle(-mat[0, 1])
    else:
        theta = np.arccos(abs(mat[0, 0])) * 2
        phi = np.angle(mat[1, 0]/np.sin(theta/2)) - np.angle(mat[0, 0])
        lamb = np.angle(mat[1, 1]/mat[0, 0]) - phi

    # Convert to fractions of pi.
    theta = round(theta/np.pi, 2) % 2
    phi = round(phi/np.pi, 2) % 2
    lamb = round(lamb/np.pi, 2) % 2

    return theta, phi, lamb


def create_four_schedules(sx_sched):
    scheds = {}
    for params in _clifford_four_params():
        scheds[params] = FourGate(*params).create_schedule(sx_sched)
    return scheds


@functools.lru_cache
def _clifford_four_params():
    return {
        tuple(bake_cliffords(c1, c2).get_instructions('four')[0][0].params)
        for c1, c2 in itertools.product(CLIFFORD, CLIFFORD)
    }


class SXBenchmarkExperiment(Experiment):
    default_run_config = dict(
        shots=4096,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.SINGLE,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._benchmark_templates = {}
        self._four_schedules = {}

    def generate_parameters(self, qubit: int, seed: int, length: int, pulse: str, **pulse_params: float):
        if length%2 != 0:
            raise ValueError("'length' must be a multiple of 2.")

        return pd.DataFrame(
            data=[(qubit, seed, length, pulse, *pulse_params.values())],
            columns=('qubit', 'seed', 'length', 'pulse', *pulse_params.keys()),
        )

    def _get_benchmark_template(self, seed: int, length: int) -> QuantumCircuit:
        if (seed, length) not in self._benchmark_templates:
            self._create_benchmark_templates(seed)

        return self._benchmark_templates[seed, length]

    def _create_benchmark_templates(self, seed: int) -> None:
        assert self.parameters_table is not None

        lengths = self.parameters_table.query(f'seed == {seed}')['length'].unique()
        circs = create_sx_benchmark_sequences(lengths, seed, verbose=True)
        for l, circ in zip(lengths, circs):
            self._benchmark_templates[seed, l] = circ

    def _get_four_schedules(self, qubit: int, pulse: str, pulse_params: Dict[str, float]) -> Dict[tuple, qpulse.Schedule]:
        key = (qubit, pulse, *pulse_params.values())
        if key not in self._four_schedules:
            sx_sched = self.calibrations.get_schedule(pulse, qubit, assign_params=pulse_params)
            self._four_schedules[key] = create_four_schedules(sx_sched)
        return self._four_schedules[key]

    def build(self, circuit: QuantumCircuit, qubit: int, seed: int, length: int, pulse: str, **pulse_params: float) -> None:
        creg = ClassicalRegister(1)
        circuit.add_register(creg)
        template = self._get_benchmark_template(seed, length)
        circuit.compose(template, qubits=[qubit], inplace=True)
        circuit.measure(qubit, creg[0])

        sx_sched = self.calibrations.get_schedule(pulse, qubit, assign_params=pulse_params)
        sx_pulse = get_play_instruction(sx_sched).pulse

        if sx_pulse.duration >= 64 or isinstance(sx_pulse, (qpulse.Gaussian, qpulse.GaussianSquare, qpulse.Drag)) and sx_pulse.duration >= 64:
            circuit.data = circuit.decompose('four').data
            circuit.add_calibration('sx', [qubit], sx_sched)
        else:
            for params, sched in self._get_four_schedules(qubit, pulse, pulse_params).items():
                circuit.add_calibration('four', [qubit], sched, params)
