# Required in Python 3.7 to enable PEP 563 -- Postponed Evaluation of Annotations
from __future__ import annotations
import datetime
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue

from qiskit_tricks.experiments import Experiment
from qiskit_tricks.experiments import Analysis
from qiskit_tricks.fit import cosine_fit, line_fit


__all__ = [
    'RabiExperiment',
    'EFRabiExperiment',
    'RabiAnalysis',
]


class RabiExperiment(Experiment):
    default_run_config = dict(
        shots=4096,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
    )

    parameter_names = ('qubit', 'pulse', 'amp')

    _gate = Gate('rabi', num_qubits=1, params=[])

    def generate_parameters(
            self,
            qubit: int,
            pulse: str,
            amp: Optional[complex] = None,
            amp_start=None,
            amp_stop=None,
            amp_num=None,
    ):
        if amp is not None:
            yield qubit, pulse, amp
            return

        for amp in np.linspace(amp_start, amp_stop, amp_num):
            yield qubit, pulse, amp

    def build(self, circuit: QuantumCircuit, qubit: int, pulse: str, amp: complex) -> None:
        creg = ClassicalRegister(1)
        circuit.add_register(creg)
        circuit.append(self._gate, [qubit])
        circuit.measure(qubit, creg[0])

        sched = self.calibrations.get_schedule(pulse, qubit, assign_params={'amp': amp})
        circuit.add_calibration(self._gate, [qubit], sched)


class EFRabiExperiment(RabiExperiment):
    def build(self, circuit: QuantumCircuit, qubit: int, **params) -> None:
        circuit.x(qubit)
        super().build(circuit, qubit, **params)


class RabiAnalysis(Analysis):
    dont_groupby = ('amp',)

    signal: pd.Series
    fit: pd.Series

    def create_tables(
            self,
            data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        amp = data.index.get_level_values('amp')
        values = data.values.ravel()

        y = line_fit(np.array([values.real, values.imag])).projection
        y = y - y.mean()
        y /= 0.5 * np.ptp(y) * np.sign(y[abs(amp).argmin()])

        res = cosine_fit(amp, y)

        signal = pd.DataFrame({'obs': y, 'calc': res.calculated}, amp)
        fit = pd.Series({
            'pi_amp': res.wavelength/2,
            'pi_amp_err': res.wavelength_err/2,
            'mean': res.mean,
            'mean_err': res.mean_err,
            'origin': res.origin,
            'origin_err': res.origin_err,
            'chisq': res.reduced_chisq,
        })

        return dict(signal=signal, fit=fit)

    def update(
            self,
            calibrations: Calibrations,
            radians: Union[float, str] = np.pi,
            amp='amp',
            qubit='{qubit}',
            schedule='{pulse}',
            group='default',
    ) -> None:
        for _, row in self.fit.reset_index().iterrows():
            if isinstance(radians, str):
                radiansf = float(radians.format(**row))
            else:
                radiansf = radians

            calibrations.add_parameter_value(
                value=ParameterValue(
                    value=radiansf/np.pi * row['pi_amp'],
                    date_time=datetime.datetime.now(),
                    exp_id=row.get('job_id'),
                    group=group.format(**row),
                ),
                param=amp.format(**row),
                qubits=int(qubit.format(**row)),
                schedule=schedule.format(**row),
            )

    def plot(self, *, c: Optional[str] = None, label: Optional[str] = None):
        import matplotlib.pyplot as plt

        signal = self.first().signal

        if self.index is not None:
            key = self.first().index.to_frame().iloc[0].to_dict()
            if label:
                label = label.format(**key)
            if c:
                c = c.format(**key)

        amp = signal.index.get_level_values('amp')

        plt.plot(
            amp,
            signal['calc'],
            c=c,
            alpha=0.5,
            label=label,
        )
        plt.scatter(amp, signal['obs'], c=c)
        plt.grid(True)

        plt.xlabel('amplitude')
        plt.ylabel('signal [a.u.]')

        return self
