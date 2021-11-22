# Required in Python 3.7 to enable PEP 563 -- Postponed Evaluation of Annotations
from __future__ import annotations
import datetime
from typing import Optional

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
from qiskit_experiments.calibration_management import BackendCalibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue

from qiskit_tricks.experiments import Experiment
from qiskit_tricks.fit import cosine_fit, line_fit
from qiskit_tricks.result import ResultLike, resultdf


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


class RabiAnalysis:
    def __init__(self, data: pd.Series, level='amp') -> None:
        self.data = data
        self.level = level
        other_levels = data.index.names.difference([level])

        fit_results = {}
        signals = []
        groupby = data.groupby(other_levels)

        for name, group in groupby:
            amps = group.index.get_level_values(level)
            z = group.values.ravel()

            y = line_fit(np.array([z.real, z.imag])).projection
            y = y - y.mean()
            y /= 0.5 * np.ptp(y) * np.sign(y[abs(amps).argmin()])

            fitres = fit_results[name] = cosine_fit(amps, y)

            sig = pd.DataFrame({'obs': y, 'calc': fitres.calculated}, group.index)
            signals.append(sig)

        self.groups = pd.DataFrame(groupby.groups.keys(), columns=other_levels)

        self.signal = pd.concat(signals)
        self.fit_info = pd.DataFrame(
            ({'pi_amp': x.wavelength/2
             ,'pi_amp_err': x.wavelength_err/2
             ,'mean': x.mean
             ,'mean_err': x.mean_err
             ,'origin': x.origin
             ,'origin_err': x.origin_err
             ,'chisq': x.reduced_chisq}
             for x in fit_results.values()),
            pd.MultiIndex.from_tuples(fit_results.keys(), names=other_levels),
        )

    @classmethod
    def from_experiment(cls, obj: ResultLike, **kwargs) -> RabiAnalysis:
        return cls(resultdf(obj), **kwargs)

    def update(
            self,
            calibrations: BackendCalibrations,
            radians=np.pi,
            amp: Optional[str] = None,
            qubit='qubit',
            schedule='pulse',
            group='default'
    ) -> None:
        amp = amp or self.level

        for i, row in self.fit_info.reset_index().iterrows():
            calibrations.add_parameter_value(
                value=ParameterValue(
                    value=radians/np.pi * row['pi_amp'],
                    date_time=datetime.datetime.now(),
                    exp_id=row.get('job_id'),
                    group=group,
                ),
                param=amp,
                qubits=row[qubit],
                schedule=row[schedule],
            )

    def plot(self, i: Optional[int] = None, **kwargs):
        import matplotlib.pyplot as plt

        if i is None:
            mask = self.groups[kwargs.keys()].agg(lambda x: x.to_dict() == kwargs, axis=1)
            candidates = self.groups.index[mask]
            if len(candidates) == 0:
                raise ValueError(f"No groups for {kwargs}")
            i = candidates[0]

        if i is not None:
            kwargs = self.groups.iloc[i].to_dict()

        from matplotlib.offsetbox import AnchoredText

        caption = '\n'.join(map('{0[0]} = {0[1]!r}'.format, self.groups.iloc[i].items()))

        at = AnchoredText(
            caption,
            prop=dict(fontfamily='monospace', alpha=0.5),
            frameon=True,
            loc='lower right',
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_alpha(0.5)
        plt.gca().add_artist(at)

        sig = self.signal.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))

        plt.plot(sig.index, sig['calc'], c=f'C{i}', alpha=0.5)
        plt.scatter(sig.index, sig['obs'], c=f'C{i}')
        plt.grid()

        plt.xlabel(self.level)
        plt.ylabel('signal [a.u.]')
