from datetime import datetime, timezone
from typing import Dict, Union

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel
import qiskit.pulse as qpulse
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue

from qiskit_tricks import bake_schedule, get_play_instruction
from qiskit_tricks.experiments import Analysis, Experiment
from qiskit_tricks.fit import cosine_fit


__all__ = (
    'PseudoIdentityExperiment',
    'ORRAnalysis',
)


class PseudoIdentityExperiment(Experiment):
    default_run_config = dict(
        shots=4096,
        meas_level=MeasLevel.CLASSIFIED,
    )

    _gate = Gate('mygate', 1, [])

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schedules = {}

    def generate_parameters(self, qubit: int, reps: int, pulse: str, **pulse_params: float):
        if reps%4 != 0:
            raise ValueError("'reps' must be a multiple of 4.")

        return pd.DataFrame(
            data=[(qubit, reps, pulse, *pulse_params.values())],
            columns=('qubit', 'reps', 'pulse', *pulse_params.keys()),
        )

    def _get_schedule(self, qubit: int, pulse: str, pulse_params: Dict[str, float]) -> qpulse.Schedule:
        key = (qubit, pulse, *pulse_params.values())
        if key in self._schedules:
            return self._schedules[key]

        sx_sched = self.calibrations.get_schedule(pulse, qubit, assign_params=pulse_params)
        dchan = qpulse.DriveChannel(qubit)

        with qpulse.build(default_alignment='sequential') as sched:
            qpulse.call(sx_sched)
            qpulse.shift_phase(np.pi, dchan)
            qpulse.call(sx_sched)
            qpulse.shift_phase(np.pi, dchan)
            qpulse.call(sx_sched)
            qpulse.shift_phase(np.pi, dchan)
            qpulse.call(sx_sched)
            qpulse.shift_phase(np.pi, dchan)

        play = get_play_instruction(sx_sched)
        if play and not isinstance(play.pulse,
                (qpulse.Gaussian, qpulse.GaussianSquare, qpulse.Drag, qpulse.Constant)
        ):
            sched = bake_schedule(sched)

        self._schedules[key] = sched

        return sched

    def build(
            self,
            circuit: QuantumCircuit,
            qubit: int,
            reps: int,
            pulse: str,
            **pulse_params: float,
    ) -> None:
        creg = ClassicalRegister(1)
        circuit.add_register(creg)
        for i in range(reps//4):
            circuit.append(self._gate, [qubit])
        circuit.measure(qubit, creg[0])
        sched = self._get_schedule(qubit, pulse, pulse_params)
        circuit.add_calibration(self._gate, [qubit], sched)


class ORRAnalysis(Analysis):
    dont_groupby = ('reps', 'state')

    signal: pd.DataFrame
    fit: pd.DataFrame

    def __init__(self, *args, param: str, **kwargs) -> None:
        self.dont_groupby = self.dont_groupby + (param,)
        self._param = param

        super().__init__(*args, **kwargs)

    def create_tables(
            self,
            data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        param = self._param
        reps = data.index.unique('reps')
        signals = []
        fits = []
        for rep, group in data.groupby('reps'):
            state0_count = group.xs(0, level='state')#.values
            state1_count = group.xs(1, level='state')#.values
            y = state0_count/(state0_count + state1_count)

            x = y.index.get_level_values(param)
            y = y.values.ravel()

            res = cosine_fit(x, y)
            signal = pd.DataFrame({'obs': y, 'calc': res.calculated}, x)
            fit = pd.Series({
                'origin': res.origin,
                'origin_err': res.origin_err,
                'period': res.wavelength,
                'chisq': res.reduced_chisq,
            })

            signals.append(signal)
            fits.append(fit)

        signal = pd.concat(signals, keys=reps, names=reps.names)
        fit = pd.concat(fits, keys=reps, names=reps.names).unstack()

        x = data.index.unique(param).values
        period = fit['period'].values
        origin = fit['origin'].values
        sweep = (origin - x[:, None] + period/2)%period - period/2
        i = np.mean(abs(sweep), axis=1).argmin()
        fit['origin'] = sweep[i] + x[i]

        return dict(signal=signal, fit=fit)

    def update(
            self,
            calibrations: Calibrations,
            qubit='{qubit}',
            schedule='{pulse}',
            group='default',
    ) -> None:
        assert self.index is not None

        for key in self.index:
            fit = self.fit.xs(key, level=self.index.names, drop_level=False).reset_index()
            row = fit.iloc[0]
            calibrations.add_parameter_value(
                value=ParameterValue(
                    value=fit['origin'].mean(),
                    date_time=datetime.now(timezone.utc).astimezone(),
                    exp_id=row.get('job_id'),
                    group=group.format(**row),
                ),
                param=self._param,
                qubits=int(qubit.format(**row)),
                schedule=schedule.format(**row),
            )

    def plot(self):
        import matplotlib.pyplot as plt

        param = self._param
        signal = self.first().signal
        fit = self.first().fit

        for rep, group in signal.groupby('reps'):
            x = group.index.get_level_values(param)
            plt.scatter(x, group['obs'])
            plt.plot(x, group['calc'], label=f'{rep} reps')
            plt.gca().axvline(fit.xs(rep, level='reps')['origin'].iloc[0], c='red')

        plt.grid(True)

        plt.xlabel(param)
        plt.ylabel(r'$|0\rangle$ population')

        return self
