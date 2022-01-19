from datetime import datetime, timezone
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel
import qiskit.pulse as qpulse
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from scipy.optimize import curve_fit

from qiskit_tricks import bake_schedule, get_play_instruction
from qiskit_tricks.experiments import Analysis, Experiment


__all__ = (
    'FineSXAmpExperiment',
    'FineSXAmpAnalysis',
)


class FineSXAmpExperiment(Experiment):
    default_run_config = dict(
        shots=4096,
        meas_level=MeasLevel.CLASSIFIED,
    )

    _gate = Gate('mygate', 1, [])

    def generate_parameters(self, qubit: int, reps: int, pulse: str, **pulse_params: float):
        return pd.DataFrame(
            data=[(qubit, reps, pulse, *pulse_params.values())],
            columns=('qubit', 'reps', 'pulse', *pulse_params.keys()),
        )

    def _create_schedule(
            self,
            qubit: int,
            reps: int,
            pulse: str,
            pulse_params: Dict[str, float],
    ) -> qpulse.Schedule:
        sx_sched = self.calibrations.get_schedule(pulse, qubit, assign_params=pulse_params)

        with qpulse.build(default_alignment='sequential') as sched:
            for _ in range(reps):
                qpulse.call(sx_sched)

        play = get_play_instruction(sx_sched)
        if play and not isinstance(play.pulse,
                (qpulse.Gaussian, qpulse.GaussianSquare, qpulse.Drag, qpulse.Constant)
        ):
            min_duration = self.backend.configuration().timing_constraints.get('min_length')
            if min_duration and play.duration < min_duration:
                sched = bake_schedule(sched, min_duration=min_duration)

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
        circuit.append(self._gate, [qubit])
        circuit.measure(qubit, creg[0])
        sched = self._create_schedule(qubit, reps, pulse, pulse_params)
        circuit.add_calibration(self._gate, [qubit], sched)

        if 'amp' in pulse_params:
            amp = pulse_params['amp']
        else:
            amp = self.calibrations.get_parameter_value('amp', qubits=qubit, schedule=pulse)

        circuit.metadata['amp'] = amp


class FineSXAmpAnalysis(Analysis):
    dont_groupby = ('reps', 'state')

    signal: pd.Series
    fit: pd.Series

    def create_tables(
            self,
            data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        state0_count = data.xs(0, level='state')
        state1_count = data.xs(1, level='state')
        N = (state0_count + state1_count).values.ravel()
        y = state0_count/N[:, None]
        x = y.index.get_level_values('reps')
        y = y.values.ravel()

        def f(x, freq):
            origin = np.pi/2 / (freq + np.pi/2)
            return 0.5 - 0.5 * np.sin((freq + np.pi/2)*(x - origin))

        popt, pcov = curve_fit(
            f,
            x,
            y,
            p0=0.0,
            method='trf',
            x_scale='jac',
        )
        calc = f(x, *popt)
        # Binomial distribution variance (divided by N-squared).
        var = 0.5**2/N
        reduced_chisq = np.sum((y - calc)**2/var)/(x.size - popt.size)

        signal = pd.DataFrame({'obs': y, 'calc': calc}, x)
        fit = pd.Series({
            'rotation': popt[0],
            'rotation_err': np.sqrt(pcov[0, 0]),
            'chisq': reduced_chisq,
        })

        return dict(signal=signal, fit=fit)

    def update(
            self,
            calibrations: Calibrations,
            amp='amp',
            qubit='{qubit}',
            schedule='{pulse}',
            group='default',
    ) -> None:
        for _, row in self.fit.reset_index().iterrows():
            calibrations.add_parameter_value(
                value=ParameterValue(
                    value=(np.pi/2)/(np.pi/2 + row['rotation']) * row['amp'],
                    date_time=datetime.now(timezone.utc).astimezone(),
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
        fit = self.first().fit

        if self.index is not None:
            key = self.first().index.to_frame().iloc[0].to_dict()
            if label:
                label = label.format(**key)
            if c:
                c = c.format(**key)

        reps = signal.index.get_level_values('reps')
        reps2 = np.linspace(0, max(reps), 200)

        plt.plot(
            reps2,
            0.5 + 0.5*np.cos((np.pi/2 + fit['rotation'][0])*reps2),
            c=c,
            alpha=0.5,
            label=label,
        )
        plt.scatter(reps, signal['obs'], c=c)
        plt.grid(True)

        plt.xlabel('repetitions')
        plt.ylabel(r'$|0\rangle$ population')

        return self
