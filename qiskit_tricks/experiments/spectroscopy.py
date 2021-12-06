from __future__ import annotations
from typing import Optional, cast, Dict, Union

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
import qiskit.pulse as qpulse
import scipy.optimize
import scipy.signal

from qiskit_tricks.experiments import Experiment, Analysis
from qiskit_tricks.fit import line_fit


__all__ = [
    'SpectroscopyExperiment',
    'EFSpectroscopyExperiment',
    'SpectroscopyAnalysis',
]


class SpectroscopyExperiment(Experiment):
    default_run_config = dict(  # type: ignore
        shots=2048,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE,
    )

    parameter_names = ('qubit', 'freq', 'sigma', 'amp')

    _gate = Gate('probe', num_qubits=1, params=[])

    def generate_parameters(
            self,
            qubit: int,
            freq: Optional[float] = None,
            freq_offset: Optional[float] = None,
            sigma: Optional[float] = None,
            resolution: Optional[float] = None,
            amp: Optional[complex] = None,
    ):
        if sigma is None:
            assert resolution is not None
            sigma = 1/(2*np.pi * resolution)

        if freq is None:
            assert freq_offset is not None
            lo_freq = cast(float, self.calibrations.get_parameter_value('qubit_lo_freq', qubits=qubit))
            freq = lo_freq + freq_offset

        if amp is None:
            # Automatically calculate an "almost pi" pulse amplitude.
            ham = self.backend.configuration().hamiltonian['vars']
            omegad = ham['omegad%d'%qubit]
            amp = 0.9*np.pi / (np.sqrt(2*np.pi)*sigma*omegad)

        yield qubit, freq, sigma, amp

    def build(self, circuit: QuantumCircuit, qubit: int, freq: float, sigma: float, amp: complex) -> None:
        creg = ClassicalRegister(1)
        circuit.add_register(creg)
        circuit.append(self._gate, [qubit])
        circuit.measure(qubit, creg[0])

        run_config = self.run_config()
        dt = self.backend.configuration().dt
        dchan = qpulse.DriveChannel(qubit)
        sigma_ = sigma/dt
        num_samples = max(int(16*round(4*sigma_/16)), 64)
        with qpulse.build(self.backend) as sched:
            with qpulse.frequency_offset(freq - run_config['qubit_lo_freq'][qubit], dchan):
                qpulse.play(qpulse.Gaussian(num_samples, amp, sigma_), dchan)
        circuit.add_calibration(self._gate, [qubit], sched)


class EFSpectroscopyExperiment(SpectroscopyExperiment):
    def build(self, circuit: QuantumCircuit, qubit: int, **params):
        circuit.x(qubit)
        super().build(circuit, qubit, **params)


class SpectroscopyAnalysis(Analysis):
    dont_groupby = ('freq',)

    signal: pd.Series
    peaks: pd.DataFrame

    def create_tables(
            self,
            data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        raw = data.values.ravel()
        y = line_fit(np.array([raw.real, raw.imag])).projection
        y = y - y.mean()
        if abs(y.min()) > abs(y.max()):
            y = -y
        y = (y-y.min())/np.ptp(y)

        signal = pd.Series(y, data.index)
        peaks = find_peaks(signal)

        return dict(signal=signal, peaks=peaks)

    def plot(self, c: Optional[str] = None, label: Optional[str] = None):
        import matplotlib.pyplot as plt

        head = self if self.index is None else self[0]

        signal = head.signal
        peaks = head.peaks

        if self.index is not None and not self.index.empty:
            key = self.index[[0]].to_frame().iloc[0].to_dict()
            if label:
                label = label.format(**key)
            if c:
                c = c.format(**key)

        for _, peak in peaks.iterrows():
            plt.gca().axvline(peak['freq'], c='red', lw=1.0)
            plt.gca().axvspan(
                peak['freq'] - 2*peak['freq_err'],
                peak['freq'] + 2*peak['freq_err'],
                facecolor='red',
                edgecolor='none',
                alpha=0.2
            )

        plt.scatter(signal.index, signal, c=c, s=2.0, label=label)
        plt.grid()

        plt.xlabel('frequency [Hz]')
        plt.ylabel('signal [a.u.]')

        return self


def find_peaks(signal, prominence=0.1):
    def gaussian(x, *params):
        x0, sigma, amp, base = params
        return amp * np.exp(-0.5*(x-x0)**2/sigma**2) + base

    signal = signal.sort_index()
    peak_indices, peak_props = scipy.signal.find_peaks(signal.values, prominence=prominence)
    out = []

    for peak_index, peak_prom in zip(peak_indices, peak_props['prominences']):
        peak_freq = signal.index[peak_index]
        peak_height = signal.iloc[peak_index]
        half_height = peak_height - 0.5*peak_prom
        right_x = (signal.iloc[peak_index:] <= half_height).idxmax()
        left_x = (signal.iloc[peak_index::-1] <= half_height).idxmax()
        fwhm = right_x - left_x
        peak_sigma = fwhm/2.355

        win = signal[(peak_freq-2*peak_sigma <= signal.index) & (signal.index <= peak_freq+2*peak_sigma)]

        p0 = (peak_freq, peak_sigma, peak_height, 0)
        if win.size > 2:
            popt, pcov = scipy.optimize.curve_fit(
                gaussian,
                win.index,
                win.values,
                p0=p0,
                method='trf',
                x_scale='jac',
            )
            out.append((*popt, np.sqrt(pcov[0,0])))
        else:
            out.append((*p0, np.nan))

    # Sort peaks by height (largest to smallest).
    out = sorted(out, key=lambda t: t[2], reverse=True)

    out = pd.DataFrame(out, columns=['freq', 'sigma', 'height', 'base', 'freq_err']).rename_axis('peak')

    return out
