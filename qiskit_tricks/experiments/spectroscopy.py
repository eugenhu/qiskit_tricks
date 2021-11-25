from __future__ import annotations
from typing import Optional, cast

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
import qiskit.pulse as qpulse
import scipy.optimize
import scipy.signal

from qiskit_tricks.experiments import Experiment
from qiskit_tricks.fit import line_fit
from qiskit_tricks.result import ResultLike, resultdf


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
            lo_freq = cast(float, self.calibrations.get_parameter_value('qubit_lo_freq', qubits=0))
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


class SpectroscopyAnalysis:
    def __init__(self, data: pd.Series) -> None:
        self.data = data
        other_levels = data.index.names.difference(['freq'])

        groupby = data.groupby(other_levels)
        signals = []
        for _, group in groupby:
            z = group.values.ravel()
            y = line_fit(np.array([z.real, z.imag])).projection
            y = y - y.mean()
            if abs(y.min()) > abs(y.max()):
                y = -y
            y = (y-y.min())/np.ptp(y)
            sig = pd.Series(y, group.index)
            signals.append(sig)

        self.groups = pd.DataFrame(groupby.groups.keys(), columns=other_levels)
        self.signal = signal = pd.concat(signals)
        self.peaks = signal.groupby(other_levels).apply(lambda x: find_peaks(x.droplevel(other_levels)))

    @classmethod
    def from_experiment(cls, obj: ResultLike, **kwargs) -> SpectroscopyAnalysis:
        return cls(resultdf(obj), **kwargs)

    def plot(self, i: Optional[int] = None, caption=True, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText

        if i is None:
            mask = self.groups[kwargs.keys()].agg(lambda x: x.to_dict() == kwargs, axis=1)
            candidates = self.groups.index[mask]
            if len(candidates) == 0:
                raise ValueError(f"No groups for {kwargs}")
            i = candidates[0]

        if i is not None:
            kwargs = self.groups.iloc[i].to_dict()

        if caption:
            caption_txt = '\n'.join(map('{0[0]} = {0[1]!r}'.format, self.groups.iloc[i].items()))
            at = AnchoredText(
                caption_txt,
                prop=dict(fontfamily='monospace', alpha=0.5),
                frameon=True,
                loc='right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            at.patch.set_alpha(0.5)
            plt.gca().add_artist(at)

        signal = self.signal.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))
        peaks = self.peaks.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))


        for _, peak in peaks.iterrows():
            plt.gca().axvline(peak['freq'], c='red', lw=1.0)
            plt.gca().axvspan(
                peak['freq'] - 2*peak['freq_err'],
                peak['freq'] + 2*peak['freq_err'],
                facecolor='red',
                edgecolor='none',
                alpha=0.2
            )

        plt.scatter(signal.index, signal, c=f'C{i}', s=2.0)
        plt.grid()

        plt.xlabel('frequency [Hz]')
        plt.ylabel('signal [a.u.]')


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
