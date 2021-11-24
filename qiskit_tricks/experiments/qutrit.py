# Required in Python 3.7 to enable PEP 563 -- Postponed Evaluation of Annotations
from __future__ import annotations
from typing import Optional, Union, cast, overload
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit
from qiskit.compiler.assembler import MeasLevel, MeasReturnType
import scipy.optimize

from qiskit_tricks.experiments import Experiment
from qiskit_tricks.result import ResultLike, resultdf


class QutritMeasureExperiment(Experiment):
    parameter_names = ('qubit', 'x', 'x12', 'prep')

    _xgate = Gate('myx', num_qubits=1, params=[])
    _x12gate = Gate('myx12', num_qubits=1, params=[])

    def generate_parameters(self, qubit: int, x12: str, x: Optional[str] = None):
        for i in range(3):
            yield qubit, x, x12, i

    def build(self, circuit: QuantumCircuit, qubit: int, x: Optional[str], x12: str, prep: int):
        creg = ClassicalRegister(1)
        circuit.add_register(creg)

        # Prepare state
        if prep >= 1:
            if x is None:
                # If custom x schedule not provided, use default x gate.
                circuit.x(qubit)
            else:
                circuit.append(self._xgate, [qubit])

        if prep == 2:
            circuit.append(self._x12gate, [qubit])

        # Measure
        circuit.measure(qubit, creg[0])

        # Destruct state
        if prep == 2:
            circuit.append(self._x12gate, [qubit])

        if prep >= 1:
            if x is None:
                circuit.x(qubit)
            else:
                circuit.append(self._xgate, [qubit])

        # Add calibrations for x and x12 gates
        if prep >=1 and x is not None:
            x_sched = self.calibrations.get_schedule(x, qubit)
            circuit.add_calibration(self._xgate, [qubit], x_sched)

        if prep == 2:
            x12_sched = self.calibrations.get_schedule(x12, qubit)
            circuit.add_calibration(self._x12gate, [qubit], x12_sched)

    def default_run_config(self) -> Dict[str, Any]:
        run_config = dict(
            shots=min(8192, self.backend.configuration().max_shots),
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
        )

        # Use longest possible rep delay to allow |2> state to fully decay. This might be unnecessary.
        if self.backend.configuration().dynamic_reprate_enabled:
            run_config['rep_delay'] = self.backend.configuration().rep_delay_range[-1]
        else:
            run_config['rep_time'] = self.backend.configuration().rep_times[-1]

        return run_config


class QutritMeasureAnalysis:
    def __init__(self, data: pd.Series) -> None:
        other_levels = data.index.names.difference(['shot', 'prep'])

        self.data = data
        self.groups = groups = pd.DataFrame(data.groupby(other_levels).groups.keys(), columns=other_levels)
        self.prob_params = cast(
            pd.DataFrame,
            data.to_frame().groupby(groups.columns.to_list()).apply(self._max_likelihood_est)
        )
        self.classifier = classifier = QutritClassifier(self.prob_params)
        self.confusion = data.groupby([*groups.columns, 'prep']).apply(classifier.population).unstack()

    @staticmethod
    def _max_likelihood_est(group):
        group = group.squeeze(axis=1)

        # Scale down (I,Q) plane so std ≈ 1 to aid the numerical minimization routines.
        scale = group.xs(0, level='prep').values.reshape(-1, 1).view(np.double).std(axis=0).mean()
        norm_group = group/scale

        pop = np.ones(3)/3
        mean = norm_group.groupby('prep').apply(np.mean).values
        std = 1.0

        def pack(pop: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
            return np.array([
                *stereo_from_sphere(*np.sqrt(pop)),
                *mean.real,
                *mean.imag,
                1/std,  # Easier to optimize the reciprocal.
            ])

        def unpack(params: np.ndarray) -> tuple:
            X, Y = params[0:2]
            mean = params[2:5] + 1j*params[5:8]
            std = 1/params[8]
            pop = sphere_from_stereo(X, Y)**2
            return pop, mean, std

        res = scipy.optimize.minimize(
            lambda params: -log_likelihood(*unpack(params), norm_group.values),
            x0=pack(pop, mean, std),
            method='L-BFGS-B',
        )

        pop, mean, std = unpack(res.x)
        mean *= scale
        std *= scale

        return pd.Series({'mean0': mean[0], 'mean1': mean[1], 'mean2': mean[2], 'std': std}, dtype=object)

    @classmethod
    def from_experiment(cls, obj: ResultLike, **kwargs) -> QutritMeasureAnalysis:
        return cls(resultdf(obj), **kwargs)

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

        data = self.data.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))
        data = data.droplevel(data.index.names.difference(['prep']))
        prob_params = self.prob_params.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))
        confusion = self.confusion.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()))
        confusion = confusion.droplevel(confusion.index.names.difference(['prep'])).sort_index()

        mean = prob_params[['mean0', 'mean1', 'mean2']].iloc[0].values.ravel()
        std = prob_params['std'].iloc[0]

        xmin = mean.real.min() - 4*std
        xmax = mean.real.max() + 4*std
        ymin = mean.imag.min() - 4*std
        ymax = mean.imag.max() + 4*std

        x = np.linspace(xmin, xmax, 1024)
        y = np.linspace(ymin, ymax, 1024)
        xx, yy = np.meshgrid(x, y)
        zz = xx + 1j*yy

        def gaussian(z, mu, std):
            return np.exp(-0.5 * abs(z-mu)**2/std**2)

        color = [
            cmap(gaussian(zz, mean[i], std))/cmap(0.0)
            for i, cmap in enumerate([plt.cm.Blues, plt.cm.Reds, plt.cm.Greens])
        ]

        plt.imshow(np.prod(color, axis=0), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
        plt.xlabel('I [a.u.]')
        plt.ylabel('Q [a.u.]')

        # Randomize observed data so points of different states overlap randomly when drawn.
        shuffled_data = data.sample(frac=1)
        size = plt.gcf().get_size_inches().mean()**2/500
        plt.scatter(
            shuffled_data.values.real,
            shuffled_data.values.imag,
            c=np.take(['#1f77b4', '#ff7f0e', '#2ca02c'], shuffled_data.index),
            s=size,
        )

        # Draw a marker at each mean.
        for i in range(3):
            color = ['blue', 'red', 'green'][i]
            shape = ['o', 's', '^'][i]
            plt.scatter(
                mean[i].real,
                mean[i].imag,
                c=color,
                marker=shape,
                label=f'$|{i}\\rangle$',
                edgecolors='white',
            )
            mu_with_error = confusion.loc[i].values @ mean
            plt.scatter(
                mu_with_error.real,
                mu_with_error.imag,
                facecolors='none',
                marker=shape,
                edgecolors='yellow',
            )

        # Draw circles with radius 1 std.
        for mu in mean:
            circle = plt.Circle(
                (mu.real, mu.imag),
                std,
                facecolor='none',
                edgecolor='white',
                linestyle='--',
            )
            plt.gca().add_patch(circle)

        plt.gca().set_aspect(1.0)
        plt.grid()
        plt.legend(loc='upper left')


class QutritClassifier:
    def __init__(self, params: pd.DataFrame) -> None:
        self.params = params

    @overload
    def population(self, obs: Union[pd.Series, pd.DataFrame]) -> pd.Series: ...
    @overload
    def population(self, obs: np.ndarray, **kwargs) -> np.ndarray: ...

    def population(self, obs, **kwargs):
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            if isinstance(obs, pd.DataFrame):
                if obs.shape[1] == 1:
                    obs = obs.squeeze(axis=1)
                else:
                    raise ValueError("'obs' must be a Series or single-columned DataFrame.")
            kwargs = {
                k: obs.index.get_level_values(k)[0]
                for k in self.params.index.names
                if k in obs.index.names
            }

        unexpected_kw = set(kwargs.keys()).difference(self.params.index.names)
        if unexpected_kw:
            raise TypeError(f"Unexpected keywords: {unexpected_kw}.")

        if kwargs:
            indices = np.flatnonzero(
                self.params.index.to_frame()[kwargs.keys()]
                                 .agg(lambda x: x.to_dict() == kwargs, axis=1)
                                 .to_numpy()
            )
            if indices.size == 0:
                raise ValueError(f"No matches for filter {kwargs}.")
            key = self.params.index[indices[0]]
        else:
            key = self.params.index[0]

        mean = self.params[['mean0', 'mean1', 'mean2']].loc[key].values
        std = self.params['std'].loc[key]

        res = scipy.optimize.minimize(
            lambda x: -log_likelihood(sphere_from_stereo(*x)**2, mean, std, np.asarray(obs)),
            x0=stereo_from_sphere(*np.ones(3)/np.sqrt(3)),
            method='BFGS',
        )

        out = sphere_from_stereo(*res.x)**2

        if isinstance(obs, pd.Series):
            out = pd.Series(out).rename_axis('pop')

        return out

    def classify(self, obs, prior, **kwargs):
        # Bayes classifier.
        raise NotImplementedError


def stereo_from_sphere(x, y, z):
    return x/(1+z), y/(1+z)


def sphere_from_stereo(X, Y):
    return np.array([2*X, 2*Y, 1-(X**2+Y**2)])/(1 + X**2 + Y**2)


def log_likelihood(pop: np.ndarray, mean: np.ndarray, std: float, y: np.ndarray) -> float:
    def pdf(y):
        return pop/(2*np.pi*std**2) @ np.exp(-0.5 * abs(y-mean[:,np.newaxis])**2/std**2)

    score = np.sum(np.log(np.clip(pdf(y), 1e-20/std**2, np.inf)))

    return score
