from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypeVar, Union, cast

from injector import Inject
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.providers import JobV1 as Job
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Result
from qiskit_experiments.calibration_management import BackendCalibrations

from qiskit_tricks.result import resultdf
from qiskit_tricks.transform import parallelize_circuits


__all__ = [
    'Experiment',
    'Analysis',
]


class ExperimentMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        if 'tag' not in namespace:
            namespace['tag'] = name
        cls = super().__new__(mcls, name, bases, namespace)
        return cls


class Experiment(ABC, metaclass=ExperimentMeta):
    tag: str
    parameter_names: Sequence[str] = ()

    backend: Backend
    calibrations: BackendCalibrations

    _parallelize: Optional[list] = None

    _Self = TypeVar('_Self', bound='Experiment')

    def __init__(
            self,
            backend: Inject[Backend],
            calibrations: Inject[Optional[BackendCalibrations]] = None,
    ) -> None:
        self.backend = backend
        self.calibrations = calibrations or BackendCalibrations(backend)

    def configure(self: _Self, *args, **kwargs) -> _Self:
        tmp = []
        for args_i, kwargs_i in self._broadcast_args(*args, **kwargs):
            params = self.generate_parameters(*args_i, **kwargs_i)
            if isinstance(params, pd.DataFrame):
                pass
            elif isinstance(params, tuple):
                params = pd.DataFrame([params], columns=self.parameter_names)
            elif isinstance(params, Iterable):
                params = pd.DataFrame(params, columns=self.parameter_names)
            else:
                raise TypeError
            tmp.append(params)
        self.parameters_table = pd.concat(tmp, ignore_index=True)
        return self

    @staticmethod
    def _broadcast_args(*args, **kwargs):
        all_args = (*args, *kwargs.values())
        b = np.broadcast(*(np.array(x, dtype=object) for x in all_args))
        for x in b:
            args_i = x[:len(args)]
            kwargs_i = dict(zip(kwargs.keys(), x[len(args):]))
            yield args_i, kwargs_i

    def generate_parameters(self, *args, **params) -> Union[Iterable, pd.DataFrame]:
        parameter_names = self.parameter_names

        for i, x in enumerate(args):
            if i >= len(parameter_names):
                raise TypeError("Too many arguments.")
            name = parameter_names[i]
            if name in params:
                raise TypeError(f"Multiple values for parameter '{name}'")
            params[name] = x
        del args

        params = {**dict.fromkeys(parameter_names), **params}
        if set(params.keys()) - set(parameter_names):
            extra_key = next(iter(set(params.keys()) - set(parameter_names)))
            raise TypeError(f"Got unexpected parameter '{extra_key}'")

        return pd.DataFrame(params, index=[0], dtype=object).reindex(columns=self.parameter_names)

    def parallelize(self, coupling_map: Union[list, bool] = True):
        if coupling_map is False:
            self._parallelize = None
            return self

        if coupling_map is True:
            coupling_map = self.backend.configuration().coupling_map

        self._parallelize = cast(list, coupling_map)

        return self

    def run(self, **run_config) -> Job:
        circuits = self.circuits()
        run_config = self.run_config(**run_config)
        job = self.backend.run(circuits, **run_config)

        # Backend.run() is untyped...
        job = cast(Job, job)

        print(f"Sent {len(circuits)} circuits to {self.backend.name()}")
        print(f"Job ID: {job.job_id()}")
        print(f"  Tags: {run_config['job_tags']}")

        return job

    def run_config(self, **override) -> Dict[str, Any]:
        run_config: Dict[str, Any] = {}

        run_config['qubit_lo_freq'] = self.calibrations.get_qubit_frequencies()
        run_config['meas_lo_freq'] = self.calibrations.get_meas_frequencies()

        if isinstance(self.default_run_config, dict):
            run_config.update(self.default_run_config)
        else:
            run_config.update(self.default_run_config())

        run_config.update(override)

        run_config['job_tags'] = list(run_config.get('job_tags', []))
        run_config['job_tags'].append(self.tag)

        return run_config

    def circuits(self) -> List[QuantumCircuit]:
        qreg = QuantumRegister(self.backend.configuration().n_qubits)
        circuits: List[QuantumCircuit] = []

        for params in self.parameters_table.to_dict(orient="records"):
            circ = QuantumCircuit(qreg, metadata=params)
            self.build(circ, **params)
            circuits.append(circ)

        if self._parallelize is not None:
            circuits = parallelize_circuits(circuits, self._parallelize)

        return circuits

    def default_run_config(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def build(self, circuit: QuantumCircuit, **kwargs) -> None:
        ...


class Analysis:
    dont_groupby: Sequence[str] = ()

    _Self = TypeVar('_Self', bound='Analysis')

    source: Union[pd.Series, pd.DataFrame]

    def __init__(self, source: Union[Job, Result, pd.Series, pd.DataFrame], **kwargs) -> None:
        if isinstance(source, (Job, Result)):
            source = resultdf(source)
        else:
            source = source.copy()
        assert isinstance(source, (pd.Series, pd.DataFrame))

        tables = {'source': source}

        if source.index.names.difference(self.dont_groupby):
            self.index = index = source.index.droplevel(
                [x for x in self.dont_groupby if x in source.index.names]
            ).unique()
            groupby = source.groupby(index.names)
            tables_parts = defaultdict(list)
            for _, group in groupby:
                group = group.droplevel(index.names)
                new_tables = self.create_tables(group, **kwargs) or {}
                for k, v in new_tables.items():
                    tables_parts[k].append(v)
            for k, v in tables_parts.items():
                table = pd.concat(v, keys=index, names=index.names)
                if isinstance(v[0], pd.Series):
                    # Heuristic to determine if series levels should be unstacked into a dataframe.
                    if v[0].index.names[0] == None:
                        table = table.unstack(v[0].index.names).infer_objects()
                tables[k] = table
        else:
            self.index = None
            tables.update(self.create_tables(source, **kwargs) or {})
            self._tables = list(tables.keys())

        self._tables = list(tables.keys())
        for k, v in tables.items():
            setattr(self, k, v)

    def create_tables(
            self,
            data: Union[pd.Series, pd.DataFrame],
            **kwargs
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        return {}

    def first(self: _Self) -> _Self:
        return self.nth(0)

    def nth(self: _Self, n: int) -> _Self:
        if self.index is None and n == 0:
            return self

        return self[[n]]

    def xs(self: _Self, key: Any = None, *, drop_level=False, **kwargs) -> _Self:
        assert self.index is not None

        if key is not None:
            new_self = self[[self.index.get_loc(key)]]
        else:
            mask = self.index.to_frame()[kwargs.keys()].agg(lambda x: x.to_dict() == kwargs, axis=1)
            new_self = self[mask]

        assert new_self.index is not None

        if drop_level:
            if self.index.names.difference(kwargs.keys()):
                new_self.index = new_self.index.droplevel(list(kwargs.keys()))
            else:
                new_self.index = None

            for k in new_self._tables:
                table = getattr(new_self, k)
                assert table is not None
                if table.index.names.difference(kwargs.keys()):
                    table = table.droplevel(list(kwargs.keys()))
                else:
                    if not table.empty:
                        table = table.iloc[0]
                    else:
                        table = None
                setattr(new_self, k, table)

        return new_self

    def __getitem__(self: _Self, x: Any) -> _Self:
        assert self.index is not None

        if isinstance(x, int):
            new_self = self[[x]]
            new_self.index = None
            for k in self._tables:
                table = getattr(new_self, k)
                assert table is not None
                if table.index.names.difference(self.index.names):
                    table = table.droplevel(self.index.names)
                else:
                    if not table.empty:
                        table = table.iloc[0]
                    else:
                        table = None
                setattr(new_self, k, table)
            return new_self

        new_self = self._new_like()
        new_self.index = new_index = self.index[x]
        new_tables = defaultdict(list)

        if new_index.empty:
            for name in self._tables:
                table = getattr(self, name)
                assert table is not None
                setattr(new_self, name, table[0:0])
            return new_self

        for key in new_index:
            if isinstance(new_index, pd.MultiIndex):
                assert isinstance(key, tuple)
                for name in self._tables:
                    table = getattr(self, name)
                    assert table is not None
                    try:
                        new_tables[name].append(table.xs(
                            key,
                            level=new_index.names,
                            drop_level=False,
                        ))
                    except KeyError:
                        new_tables[name].append(table.iloc[0:0])
            else:
                for name in self._tables:
                    table = getattr(self, name)
                    assert table is not None
                    if isinstance(table.index, pd.MultiIndex):
                        try:
                            new_tables[name].append(table.xs(
                                key,
                                level=new_index.name,
                                drop_level=False,
                            ))
                        except KeyError:
                            new_tables[name].append(table.iloc[0:0])
                    else:
                        if key in table.index:
                            new_tables[name].append(table.loc[[key]])
                        else:
                            new_tables[name].append(table.iloc[0:0])

        for k, v in new_tables.items():
            setattr(new_self, k, pd.concat(v))

        return new_self

    def _new_like(self: _Self) -> _Self:
        new_self = copy.copy(self)
        new_self.index = None
        for k in self._tables:
            setattr(new_self, k, None)
        return new_self

    def caption(self: _Self, loc='lower right') -> _Self:
        """Add a descriptive caption to the current axes."""
        if self.index is None:
            return self

        # Caption is generated from first index.
        props = dict(zip(self.index.names, self.index[0]))

        if None in props:
            del props[None]

        if not props:
            return self

        caption = '\n'.join(map('{0[0]} = {0[1]!r}'.format, props.items()))

        at = AnchoredText(
            caption,
            prop=dict(fontfamily='monospace', alpha=0.5),
            frameon=True,
            loc=loc,
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_alpha(0.5)
        plt.gca().add_artist(at)

        return self
