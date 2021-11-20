from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypeVar, Union, cast

from injector import Inject
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.providers import JobV1 as Job
from qiskit.providers.backend import BackendV1 as Backend
from qiskit_experiments.calibration_management import BackendCalibrations

from qiskit_tricks import parallelize_circuits


__all__ = ['Experiment']


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
            if isinstance(params, tuple):
                params = pd.DataFrame([params], columns=self.parameter_names)
            elif isinstance(params, Iterable):
                params = pd.DataFrame(params, columns=self.parameter_names)
            else:
                assert isinstance(params, pd.DataFrame)
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

    def generate_parameters(self, *args, **params) -> Optional[dict]:
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
