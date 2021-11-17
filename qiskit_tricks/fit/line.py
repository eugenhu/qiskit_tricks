from typing import NamedTuple
from enum import IntEnum, auto

import numpy as np
import scipy.optimize

from .util import rotation_mat2d

__all__ = ['LineFitResult', 'line_fit']


DELTA_TOL     = 1.e-8
GRADIENT_TOL  = 1.e-8
OBJECTIVE_TOL = 1.e-8


class LineFitResult(NamedTuple):
    angle: float
    rho: float
    objective: float
    residuals: np.ndarray
    projection: np.ndarray


class LineParam(IntEnum):
    ANGLE = 0
    RHO   = auto()


class LineModel:
    def __init__(self, data: np.ndarray) -> None:
        if data.flags.writeable:
            data = data.copy()
            data.flags.writeable = False

        self.data = data

        self._params = np.empty(len(LineParam))
        self._params_set = False
        self._residuals = np.empty(shape=(self.data.shape[1],))
        self._jac = np.empty(shape=(self.data.shape[1], len(self._params)))

        self._projection = np.empty(shape=(self.data.shape[1],))

    def set_params(self, params: np.ndarray) -> None:
        if self._params_set and (self._params == params).all():
            return

        q   = params[LineParam.ANGLE]
        rho = params[LineParam.RHO]

        e       = self._residuals
        de_dq   = self._jac[:, LineParam.ANGLE]
        de_drho = self._jac[:, LineParam.RHO]

        Q = rotation_mat2d(q)
        r, z = Q.T @ self.data - [[0], [rho]]

        e[:] = z
        de_dq[:] = -r
        de_drho[:] = -1

        self._params[:] = params
        self._projection[:] = r

    @property
    def params(self) -> np.ndarray:
        params = self._params[:]
        params.flags.writeable = False
        return params

    @property
    def dof(self) -> int:
        return self.data.shape[1] - len(self.params) + 1

    @property
    def jac(self) -> np.ndarray:
        jac = self._jac[:]
        jac.flags.writeable = False
        return jac

    @property
    def residuals(self) -> np.ndarray:
        residuals = self._residuals[:]
        residuals.flags.writeable = False
        return residuals

    @property
    def projection(self) -> np.ndarray:
        projection = self._projection[:]
        projection.flags.writeable = False
        return projection


def line_fit(
        data: np.ndarray,
        *,
        loss: str = 'linear',
        f_scale: float = 1.0,
        verbose: bool = False,
) -> LineFitResult:
    if data.shape[1] < 2:
        raise ValueError("Need at least two points to do a line fit.")

    model = LineModel(data)

    def fun(params: np.ndarray) -> np.ndarray:
        model.set_params(params)
        residuals = model.residuals.copy()
        return residuals

    def jac(params: np.ndarray) -> np.ndarray:
        model.set_params(params)
        jac = model.jac.copy()
        return jac

    model.set_params(line_guess(data))

    try:
        optimize_result = scipy.optimize.least_squares(
            fun,
            model.params,
            jac,
            method='lm' if loss == 'linear' else 'trf',
            loss=loss,
            f_scale=f_scale,
            x_scale='jac',
            ftol=OBJECTIVE_TOL,
            xtol=DELTA_TOL,
            gtol=GRADIENT_TOL,
            max_nfev=50,
            verbose=2 if verbose else 0,
        )
    except ValueError:
        raise

    # Update model parameters to final result.
    model.set_params(optimize_result.x)

    result = LineFitResult(
        angle=model.params[LineParam.ANGLE],
        rho=model.params[LineParam.RHO],

        objective=(model.residuals**2).sum()/model.dof,
        residuals=model.residuals,
        projection=model.projection,
    )

    return result


def line_guess(data: np.ndarray) -> np.ndarray:
    """A very crude guess that just fits a line between two arbitrary points."""
    params = np.empty(len(LineParam))

    pt0 = data[:,  0]
    pt1 = data[:, -1]

    if np.all(pt0 == pt1):
        # Just initialize to anything.
        params[LineParam.RHO] = 0
        params[LineParam.ANGLE] = 0
    else:
        unit = (pt1 - pt0)/np.linalg.norm(pt1 - pt0)
        perp = np.array([-unit[1], unit[0]])

        params[LineParam.RHO] = perp @ pt0
        params[LineParam.ANGLE] = np.arctan2(unit[1], unit[0])

    return params
