from typing import NamedTuple

import numpy as np


__all__ = ['CosineFitResult', 'cosine_fit']


class CosineFitResult(NamedTuple):
    freq: float
    origin: float
    amp: float
    mean: float
        
    freq_err: float
    origin_err: float
    amp_err: float
    mean_err: float

    calculated: np.ndarray
    reduced_chisq: float
    
    @property
    def wavelength(self) -> float:
        if self.freq == 0:
            return np.inf
        
        return 1/self.freq
    
    @property
    def wavelength_err(self) -> float:
        if self.freq == 0:
            return np.inf
        
        return self.freq_err/self.freq**2


def cosine_fit(x, y):
    from numpy import pi
    from scipy.optimize import curve_fit
    
    x = np.ravel(x)
    y = np.ravel(y)
    
    dx = np.quantile(np.diff(x[np.argsort(x)]), 0.05)
    L = np.ptp(x)

    k = 1/L * np.arange(np.clip(int(0.5*L/dx) if dx > 0 else np.inf, 1, 10000))
    phi = np.linspace(-pi, pi)
    
    mean = y.mean()
    Y = (y-mean) @ np.exp(2j*pi*k*x[:, None])

    freq = k[np.argmax(abs(Y))]
    phase = phi[np.argmax((y-mean) @ np.cos(2*pi*freq*x[:, np.newaxis] + phi))]
    amp = np.ptp(y)/2
    
    if freq != 0:
        origin = -phase/(2*pi*freq)
    else:
        origin = 0.0
    
    def cosine(x, *params):
        freq, origin, amp, mean = params
        return mean + amp * np.cos(2*pi*freq*(x - origin))

    popt, pcov = curve_fit(
        cosine,
        x,
        y,
        p0=(freq, origin, amp, mean),
        method='trf',
        x_scale='jac',
    )
    
    calc = cosine(x, *popt)
    
    if x.size > popt.size:
        reduced_chisq = np.sum((y - calc)**2)/(x.size - popt.size)
    else:
        reduced_chisq = np.nan
    
    return CosineFitResult(
        *popt,
        *np.sqrt(np.diag(pcov)),
        calc,
        reduced_chisq,
    )
