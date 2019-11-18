import numpy as np
import pandas as pd
import math
import scipy
import scipy.optimize
from numpy.polynomial.polynomial import polyval

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

def linear(x, a, b):
    return polyval(x, [a, b])

def polynomial2(x, a, b, c):
    return polyval(x, [a, b, c])

def polynomial3(x, a, b, c, d):
    return polyval(x, [a, b, c, d])

def polynomial4(x, a, b, c, d, e):
    return polyval(x, [a, b, c, d, e])

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

def quadratic(x, a, b, c):
    return a * x**2 + b*x + c

def zipf_mandelbrot(x, a, b, c):
    return c / (x + b) ** a

def inversed(x, a, b):
    return a / (x - b)

def square_root(x, a, b, c):
    return a * np.sqrt(x / c - b)

def pchip_spline(xs, ys):
    spliner = scipy.interpolate.PchipInterpolator(xs, ys)
    s_xs = np.arange(xs.min(), xs.max()+0.1, 0.1)
    s_ys = spliner(s_xs)
    return s_xs, s_ys

def fit_curve(fx, xs, ys, step=0.1):
    popt, _ = scipy.optimize.curve_fit(fx, xs, ys)
    s_xs = np.arange(xs.min(), xs.max()+step, step)
    s_ys = fx(s_xs, *popt)
    return s_xs, s_ys

fit_polynomial2 = lambda xs, ys: fit_curve(polynomial2, xs, ys)
fit_polynomial3 = lambda xs, ys: fit_curve(polynomial3, xs, ys)
fit_polynomial4 = lambda xs, ys: fit_curve(polynomial4, xs, ys)

def fit_curve_ravel(fx,xs, Y, step=0.1):

    assert len(Y.shape) == 2

    if xs is None:
        xs = np.arange(Y.shape[0])

    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    return fit_curve(fx, xsr, ysr, step=0.1)

rolling_average_methods = {
    'pandas':       lambda xs, ys, n: (xs, pd.Series(ys).rolling(window=n).mean().iloc[n-1:].values),
    'fftconvolve':  lambda xs, ys, n: (xs, scipy.signal.fftconvolve(ys, np.ones((n,))/n, mode='valid')),
    'reflect':      lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='reflect')),
    'nearest':      lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='nearest')),
    'mirror':       lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='mirror')),
    'constant':     lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='constant'))
}

def rolling_average_smoother(key, window_size):
    def smoother(xs, ys):
        xs, ys = rolling_average_methods[key](xs, ys, window_size)
        ys = ys / ys.sum()
        return xs, ys
    return smoother



