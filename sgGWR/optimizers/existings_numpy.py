"""
Optimizers using other packages. Currently we support optimizers in scipy.
"""

import numpy as np

from tqdm.auto import tqdm
from scipy import optimize

from .. import models

__all__ = ["scipy_optimzer", "scipy_L_BFGS_B"]


class scipy_optimizer(object):
    def __init__(self) -> None:
        pass

    def run(self, model, method=None, diff_mode="manual", tol=None, kwargs_minimize={}):
        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "num"]

        if type(model) is models.GWR_Ridge:
            x0 = np.concatenate(
                [np.array(model.kernel.params), np.array([model.penalty])]
            )
        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = np.array(model.kernel.params)
        else:
            raise ValueError("Unknown model class")

        x0 = model._to_unconstrained(x0)

        def f(x):
            return model.unconstrained_loss(x)

        g = None
        if diff_mode == "manual":

            def g(x):
                return model.unconstrained_grad(x)

        res = optimize.minimize(
            fun=f, x0=x0, method=method, jac=g, tol=tol, **kwargs_minimize
        )

        model.set_params(res.x)

        return res

    def run_scalar(
        self,
        model,
        method="brent",
        bracket=None,
        bounds=None,
        tol=None,
        options=None,
        aicc=False,
        kwargs_minimize=dict(),
    ):
        if aicc:

            def f(x):
                z = model._to_constrained(np.array([x]))
                return model.AICc([z])

        else:

            def f(x):
                return model.unconstrained_loss(np.array([x]))

        res = optimize.minimize_scalar(
            fun=f,
            bracket=bracket,
            bounds=bounds,
            method=method,
            tol=tol,
            options=options,
            **kwargs_minimize,
        )

        model.set_params(np.array([res.x]))

        return res


class scipy_L_BFGS_B(object):
    """same setting to the 'scgwr' package in R
    see:  https://github.com/cran/scgwr/blob/master/R/scgwr.R
    """

    def __init__(self) -> None:
        self.xla_jit = True

    def run(self, model, diff_mode="manual", tol=None, kwargs_minimize=dict()):

        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "num"]

        if type(model) is models.GWR_Ridge:
            x0 = np.concatenate(
                [np.array(model.kernel.params), np.array([model.penalty])]
            )

            def f(x):
                return model.loocv_loss(x[:-1], x[-1])

        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = np.array(model.kernel.params)

            def f(x):
                return model.loocv_loss(x)

        else:
            raise ValueError("Unknown model class")

        def f64(x):
            return np.array(f(x), dtype=np.float64)

        g = None
        if diff_mode == "manual":
            if type(model) is models.GWR_Ridge:

                def g(x):
                    g1 = model.grad_params_loocv(x[:-1], x[-1])
                    g2 = model.grad_penalty_loocv(x[:-1], x[-1])

                    return np.concatnate([g1, np.array([g2])])

            elif type(model) is models.GWR or type(model) is models.ScaGWR:

                def g(x):
                    return model.grad_params_loocv(x)

        if g is None:
            g64 = None
        else:

            def g64(x):
                return np.array(g(x)).astype(np.float64)

        res = optimize.minimize(
            fun=f64,
            x0=np.array(x0, dtype=np.float64),
            method="L-BFGS-B",
            jac=g64,
            bounds=optimize.Bounds(lb=np.zeros(len(x0)), ub=np.inf * np.ones(len(x0))),
            tol=tol,
            **kwargs_minimize,
        )

        model.set_params(res.x, transform=False)

        return res

