"""
Optimizers using other packages. Currently we support optimizers in scipy and optax.
"""

import numpy as np
from jax import numpy as jnp
from jax import grad, value_and_grad, random, jit
import optax

from tqdm.auto import tqdm
from scipy import optimize

from .. import models

__all__ = ["optax_optimizer", "scipy_optimzer", "scipy_L_BFGS_B"]


class optax_optimizer(object):
    def __init__(
        self, optax_optim=optax.sgd(lambda count: max(1e-5, 1 / (1 + count)))
    ) -> None:
        super().__init__()
        self.optax_optim = optax_optim

    def run(
        self,
        model,
        maxiter=1000,
        batchsize=100,
        PRNGkey=None,
        diff_mode="manual",
        tol=0.001,
        n_iter_no_change=100,
        verbose=True,
    ):
        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "auto"]

        if batchsize is None:
            batchsize = model.N
        else:
            if PRNGkey is None:
                if batchsize != model.N:
                    raise ValueError("jax random.PRNGkey should be specified")
                else:
                    print("Batch learning mode")
            assert batchsize <= model.N

        if type(model) is models.GWR_Ridge:
            x0 = jnp.concatenate(
                [jnp.array(model.kernel.params), jnp.array([model.penalty])]
            )
        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = jnp.array(model.kernel.params)
        else:
            raise ValueError("Unknown model class")

        x0 = model._to_unconstrained(x0)

        opt_state = self.optax_optim.init(x0)

        if diff_mode == "manual":

            def step(x, opt_state, idx):
                value = model.unconstrained_loss(x, idx)
                grads = model.unconstrained_grad(x, idx)
                updates, opt_state = self.optax_optim.update(grads, opt_state)
                x = optax.apply_updates(x, updates)

                return value, x, opt_state

        elif diff_mode == "auto":

            def step(x, opt_state, idx):
                value, grads = value_and_grad(model.unconstrained_loss, argnums=0)(
                    x, idx
                )

                updates, opt_state = self.optax_optim.update(grads, opt_state)
                x = optax.apply_updates(x, updates)

                return value, x, opt_state

        loss = []
        best_loss = jnp.inf
        count = 0
        x = x0
        with tqdm(total=int(maxiter), disable=not verbose) as pbar:
            for i in range(maxiter):
                pbar.update(1)
                if PRNGkey is None:
                    idx = None
                else:
                    key, PRNGkey = random.split(PRNGkey)
                    idx = random.choice(key, model.N, shape=(batchsize,), replace=True)
                value, x, opt_state = step(x, opt_state, idx)
                loss.append(value)
                pbar.set_description("loss = {:.5f}, raw_params = {}".format(value, x))

                # convergence check
                if value - best_loss < tol:
                    best_loss = float(min(loss))
                    count = 0
                else:
                    count += 1
                    if count >= n_iter_no_change:
                        self.converged = True
                        break
            else:
                self.converged = False

        model.set_params(x)

        return loss


class scipy_optimizer(object):
    def __init__(self) -> None:
        self.xla_jit = True

    def run(
        self, model, method=None, diff_mode="manual", tol=None, kwargs_minimize=dict()
    ):
        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "auto", "num"]

        if type(model) is models.GWR_Ridge:
            x0 = jnp.concatenate(
                [jnp.array(model.kernel.params), jnp.array([model.penalty])]
            )
        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = jnp.array(model.kernel.params)
        else:
            raise ValueError("Unknown model class")

        x0 = model._to_unconstrained(x0)

        def f(x):
            return model.unconstrained_loss(x)

        g = None
        if diff_mode == "manual":

            def g(x):
                return model.unconstrained_grad(x)

        elif diff_mode == "auto":

            def g(x):
                return grad(f)(x)

        if self.xla_jit:
            f = jit(f)
            if g is not None:
                g = jit(g)

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
                z = model._to_constrained(jnp.array([x]))
                return model.AICc([z])

        else:

            def f(x):
                return model.unconstrained_loss(jnp.array([x]))

        if self.xla_jit:
            f = jit(f)

        res = optimize.minimize_scalar(
            fun=f,
            bracket=bracket,
            bounds=bounds,
            method=method,
            tol=tol,
            options=options,
            **kwargs_minimize
        )

        model.set_params(jnp.array([res.x]))

        return res


class scipy_L_BFGS_B(object):
    """same setting to the 'scgwr' package in R
    see:  https://github.com/cran/scgwr/blob/master/R/scgwr.R
    """

    def __init__(self) -> None:
        self.xla_jit = True

    def run(self, model, diff_mode="auto", tol=None, kwargs_minimize=dict()):

        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "auto", "num"]

        if type(model) is models.GWR_Ridge:
            x0 = jnp.concatenate(
                [jnp.array(model.kernel.params), jnp.array([model.penalty])]
            )

            def f(x):
                return model.loocv_loss(x[:-1], x[-1])

        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = jnp.array(model.kernel.params)

            def f(x):
                return model.loocv_loss(x)

        else:
            raise ValueError("Unknown model class")

        if self.xla_jit:
            f = jit(f)

        def f64(x):
            return np.array(f(x), dtype=np.float64)

        g = None
        if diff_mode == "manual":
            if type(model) is models.GWR_Ridge:

                def g(x):
                    g1 = model.grad_params_loocv(x[:-1], x[-1])
                    g2 = model.grad_penalty_loocv(x[:-1], x[-1])

                    return jnp.concatnate([g1, jnp.array([g2])])

            elif type(model) is models.GWR or type(model) is models.ScaGWR:

                def g(x):
                    return model.grad_params_loocv(x)

        elif diff_mode == "auto":

            def g(x):
                return grad(f)(x)

        if self.xla_jit and g is not None:
            g = jit(g)

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
            **kwargs_minimize
        )

        model.set_params(res.x, transform=False)

        return res

