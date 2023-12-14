"""
Optimizers using stochastic gradients with second order informations.
Faster convergence can be expected.
"""
from jax import numpy as jnp
from jax import random

from tqdm.auto import tqdm

from .. import models
from .sg import SGD

__all__ = ["SGN", "SGN_BFGS", "SGN_LM"]


class SGN(SGD):
    """
    Stochastic Gauss-Newton methods.
    Small value is added to approximated Hessian to guarantee its positive definitness.

    refernces:
    Bottou, L., Curtis, F. E., & Nocedal, J. (2018).
    Optimization methods for large-scale machine learning.
    In SIAM Review (Vol. 60, Issue 2, pp. 223–311).
    """

    def __init__(self, learning_rate0=1.0, lam=0):
        super().__init__(learning_rate0=learning_rate0, lam=lam)

        self._delta = 1e-3

    def run(
        self,
        model,
        maxiter=1000,
        batchsize=100,
        PRNGkey=random.PRNGKey(seed=123),
        diff_mode="manual",  # ignored
        tol=0.001,
        n_iter_no_change=100,
        verbose=True,
    ):

        maxiter = int(maxiter)
        batchsize = int(batchsize)
        assert batchsize <= model.N
        x0, f_g_J = self._init_optimizer(model)

        key, PRNGkey = random.split(PRNGkey)
        idx = random.choice(key, model.N, shape=(batchsize,), replace=True)
        l, g, J = f_g_J(x0, idx=idx)
        loss = [float(l)]
        x = x0
        best_loss = float(loss[-1])
        count = 0
        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for t in range(maxiter):
                pbar.update(1)
                pbar.set_description("loss={}".format(loss[-1]))

                self.lr = self.lr_schedule(t + 1)
                key, PRNGkey = random.split(PRNGkey)
                idx = random.choice(key, model.N, shape=(batchsize,), replace=True)
                x, l = self.step(t + 1, x, f_g_J, idx)

                loss.append(l)

                # convergence check
                if l - best_loss < tol:
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

    def _init_optimizer(self, model):
        if type(model) is models.GWR_Ridge:
            x0 = jnp.concatenate(
                [jnp.array(model.kernel.params), jnp.array([model.penalty])]
            )

        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = jnp.array(model.kernel.params)
        else:
            raise ValueError("Unknown model class")

        x0 = model._to_unconstrained(x0)

        def f_g_J(x, idx):
            return model.unconstrained_GN(x, idx)

        self._dim = len(x0)
        self.H = 2 * jnp.eye(1)
        self.Hinv = 1 / self.H

        return x0, f_g_J

    def step(self, t, x, f_g_J, idx):
        loss, g, J = f_g_J(x, idx)
        p = self._direction(g, J)
        x_new = x + self.lr * p
        return x_new, float(loss)

    def _direction(self, g, J):
        H = jnp.repeat(self.H, len(J)).reshape((-1, 1, 1))
        H_approx = jnp.mean(J[..., None] @ H @ J[:, None, :], axis=0)
        # direct inversion in not stable
        p = jnp.linalg.solve(H_approx + self._delta * jnp.eye(self._dim), -g)
        return p


class SGN_BFGS(SGN):
    """
    Stochastic Gauss-Newton methods.
    BFGS formula is applied to guarantee the positive definiteness of approximated Hessian matrix.
    Note: I recommend small learning rate

    refernces:
    Bottou, L., Curtis, F. E., & Nocedal, J. (2018).
    Optimization methods for large-scale machine learning.
    In SIAM Review (Vol. 60, Issue 2, pp. 223–311).
    """

    def _init_optimizer(self, model):
        ret = super()._init_optimizer(model)
        # initalize BFGS formula
        self._Ginv = jnp.eye(self._dim)

        return ret

    def step(self, t, x, f_g_J, idx):
        loss, g, J = f_g_J(x, idx)
        p = self._direction(g, J)
        s = self.lr * p
        x_new = x + s
        self._bfgs_update(s, J)
        return x_new, float(loss)

    def _direction(self, g, J):
        p = -self._Ginv @ g
        return p

    def _bfgs_update(self, s, J):
        H = jnp.repeat(self.H, len(J)).reshape((-1, 1, 1))
        H_approx = jnp.mean(J[..., None] @ H @ J[:, None, :], axis=0)
        # direct inversion in not stable
        # p = jnp.linalg.solve(H_approx, -g)
        v = H_approx @ s

        # BFGS formula
        I = jnp.eye(self._dim)
        sv = jnp.sum(v * s)
        a = I - v[:, None] @ s[None] / sv
        self._Ginv = a.T @ self._Ginv @ a + s[:, None] @ s[None] / sv


class SGN_LM(SGN):
    """
    Stochastic Gauss-Newton methods with damping of Levenberg-Marquardt (LM) method.
    The damping improves the stability.
    This algorithm refered to as "SMW-GN" in Ren and Goldfarb (2019)
    because they used Sherman-Morrison-Woodbury(SMW) formula.
    SMW formula is not used in this implementation
    because we assume that the number of parameters is not far larger than the mini-batch size.

    refernces:
    Ren, Y., & Goldfarb, D. (2019).
    Efficient Subsampled Gauss-Newton and Natural Gradient Methods for Training Neural Networks.
    https://arxiv.org/abs/1906.02353v1
    """

    def __init__(
        self,
        learning_rate0=1.0,
        lam=0,
        lam_LM0=1.0,
        boost=1.01,
        drop=0.99,
        eps=0.25,
        tau=0.001,
    ):
        super().__init__(learning_rate0=learning_rate0, lam=lam)

        assert tau <= lam_LM0
        assert drop < 1 < boost
        assert 0 < eps < 0.5
        assert 0 < tau

        self.lam_LM0 = float(lam_LM0)
        self.boost = float(boost)
        self.drop = float(drop)
        self.eps = float(eps)
        self.tau = float(tau)

        self.lam_LM = float(self.lam_LM0)
        self._lam_LM = self.lam_LM0 - self.tau

    def step(self, t, x, f_g_J, idx):
        loss, g, J = f_g_J(x, idx)
        self._prev_loss = float(loss)
        if t > 1:
            self._LM(loss, g)
        p = self._direction(g, J)
        self._prev_p = jnp.array(p)
        x_new = x + self.lr * p
        return x_new, float(loss)

    def _direction(self, g, J):
        N2 = len(J)
        Hinv = jnp.repeat(self.Hinv, N2).reshape(-1, 1, 1)
        self.B = jnp.mean(J[..., None] @ Hinv @ J[:, None, :], axis=0)
        p = -jnp.linalg.solve(self.B + self.lam_LM + jnp.eye(self._dim), g)
        return p

    def _LM(self, loss, g):
        def m(p):
            return loss + jnp.sum(g * p) + 0.5 * p @ self.B @ p

        rho = (loss - self._prev_loss) / (loss - m(self._prev_p))

        if rho < self.eps:
            self._lam_LM = self._lam_LM * self.boost
        elif 1 - self.eps < rho:
            self._lam_LM = self._lam_LM * self.drop

        self.lam_LM = self.lam_LM + self.tau
