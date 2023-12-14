"""
Optimizers using stochastic gradients.
"""
from jax import numpy as jnp
from jax import grad, value_and_grad, random
from jax import jit, lax

from tqdm.auto import tqdm

from .. import models

__all__ = ["SGD", "ASGD", "SGDarmijo", "Adam", "Yogi"]


class SGD(object):
    """
    Stochastic Gradient Descent Algorithm

    reference:
    Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent.
    Proceedings of COMPSTAT 2010 - 19th International Conference on Computational Statistics,
    Keynote, Invited and Contributed Papers, 177–186. https://doi.org/10.1007/978-3-7908-2604-3_16
    """

    def __init__(self, learning_rate0=1.0, lam=1e-4):
        self.learning_rate0 = float(learning_rate0)
        self.lam = float(lam)

        self.xla_jit = True

    def lr_schedule(self, t):
        return self.learning_rate0 / (1 + self.lam * self.learning_rate0 * t)

    def run(
        self,
        model,
        maxiter=1000,
        batchsize=100,
        PRNGkey=random.PRNGKey(seed=123),
        diff_mode="manual",
        tol=0.001,
        n_iter_no_change=100,
        verbose=True,
    ):
        maxiter = int(maxiter)
        batchsize = int(batchsize)
        assert batchsize <= model.N

        self.batchsize = batchsize
        self.N = model.N

        x0, [f, g, f_and_g] = self._init_optimizer(model, diff_mode)

        key, PRNGkey = random.split(PRNGkey)
        idx = random.choice(key, model.N, shape=(batchsize,), replace=True)
        loss = [float(f(x0, idx=idx))]
        x = x0
        best_loss = float(loss[-1])
        count = 0
        self.lr_log = []
        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for t in range(maxiter):
                pbar.update(1)
                pbar.set_description("loss={}".format(loss[-1]))

                self.lr = self.lr_schedule(t + 1)
                key, PRNGkey = random.split(PRNGkey)
                idx = random.choice(key, model.N, shape=(batchsize,), replace=True)
                x, l = self.step(t + 1, x, f, g, f_and_g, idx)
                self.lr_log.append(float(self.lr))

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

    def _init_optimizer(self, model, diff_mode):
        diff_mode = diff_mode.lower()
        assert diff_mode in ["manual", "auto"]

        if type(model) is models.GWR_Ridge:
            x0 = jnp.concatenate(
                [jnp.array(model.kernel.params), jnp.array([model.penalty])]
            )

        elif type(model) is models.GWR or type(model) is models.ScaGWR:
            x0 = jnp.array(model.kernel.params)
        else:
            raise ValueError("Unknown model class")

        x0 = model._to_unconstrained(x0)

        def f(x, idx):
            return model.unconstrained_loss(x, idx)

        if diff_mode == "manual":

            def g(x, idx):
                return model.unconstrained_grad(x, idx)

            def f_and_g(x, idx):
                return (
                    model.unconstrained_loss(x, idx),
                    model.unconstrained_grad(x, idx),
                )

        elif diff_mode == "auto":

            def g(x, idx):
                return grad(model.unconstrained_loss, argnums=0)(x, idx)

            def f_and_g(x, idx):
                return value_and_grad(model.unconstrained_loss, argnums=0)(x, idx)

        if self.xla_jit:
            f, g, f_and_g = jit(f), jit(g), jit(f_and_g)

        return x0, [f, g, f_and_g]

    def step(self, t, x, f, g, f_and_g, idx):
        grads = g(x, idx)
        x_new = x - self.lr * grads
        loss = f(x, idx)
        return x_new, float(loss)


class ASGD(SGD):
    """
    Avereaged Stochastic Gradient Descent Algorithm

    reference:
    Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent.
    Proceedings of COMPSTAT 2010 - 19th International Conference on Computational Statistics,
    Keynote, Invited and Contributed Papers, 177–186. https://doi.org/10.1007/978-3-7908-2604-3_16
    """

    def __init__(self, learning_rate0=1.0, lam=1e-4):
        super().__init__(learning_rate0=learning_rate0, lam=lam)

    def lr_schedule(self, t):
        return self.learning_rate0 * (1 + self.lam * self.learning_rate0 * t) ** (-0.75)

    def step(self, t, x, f, g, f_and_g, idx):
        grads = g(self._x_sgd, idx)
        self._x_sgd = self._x_sgd - self.lr * grads
        x_new = (t * x + self._x_sgd) / (t + 1)
        loss = f(x_new, idx)
        return x_new, float(loss)

    def _init_optimizer(self, model, diff_mode):
        x0, [f, g, f_and_g] = super()._init_optimizer(model, diff_mode)
        self._x_sgd = jnp.array(x0)
        return x0, [f, g, f_and_g]


class SGDarmijo(SGD):
    """
    Stochastic Gradient Descent Algorithm with Armijo Line-search

    reference:
    Vaswani, S., Mishkin, A., Laradji, I., Schmidt, M., Gidel, G., & Lacoste-Julien, S. (2019).
    Painless stochastic gradient: Interpolation, line-search, and convergence rates.
    Advances in neural information processing systems, 32.
    """

    def __init__(
        self,
        learning_rate0=1.0,
        c=0.5,
        ls_decay=0.5,
        reset_decay=2.0,
        search_from_lr0=False,
    ):
        self.learning_rate0 = float(learning_rate0)

        assert c > 0.0
        self.c = float(c)

        assert 0.0 < ls_decay and ls_decay < 1.0
        self.ls_decay = float(ls_decay)

        assert reset_decay >= 1.0
        self.reset_decay = float(reset_decay)

        self.search_from_lr0 = search_from_lr0

        self.lr = float(learning_rate0)

        self._ls_max = 100

        self.xla_jit = True

    def lr_schedule(self, t):
        # set initial lr on line search
        if self.search_from_lr0:
            lr = self.learning_rate0
        else:
            lr = self.lr * self.reset_decay ** (self.batchsize / self.N)
            # print(f"lr0 = {lr}")
        return lr  # / self.ls_decay

    def step(self, t, x, f, g, f_and_g, idx):
        grads = g(x, idx)
        loss = f(x, idx)
        # line search
        self.lr = self.armijo_search(x, grads, self.lr, f=lambda x: f(x, idx))
        x_new = x - self.lr * grads
        return x_new, float(loss)

    def armijo_search(self, x, grads, lr, f):
        g2 = jnp.sum(jnp.square(grads))
        f0 = f(x)
        cond_fun = lambda lr: not self.armijo_cond(f, f0, x, grads, g2, lr)
        body_fun = lambda lr: lr * self.ls_decay
        # lr_searched = lax.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=lr)

        # print(f"start line search (lr={lr})")
        i = 0
        while cond_fun(lr) or i > self._ls_max:
            # print(f"lr={lr}")
            lr = body_fun(lr)
            i += 1

        lr_searched = lr

        return lr_searched

    def armijo_cond(self, f, f0, x, grads, g2, lr):
        x_new = x - lr * grads
        f_new = f(x_new)
        return f_new <= f0 - self.c * lr * g2


class Adam(SGD):
    def __init__(
        self, learning_rate0=1.0, b1=0.9, b2=0.999, eps=1e-8, correct_bias=True
    ):
        assert 0 <= b1 < 1
        self.b1 = float(b1)
        assert 0 <= b2 < 1
        self.b2 = float(b2)
        self.eps = float(eps)
        self.correct_bias = correct_bias
        super().__init__(learning_rate0, lam=0)

    def _init_optimizer(self, model, diff_mode):
        x0, [f, g, f_and_g] = super()._init_optimizer(model, diff_mode)
        self.m = jnp.zeros_like(x0)  # 1st moment
        self.v = jnp.zeros_like(x0)  # 2nd moment
        self.lr = float(self.learning_rate0)
        return x0, [f, g, f_and_g]

    def step(self, t, x, f, g, f_and_g, idx):
        grads = g(x, idx)
        g2 = jnp.square(grads)

        # update moments
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * g2

        # moments' bias-correction
        if self.correct_bias:
            m = self.m / (1 - self.b1 ** (t + 1))
            v = self.v / (1 - self.b2 ** (t + 1))
        else:
            m = self.m
            v = self.v

        x_new = x - self.lr * m / (jnp.sqrt(v) + self.eps)
        loss = f(x, idx)
        return x_new, float(loss)


class Yogi(Adam):
    def step(self, t, x, f, g, f_and_g, idx):
        grads = g(x, idx)
        g2 = jnp.square(grads)

        # update moments
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.v - (1 - self.b2) * jnp.sign(self.v - g2) * g2

        x_new = x - self.lr * self.m / (jnp.sqrt(self.v) + self.eps)
        loss = f(x, idx)
        return x_new, float(loss)

