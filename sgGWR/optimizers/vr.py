"""
Optimizers using variance reduced stochastic gradients.
They are recommended if you require high-accuracy optimization.
"""
from jax import numpy as jnp
from jax import random, lax

from tqdm.auto import tqdm

from .sg import SGD


__all__ = ["SVRG", "KatyushaXs", "KatyushaXw"]


class SVRG(SGD):
    """
    Stochastic Variance Reduced Gradient with mini-batch

    refernces:
    Johnson, R., & Zhang, T. (2013).
    Accelerating Stochastic Gradient Descent using Predictive Variance Reduction.
    In Advances in Neural Information Processing Systems (Vol. 26).

    Allen-Zhu, Z. (2018).
    Katyusha X: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization.
    35th International Conference on Machine Learning, ICML 2018, 1, 284–290.
    """

    def __init__(self, learning_rate0=0.1, lam=0):
        super().__init__(learning_rate0=learning_rate0, lam=lam)

    def run(
        self,
        model,
        max_epoch=100,
        batchsize=None,
        PRNGkey=random.PRNGKey(seed=123),
        diff_mode="manual",
        tol=0.001,
        n_iter_no_change=5,
        min_epoch=5,
        # check_converge_exact=True,
        verbose=True,
        lax_scan=True,
    ):
        # maxiter = int(maxiter)
        max_epoch = int(max_epoch)
        if batchsize is None:
            batchsize = jnp.floor(jnp.sqrt(model.N)).astype(int)
        batchsize = int(batchsize)

        assert batchsize <= model.N

        if batchsize > jnp.sqrt(model.N):
            print(
                "Message: Effective batchsize is smaller than {:.1f} (=sqrt(N))".format(
                    jnp.sqrt(model.N)
                )
            )
        x0, [f, g, f_and_g] = self._init_optimizer(model, diff_mode)
        f_step = self._get_f_step(f, g)

        self.M = int(jnp.ceil(model.N / batchsize))
        # self.max_epoch = int(jnp.ceil(maxiter / self.M))
        # if self.max_epoch < min_epoch:
        #     raise ValueError(
        #         "maxiter is too small: it should be larger than {}(= min_epoch × N)".format(
        #             model.N * min_epoch
        #         )
        #     )

        # loss = [float(f(x0, idx=None))]
        loss = jnp.array([float(f(x0, idx=None))])
        self.exact_loss = []
        x = x0
        self._x0 = x0
        y = jnp.array(x)
        y_old = jnp.array(y)
        best_loss = float(loss[-1])
        count = 0
        # stop_flag = False
        t = 0

        with tqdm(total=max_epoch, disable=not verbose) as pbar:
            for epoch in range(max_epoch):
                x = self._momentum(self._x0, y, y_old, epoch)

                # SVRG^1ep
                self._x0 = jnp.array(x)
                l, self._grad_exact = f_and_g(x, idx=None)
                self.exact_loss.append(float(l))

                # convergence check
                if jnp.abs(best_loss - l) > tol:
                    best_loss = float(min(self.exact_loss))
                    count = 0
                elif epoch + 1 >= min_epoch:
                    count += 1
                    if count >= n_iter_no_change:
                        self.converged = True
                        break

                pbar.update(1)
                pbar.set_description(
                    "epoch {}/{}: loss={:.3f}".format(
                        epoch + 1, max_epoch, self.exact_loss[-1]
                    )
                )

                self.lr = self.lr_schedule(t + 1)
                t += 1
                if lax_scan:
                    key, PRNGkey = random.split(PRNGkey)
                    idxs = random.choice(key, model.N, shape=(self.M, batchsize))

                    x, ls = self.batch_step(f_step, x, idxs)
                    loss = jnp.concatenate([loss, ls])

                else:
                    for m in range(self.M):
                        pbar.set_description(
                            "epoch {}/{} ({}/{}): loss={}".format(
                                epoch + 1,
                                max_epoch,
                                m + 1,
                                self.M,
                                self.exact_loss[-1],
                            )
                        )

                        # self.lr = self.lr_schedule(t + 1)
                        key, PRNGkey = random.split(PRNGkey)
                        idx = random.choice(
                            key, model.N, shape=(batchsize,), replace=True
                        )
                        x, l = self.step(t + 1, x, f, g, f_and_g, idx)

                        loss = jnp.append(loss, l)

                # value to get momentum
                y_old = jnp.array(y)
                y = jnp.array(x)

            else:
                self.converged = False

        model.set_params(x)

        return loss

    def batch_step(self, f_step, x0, idxs):
        x, loss = lax.scan(f_step, init=x0, xs=idxs)
        return x, loss

    def _get_f_step(self, f, g):
        def f_step(carry, idx):
            return self.step(t=0, x=carry, f=f, g=g, f_and_g=None, idx=idx)

        return f_step

    def step(self, t, x, f, g, f_and_g, idx):
        grads = self._grad_exact + g(x, idx) - g(self._x0, idx)
        x_new = x - self.lr * grads
        loss = f(x, idx)
        return x_new, loss

    def _momentum(self, x, y, y_old, k):
        return y


class KatyushaXs(SVRG):
    """
    KatyushaXs algorithm

    refernces:
    Allen-Zhu, Z. (2018).
    Katyusha X: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization.
    35th International Conference on Machine Learning, ICML 2018, 1, 284–290.
    """

    def __init__(self, learning_rate0=0.1, lam=0, neg_moment=0.1):
        """
        If neg_moment=0.5, it is equivalent to SVRG
        neg_moment is refered as parameter tau in Allen-Zu (2018)
        """
        assert 0 < neg_moment <= 1
        self.neg_moment = float(neg_moment)
        super().__init__(learning_rate0=learning_rate0, lam=lam)

    def _momentum(self, x, y, y_old, k):
        return (1.5 * y + 0.5 * x - (1 - self.neg_moment) * y_old) / (
            1 + self.neg_moment
        )


class KatyushaXw(SVRG):
    """
    KatyushaXw algorithm

    refernces:
    Allen-Zhu, Z. (2018).
    Katyusha X: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization.
    35th International Conference on Machine Learning, ICML 2018, 1, 284–290.
    """

    def _momentum(self, x, y, y_old, k):
        return ((3 * k + 1) * y + (k + 1) * x - (2 * k - 2) * y_old) / (2 * k + 4)

