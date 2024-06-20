try:
    from jax import numpy as jnp
    from jax import value_and_grad, vmap

    _JAX_AVAILABLE = True
except:
    import numpy as jnp

    print("No JAX mode: use numpy instead")

    def value_and_grad(f):
        raise ValueError("Autograd is not available: Install JAX")

    def vmap(f):
        def vf(idx):
            return jnp.stack(list(map(f, idx)), axis=0)

        return vf

    _JAX_AVAILABLE = False

from . import kernels

from scipy import stats
from tqdm.auto import tqdm
import copy


class GWR_Ridge(object):
    def __init__(self, y, X, sites, kernel=kernels.Gaussian([1.0]), penalty=0.01):
        self.y = jnp.array(y).reshape(-1, 1)
        self.N = len(self.y)
        self.X = jnp.array(X).reshape(self.N, -1)
        self.D = self.X.shape[1]
        self.sites = jnp.array(sites).reshape(self.N, -1)

        assert issubclass(kernel.__class__, kernels._baseKernel)
        self.kernel = kernel

        assert penalty >= 0
        self.penalty = penalty

    def get_beta(self, s):
        return self._get_beta(s, self.kernel.params, self.penalty, loocv=None)

    def _get_beta(self, s, params, penalty, loocv):
        w = self.kernel.forward(s, self.sites, params=params, loocv=loocv)
        P = self.X.T * w.reshape(1, -1)

        beta = jnp.linalg.solve(
            P @ self.X + penalty * jnp.eye(self.D), P @ self.y
        ).flatten()

        return beta

    def set_betas_inner(self):
        self.betas = vmap(self._set_beta_inner)(jnp.arange(self.N, dtype=int))

    def _set_beta_inner(self, i):
        return self._get_beta(
            self.sites[i], self.kernel.params, self.penalty, loocv=None
        )

    def _to_constrained(self, unconstrained):
        # return jnp.exp(unconstrained)
        return jnp.logaddexp(0, unconstrained)  # softplus function

    def _grad_to_constrained(self, unconstrained):
        # return jnp.exp(unconstrained)
        return 0.5 * (
            jnp.tanh(0.5 * unconstrained) + 1
        )  # sigmoid function = derivative of softplus function

    def _to_unconstrained(self, constrained):
        # return jnp.log(constrained)
        return constrained + jnp.log(
            1 - jnp.exp(-constrained)
        )  # inverse of softplus function

    def set_params(self, unconstrained, transform=True):
        if transform:
            z = self._to_constrained(unconstrained)
        else:
            z = unconstrained
        self.kernel.params = z[:-1]
        self.penalty = z[-1]

    def unconstrained_loss(self, x, idx=None):
        z = self._to_constrained(x)
        return self.loocv_loss(z[:-1], z[-1], idx)

    def unconstrained_grad(self, x, idx=None):
        z = self._to_constrained(x)
        gz = self._grad_to_constrained(x)
        g1 = self.grad_params_loocv(z[:-1], z[-1], idx) * gz[:-1]
        g2 = self.grad_penalty_loocv(z[:-1], z[-1], idx) * gz[-1]
        return jnp.concatenate([g1, jnp.array([g2])])

    def loocv_loss(self, params, penality, idx=None):
        def f(i):
            return self._loocv_loss(i, params, penality)

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        return jnp.mean(vmap(f)(idx))

    def _loocv_loss(self, i, params, penalty):
        beta = self._get_beta(self.sites[i], params, penalty, loocv=i)
        eps = self.y[i] - jnp.sum(beta * self.X[i])
        return eps**2

    def loocv_GN(self, params, penalty, idx=None):
        def f_g(i):
            return value_and_grad(self._loocv_pred, argnums=[1, 2])(
                i, jnp.array(params), penalty
            )

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        pred, J = vmap(f_g)(idx)  # idxが長すぎるとメモリが落ちる
        J = jnp.concatenate(
            [J[0].reshape((len(idx), len(params))), J[1][:, None]], axis=-1
        )

        eps = pred.reshape(-1, 1) - self.y[idx]
        loocv = jnp.mean(jnp.square(eps))
        g = jnp.mean(eps * J, axis=0)
        return loocv, g, J

    def unconstrained_GN(self, x, idx=None):
        z = self._to_constrained(x)
        gz = self._grad_to_constrained(x)
        loocv, g, J = self.loocv_GN(z[:-1], z[-1], idx=idx)
        g = g * gz
        J = J * gz[None]
        return loocv, g, J

    def _loocv_pred(self, i, params, penalty):
        beta = self._get_beta(self.sites[i], params, penalty, loocv=i)
        return jnp.sum(beta * self.X[i])

    def grad_params_loocv(self, params, penalty, idx=None):
        def f(i):
            return self._grad_params_loocv(i, params, penalty)

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        return jnp.mean(vmap(f)(idx), axis=0)

    def _grad_params_loocv(self, i, params, penalty):
        w = self.kernel.forward(self.sites[i], self.sites, params=params, loocv=i)
        dw = self.kernel.grad(self.sites[i], self.sites, params=params, loocv=i)

        P = self.X.T * w.reshape(1, -1)

        K = P @ self.X + penalty * jnp.eye(self.D)
        beta = jnp.linalg.solve(K, P @ self.y).flatten()
        eps = self.y - self.X @ beta[:, None]

        dP = self.X.T[None] * dw.T[:, None, :]
        C = dP @ eps[None]
        grad_y = self.X[i, None, None] @ jnp.linalg.solve(K[None], C)

        return -2 * grad_y.flatten() * eps[i]

    def grad_penalty_loocv(self, params, penalty, idx=None):
        def f(i):
            return self._grad_penalty_loocv(i, params, penalty)

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        return jnp.mean(vmap(f)(idx))

    def _grad_penalty_loocv(self, i, params, penalty):
        w = self.kernel.forward(self.sites[i], self.sites, params=params, loocv=i)

        P = self.X.T * w.reshape(1, -1)

        K = P @ self.X + penalty * jnp.eye(self.D)
        beta = jnp.linalg.solve(K, P @ self.y).flatten()

        grad_y = -self.X[i] @ jnp.linalg.solve(K, beta)

        eps = self.y[i, 0] - jnp.sum(self.X[i] * beta)

        return -2 * grad_y * eps

    def setInferenceStats(self, alpha=0.05):
        def func(i):
            C = self._getC(i)
            y = self.y.flatten()
            x = self.X[i]
            eps2 = jnp.square(y[i] - x @ C @ y)
            r = x @ C[:, i]
            diagCCt = jnp.diag(C @ C.T)
            beta = C @ y

            return eps2, r, diagCCt, beta

        eps2, r, diagCCt, self.betas = vmap(func)(jnp.arange(self.N, dtype=int))

        self.ENP = jnp.sum(r)
        self.RSS = jnp.sum(eps2)
        self.df = self.N - self.ENP

        # ML estimator (which is the same setting to mgwr package in Python)
        # It is consistent to the original paper of AICc
        # (Hurvich, Simonoff, Tsai 1998, J. R. Statist. Soc. B)
        self.sigma2_ML = self.RSS / self.N
        # unbiased estimator
        self.sigma2 = self.RSS / (self.N - self.ENP)

        self.aicc = self.N * (
            jnp.log(self.sigma2_ML)
            + jnp.log(2 * jnp.pi)
            + (self.N + self.ENP) / (self.N - 2 - self.ENP)
        )

        # local test statistics
        self.tvalues = self.betas / jnp.sqrt(diagCCt * self.sigma2)
        self.alpha = float(alpha)
        cdf = stats.t(self.df).cdf(self.tvalues)
        self.pvalues = 2 * jnp.where(self.tvalues < 0, cdf, 1 - cdf)

        # ENP should be 2tr[H] -tr[H'H] in original paper, but we follows mgwr implementation
        # ENP mentioned above is not implemented currently
        self.alpha_adj = self.alpha / self.ENP * self.D

        self.significant = self.pvalues <= self.alpha_adj

    def _getC(self, i):
        w = self.kernel.forward(
            self.sites[i], self.sites, params=self.kernel.params, loocv=None
        )
        P = self.X.T * w.reshape(1, -1)

        return jnp.linalg.solve(P @ self.X + self.penalty * jnp.eye(self.D), P)

    def AICc(self, params=None, sigma2_type=0):
        """Fast Evaluation of AICc

        reference.
        Li, Z., Fotheringham, A. S., Li, W., & Oshan, T. (2019).
        Fast Geographically Weighted Regression (FastGWR):
        a scalable algorithm to investigate spatial process heterogeneity
        in millions of observations.
        International Journal of Geographical Information Science, 33(1), 155–175.
        """
        if params is not None:
            self.kernel.params = params

        eps2, r = vmap(self._aicciter)(jnp.arange(self.N))
        enp = jnp.sum(r)
        rss = jnp.sum(eps2)

        if sigma2_type != 1:
            # ML estimator (which is the same setting to mgwr package in Python)
            # It is consistent to the original paper of AICc
            # (Hurvich, Simonoff, Tsai 1998, J. R. Statist. Soc. B)
            sigma2 = rss / self.N
        else:
            # unbiased estimator (Li et al. 2019)
            sigma2 = rss / (self.N - enp)

        aicc = self.N * (
            jnp.log(sigma2) + jnp.log(2 * jnp.pi) + (self.N + enp) / (self.N - 2 - enp)
        )
        return aicc

    def _aicciter(self, i):
        Cstar = self._getC(i)
        y = self.y.flatten()
        x = self.X[i]
        eps2 = jnp.square(y[i] - x @ Cstar @ y.flatten())
        r = x @ Cstar[:, i]

        return eps2, r

    def grad_params_aicc(self, params, penalty, idx=None, sigma2_type=0):
        def f(i):
            return self._grad_params_aicc(i, params, penalty)

        idx = jnp.arange(self.N, dtype=int)

        grad_y, grad_r, eps, r = vmap(f)(idx)

        enp = jnp.sum(r)
        denp = jnp.sum(grad_r, axis=0)

        if sigma2_type != 1:
            sigma2 = jnp.mean(jnp.square(eps))
            dsigma2 = -2 * jnp.sum(grad_y * eps[:, None], axis=0) / self.N
        else:
            sigma2 = jnp.sum(jnp.square(eps)) / (self.N - enp)
            dsigma2 = (denp * sigma2 - 2 * jnp.sum(grad_y * eps[:, None], axis=0)) / (
                self.N - enp
            )

        dAICc = (
            self.N / sigma2 * dsigma2
            + 2 * self.N * (self.N - 1) / jnp.square(self.N - 2 - enp) * denp
        )

        return dAICc

    def _grad_params_aicc(self, i, params, penalty):
        w = self.kernel.forward(self.sites[i], self.sites, params=params, loocv=None)
        dw = self.kernel.grad(self.sites[i], self.sites, params=params, loocv=None)

        P = self.X.T * w.reshape(1, -1)

        K = P @ self.X + penalty * jnp.eye(self.D)
        C = jnp.linalg.solve(K, P)
        r = self.X[i] @ C[:, i]
        dC = jnp.linalg.solve(
            K[None],
            self.X.T[None] * dw.T[:, None, :] @ (-self.X @ C + jnp.eye(self.N))[None],
        )
        beta = jnp.linalg.solve(K, P @ self.y).flatten()
        eps = self.y - self.X @ beta[:, None]

        dP = self.X.T[None] * dw.T[:, None, :]
        grad_y = self.X[i, None, :] @ jnp.linalg.solve(K[None], dP @ eps[None])

        grad_r = jnp.sum(self.X[i, None, :] * dC[..., i], axis=-1)

        return (grad_y.reshape((len(params),)), grad_r, eps.flatten()[i], r)

    def grad_penalty_aicc(self, params, penalty, idx=None):
        raise NotImplementedError()


class GWR(GWR_Ridge):
    def __init__(self, y, X, sites, kernel=kernels.Gaussian([1.0])):
        super().__init__(y, X, sites, kernel=kernel, penalty=0.0)

    def loocv_loss(self, params, idx=None):
        return super().loocv_loss(params, penality=self.penalty, idx=idx)

    def loocv_GN(self, params, idx=None):
        def f_g(i):
            return value_and_grad(self._loocv_pred, argnums=1)(
                i, jnp.array(params), penalty=self.penalty
            )

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        pred, J = vmap(f_g)(idx)
        J = J.reshape((len(idx), len(params)))

        eps = pred.reshape(-1, 1) - self.y[idx]
        loocv = jnp.mean(jnp.square(eps))
        g = jnp.mean(eps * J, axis=0)
        return loocv, g, J

    def unconstrained_GN(self, x, idx=None):
        z = self._to_constrained(x)
        gz = self._grad_to_constrained(x)
        loocv, g, J = self.loocv_GN(z, idx=idx)
        g = g * gz
        J = J * gz[None]
        return loocv, g, J

    def grad_params_loocv(self, params, idx=None):
        return super().grad_params_loocv(params, penalty=self.penalty, idx=idx)

    def grad_params_aicc(self, params, idx=None, sigma2_type=0):
        return super().grad_params_aicc(
            params, penalty=self.penalty, idx=idx, sigma2_type=sigma2_type
        )

    def unconstrained_loss(self, x, idx=None):
        z = self._to_constrained(x)
        return self.loocv_loss(z, idx)

    def unconstrained_grad(self, x, idx=None):
        z = self._to_constrained(x)
        gz = self._grad_to_constrained(x)
        return self.grad_params_loocv(z, idx) * gz

    def set_params(self, unconstrained, transform=True):
        if transform:
            z = self._to_constrained(unconstrained)
        else:
            z = unconstrained
        self.kernel.params = z


class ScaGWR(GWR):
    def __init__(self, y, X, sites, kernel, precompute=True):
        super().__init__(y, X, sites, kernel=kernel)

        assert type(precompute) is bool
        self.precompute = precompute

        self.M0 = self.X.T @ self.X
        self.m0 = self.X.T @ self.y
        # precomputation
        self._set_active_index()
        if self.precompute:
            if _JAX_AVAILABLE:
                self.m, self.M = vmap(self._getM)(jnp.arange(self.N, dtype=int))
                self.m, self.M = self.m.block_until_ready(), self.M.block_until_ready()
            else:
                self.m = jnp.empty((self.N, self.kernel.n_poly, self.D, 1))
                self.M = jnp.empty((self.N, self.kernel.n_poly, self.D, self.D))

                for i in range(self.N):
                    self.m[i], self.M[i] = self._getM(i)

    def _set_beta_inner(self, i):
        active_idx = jnp.append(self._active_idx[i], i)
        w = self.kernel.forward(
            self.sites[i], self.sites[active_idx], self.kernel.params, loocv=None
        )
        X = self.X[active_idx]
        P = X.T * w.reshape(1, -1)

        beta = jnp.linalg.solve(
            P @ X + self.penalty * jnp.eye(self.D), P @ self.y[active_idx]
        ).flatten()

        return beta

    # def get_beta(self, s):
    #     # knn search does not work with jax.vmap
    #     _, active_idx = self.kernel._knn(s, k=self.kernel.n_neighbour)
    #     w = self.kernel.forward(
    #         s, self.sites[active_idx], self.kernel.params, loocv=None
    #     )
    #     X = self.X[active_idx]
    #     P = X.T * w.reshape(1, -1)

    #     beta = jnp.linalg.solve(
    #         P @ X + self.penalty * jnp.eye(self.D), P @ self.y[active_idx]
    #     ).flatten()

    #     return beta

    def _getM(self, i):
        active_idx = self._active_idx[i]
        # ith weight is excluded from active_idx
        w = self.kernel.base_kernel(self.sites[i], self.sites[active_idx], loocv=None)

        p = jnp.arange(1, self.kernel.n_poly + 1, dtype=int)
        Wp = w[None] ** (4 / jnp.power(2, p))[:, None]  # (p,n)

        X = self.X[active_idx]  # (n,d)
        y = self.y[active_idx]  # (n,1)

        P = X.T[None] * Wp[:, None]  # (1,d,n) * (p,1,n) = (p,d,n)
        M = P @ X[None]  # (p,d,n) @ (1,n,d)
        m = P @ y[None]  # (p,d,n) @ (1,n,1)

        return m, M

    def _set_active_index(self):
        _, idx = self.kernel._knn(self.sites, k=self.kernel.n_neighbour)
        self._active_idx = jnp.array(idx[:, 1:])
        # d = jnp.linalg.norm(self.sites[None] - self.sites[:, None], axis=-1)
        # self._active_idx = jnp.argsort(d, axis=1)[:, 1 : self.kernel.n_neighbour]

    def _loocv_loss(self, i, params, penalty):
        beta = self._get_beta_scagwr(i, params)
        eps = self.y[i] - jnp.sum(beta * self.X[i])
        return eps**2

    def _loocv_pred(self, i, params, penalty):
        beta = self._get_beta_scagwr(i, params)
        return jnp.sum(beta * self.X[i])

    def _get_beta_scagwr(self, i, params):
        (r, R), _, _ = self._get_scagwr_stats(i, params)
        beta = jnp.linalg.solve(R, r)
        return beta.flatten()

    def grad_params_loocv(self, params, idx=None):
        def f(i):
            return self._grad_params_loocv(i, params)

        if idx is None:
            idx = jnp.arange(self.N, dtype=int)

        return jnp.mean(vmap(f)(idx), axis=0)

    def _grad_params_loocv(self, i, params):
        b = params[1:]

        (r, R), (m, M), (m0, M0) = self._get_scagwr_stats(i, params)
        beta = jnp.linalg.solve(R, r).flatten()
        eps = self.y[i] - self.X[i] @ beta

        Rinv = jnp.linalg.inv(R)

        # dbeta/da
        drda = m0.flatten()
        dRda = M0
        dbetada = Rinv @ (-dRda @ beta + drda)

        # dbeta/db
        if len(jnp.array(b)) == 1:
            p = jnp.arange(1, self.kernel.n_poly + 1, dtype=int)
            bp = p * jnp.power(b, p - 1)
            bp = bp[:, None, None]
            drdb = jnp.sum(bp * m, axis=0)[None]
            dRdb = jnp.sum(bp * M, axis=0)[None]
        else:
            drdb = m
            dRdb = M
        dbetadb = Rinv[None] @ (-dRdb @ beta[None, :, None] + drdb)

        grad_beta = jnp.concatenate([dbetada[None, :, None], dbetadb], axis=0)

        grad_y = self.X[i, None, None] @ grad_beta

        return -2 * grad_y.flatten() * eps

    def _get_scagwr_stats(self, i, params):
        a = params[0]
        b = params[1:]

        m0 = self.m0 - (self.X[i] * self.y[i]).reshape(self.m0.shape)
        M0 = self.M0 - self.X[i][:, None] @ self.X[i][None]

        if self.precompute:
            m = self.m[i]
            M = self.M[i]
        else:
            m, M = self._getM(i)

        if len(jnp.array(b)) == 1:
            b = jnp.power(b, jnp.arange(1, self.kernel.n_poly + 1, dtype=int))

        R = a * M0 + jnp.sum(b[:, None, None] * M, axis=0)
        r = a * m0 + jnp.sum(b[:, None, None] * m, axis=0)

        return (r, R), (m, M), (m0, M0)


class MGWR(GWR_Ridge):
    def __init__(
        self,
        y,
        X,
        sites,
        kernel=kernels.Gaussian([1]),
        base_class=GWR_Ridge,
        base_class_params=dict(),
    ):
        super().__init__(y, X, sites)

        if isinstance(kernel, kernels._baseKernel):
            self.kernel = [copy.deepcopy(kernel)] * self.D
        else:
            self.kernel = kernel
            if len(self.kernel) != self.D:
                raise ValueError("kernel should be the iterable object of length D")

        if base_class in (GWR_Ridge, GWR, ScaGWR):
            self.base_class = [base_class] * self.D
        else:
            self.base_class = base_class
            if len(self.base_class) != self.D:
                raise ValueError("base_class should be the iterable object of length D")

        self.base_class_params = base_class_params

    def backfitting(
        self, optimizers, maxiter=100, verbose=True, tol=1e-5, run_params=dict()
    ):

        if len(optimizers) != self.D:
            raise ValueError(
                "optimizers should be the list of optimizer for each exogenous variable"
            )

        # initialize with OLS
        self.betas = jnp.repeat(
            jnp.linalg.solve(a=self.X.T @ self.X, b=(self.X.T @ self.y).flatten())[
                jnp.newaxis
            ],
            self.N,
            axis=0,
        )

        self.pred = self.betas * self.X

        self.RSS = [jnp.sum(jnp.square(self.y - self.pred))]
        self.SOC = [jnp.inf]

        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for i in range(maxiter):
                pbar.update(1)
                pbar.set_description(
                    "RSS = {:.2f}, SOC = {:.2f}%".format(
                        self.RSS[-1], 100 * self.SOC[-1]
                    )
                )

                for d in range(self.D):
                    # local fit
                    resid = self.y - (
                        self.pred.sum(axis=1) - (self.betas[:, d] * self.X[:, d])
                    ).reshape(self.N, 1)
                    localmodel = self.base_class[d](
                        y=resid,
                        X=self.X[:, d].reshape(self.N, 1),
                        sites=self.sites,
                        kernel=self.kernel[d],
                        **self.base_class_params,
                    )

                    optim = optimizers[d]
                    optim.run(localmodel, **run_params)
                    # print(localmodel.kernel.params)
                    self.kernel[d] = copy.deepcopy(localmodel.kernel)

                    # update coefficients
                    localmodel.set_betas_inner()
                    if _JAX_AVAILABLE:
                        self.betas = self.betas.at[:, d].set(localmodel.betas[:, d])
                        self.pred = self.pred.at[:, d].set(
                            self.betas[:, d] * self.X[:, d]
                        )
                    else:
                        self.betas[:, d] = localmodel.betas[:, d]
                        self.pred[:, d] = self.betas[:, d] * self.X[:, d]

                # score of change
                self.RSS.append(jnp.sum(jnp.square(self.y - self.pred)))
                self.SOC.append(jnp.abs(self.RSS[-1] - self.RSS[-2]) / self.RSS[-1])

                if self.SOC[-1] < tol:
                    break

        return self.RSS
