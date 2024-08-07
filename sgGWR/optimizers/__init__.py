from .. import models, kernels

try:
    from .existings import optax_optimizer, scipy_optimizer, scipy_L_BFGS_B
    from .sg import SGD, ASGD, SGDarmijo, Adam, Yogi
    from .vr import SVRG, KatyushaXs, KatyushaXw
    from .second import SGN, SGN_BFGS, SGN_LM
    from .golden import golden_section

    __all__ = [
        "optax_optimizer",
        "scipy_optimzer",
        "scipy_L_BFGS_B",
        "SGD",
        "ASGD",
        "SGDarmijo",
        "Adam",
        "Yogi",
        "SVRG",
        "KatyushaXs",
        "KatyushaXw",
        "SGN",
        "SGN_BFGS",
        "SGN_LM",
        "golden_section",
    ]

except:
    from .existings_numpy import scipy_optimizer, scipy_L_BFGS_B
    from .sg_numpy import SGD, ASGD, SGDarmijo
    from .golden import golden_section

    print(
        "JAX is not available. Installing JAX is strongly recommended for performance."
    )

    __all__ = [
        "SGD",
        "ASGD",
        "SGDarmijo",
        "scipy_optimzer",
        "scipy_L_BFGS_B",
        "golden_section",
    ]
