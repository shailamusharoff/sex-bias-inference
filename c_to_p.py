"""
Convert between the fitted sex-bias parameter c and the proportion of females p.

The X models (three_epoch_X0/X1/X2 in dadiLrtFunctions.py) are parameterized by
c, the ratio of the X-chromosomal to the autosomal effective population size:

    c = NeX / NeA

From Eqs. 1 and 2 of the paper, for a proportion of females p in a given epoch:

    NeA = 4 * p * (1-p) * N                  (simBottlenecks_functions.fA)
    NeX = 9 * p * (1-p) / (2 * (2-p)) * N    (simBottlenecks_functions.fX)

The p*(1-p)*N factor cancels in the ratio, leaving

    c = 9 / (8 * (2-p))        and        p = 2 - 9 / (8*c)

so c alone determines p; no knowledge of N is needed. With no sex bias,
p = 0.5 and c = 0.75.

Note that c is bounded even though it is fit as a free parameter: p in [0, 1]
corresponds to c in [9/16, 9/8] = [0.5625, 1.125]. An optimizer can return a c
outside that range, which means no proportion of females explains the fit --
in practice a sign that the X fit did not converge. c_to_p() raises ValueError
in that case rather than returning a nonsensical p.
"""

# Bounds on c implied by p in [0, 1]. c is increasing in p, so:
#   p = 0 -> c = 9/16,  p = 1 -> c = 9/8.
# The endpoints are limiting cases: at p = 0 or p = 1 one sex is absent and both
# NeA and NeX are 0, but their ratio still tends to these values.
C_MIN = 9.0 / 16.0  # 0.5625
C_MAX = 9.0 / 8.0  # 1.125

C_NO_SEX_BIAS = 0.75  # value of c when p = 0.5
P_NO_SEX_BIAS = 0.5


def p_to_c(p):
    """
    Convert a proportion of females to the expected value of c.

    p: proportion of females, in [0, 1]
    returns: c = NeX / NeA
    raises ValueError if p is outside [0, 1]
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("proportion of females p = {} is outside [0, 1]".format(p))
    return 9.0 / (8.0 * (2.0 - p))


def c_to_p(c, tol=1e-9):
    """
    Convert a fitted c to the proportion of females it implies.

    c: ratio NeX / NeA, as fit by the X1 (c) or X2 (c1, c2) models
    tol: absolute tolerance on the [C_MIN, C_MAX] bounds, so that a c that
         overshoots by a rounding error is clamped rather than rejected
    returns: proportion of females p, in [0, 1]
    raises ValueError if c is non-positive or outside [C_MIN, C_MAX]
    """
    if c <= 0.0:
        raise ValueError(
            "c = {} is not a positive ratio of effective population sizes".format(c)
        )
    if c < C_MIN - tol or c > C_MAX + tol:
        raise ValueError(
            "c = {:.4f} is outside [{:.4f}, {:.4f}], so no proportion of females "
            "in [0, 1] corresponds to it; the fit likely did not converge".format(
                c, C_MIN, C_MAX
            )
        )
    p = 2.0 - 9.0 / (8.0 * c)
    # Undo the tolerance: a c within tol of a bound can give a p a hair outside [0, 1] due to weird float magic.
    return min(1.0, max(0.0, p))
