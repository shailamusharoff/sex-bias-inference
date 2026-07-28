import os
import sys
import ms_simulation_two_pops as msim
from dadiLrtFunctions import read1DParams

# compute_lrt_stats lives in examples/; add it to the path so we can reuse
# the same lrt() helper the example uses (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
from compute_lrt_stats import lrt


import argparse

parser = argparse.ArgumentParser(description="Run SB method on simulated data")
parser.add_argument(
    "--outdir", help="directory containing output (and input) files", required=True
)
parser.add_argument(
    "--simnum", type=int, help="simulation number. part of filename", required=True
)
parser.add_argument(
    "--popnum", type=int, help="population number: 1, 2, or 3", required=True
)
args = parser.parse_args()


def call_run_sb(simnum, outdir, popnum):
    """fit X and A with same three-epoch demographic model
    requires a directory lrt_test in outdir directory: where output is written"""
    fsfileA = "{}/sim_{}_A_pop{}.fs".format(outdir, simnum, popnum)
    fsfileX = "{}/sim_{}_X_pop{}.fs".format(outdir, simnum, popnum)
    outfileA = "{}/sim_{}_A_pop{}_threeEpoch.out".format(
        outdir, simnum, popnum
    )  # fit A file
    modelfileA = "{}/sim_{}_A_pop{}_threeEpoch.fs".format(outdir, simnum, popnum)
    msim.run_sb(fsfileA, fsfileX, outfileA, modelfileA)


def compute_and_report_lrt(simnum, outdir, popnum):
    """read log-likelihoods of the nested X models (X0, X1, X2) and run the
    sex-bias LRT:
      constant sex bias       X1 vs X0  (1 extra free param -> chi-sq(1))
      epoch-varying sex bias  X2 vs X1  (1 extra free param -> chi-sq(1))
    The X model .out files (written by run_sb into outdir/lrt_test) end with
    a line of optimized params + log-likelihood; read1DParams(likType="pois")
    returns ll_opt from that last line."""
    xmodels = {}
    for model in ("X0", "X1", "X2"):
        path = os.path.join(
            outdir,
            "lrt_test",
            "sim_{}_X_pop{}_pois_three_epoch_{}.out".format(simnum, popnum, model),
        )
        funcName = "three_epoch_{}".format(model)
        popt, ll_opt, theta, paramDict = read1DParams(funcName, path, likType="pois")
        xmodels[model] = {"path": path, "ll": ll_opt, "params": paramDict}

    ll_X0 = xmodels["X0"]["ll"]
    ll_X1 = xmodels["X1"]["ll"]
    ll_X2 = xmodels["X2"]["ll"]

    stat_const, p_const = lrt(ll_X0, ll_X1, df=1)
    stat_vary, p_vary = lrt(ll_X1, ll_X2, df=1)

    print()
    print("X-chromosomal fits (log-likelihood from last line of each .out file):")
    for model in ("X0", "X1", "X2"):
        print(
            "  {}: ll = {:.4f}  ({})".format(
                model, xmodels[model]["ll"], xmodels[model]["path"]
            )
        )
    print()

    # Nested-consistency guard: X0 is nested in X1 (c fixed at 0.75 vs free) and
    # X1 is nested in X2, so a converged fit must satisfy ll_X0 <= ll_X1 <= ll_X2.
    # A violation means the X optimizer did not converge -- the LRT would then be
    # meaningless (the negative-clamp in lrt() would silently report stat=0, p=1),
    # so warn loudly instead of printing a misleading result.
    if ll_X1 < ll_X0:
        print(
            "WARNING: ll_X1 ({:.4f}) < ll_X0 ({:.4f}) -- X1 fit did not converge; "
            "X1-vs-X0 LRT is not valid. Re-run the fit.".format(ll_X1, ll_X0)
        )
    if ll_X2 < ll_X1:
        print(
            "WARNING: ll_X2 ({:.4f}) < ll_X1 ({:.4f}) -- X2 fit did not converge; "
            "X2-vs-X1 LRT is not valid. Re-run the fit.".format(ll_X2, ll_X1)
        )

    c_x1 = xmodels["X1"]["params"].get("c")
    if c_x1 is not None:
        print(
            "Estimated constant sex-bias c (X1 model): {:.4f}  (X0 null fixes c = 0.75)".format(
                c_x1
            )
        )
        print()
    print("LRT statistics (vs chi-squared with 1 degree of freedom):")
    print(
        "  constant sex bias      (X1 vs X0):  stat = {:.4f}  p = {:.4g}".format(
            stat_const, p_const
        )
    )
    print(
        "  epoch-varying sex bias (X2 vs X1):  stat = {:.4f}  p = {:.4g}".format(
            stat_vary, p_vary
        )
    )

    return {"const": (stat_const, p_const), "vary": (stat_vary, p_vary)}


if __name__ == "__main__":
    call_run_sb(args.simnum, args.outdir, args.popnum)
    compute_and_report_lrt(args.simnum, args.outdir, args.popnum)
