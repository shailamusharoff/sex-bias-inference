"""
End-to-end example: simulate a sex-biased bottleneck, fit the autosomal
and X-chromosomal demographic models, and run the sex-bias LRT.

Scenario
--------
A four-epoch model (ancestral -> bottleneck -> recovery -> split)
where only the bottleneck epoch is male-biased (proportation of females is 0.20).
Because the X chromosome carries proportionally more female-inherited
copies, its effective size shrinks more than the autosomes during a
male-biased epoch. The pipeline should detect this as c < 0.75 in the
X1 model (constant sex-bias) relative to the X0 null (c = 0.75 fixed).

Pipeline
--------
1. `ms_simulation_two_pops.run_ms_simulation` runs Hudson's `ms` for
   both autosomes and chromosome X for the specified demographic model,
   writing per-population dadi-format SFS files.
2. `ms_simulation_two_pops.run_sb` fits the autosomal three-epoch model,
   then fits three nested X-chromosomal models (X0, X1, X2), which are
   constrained by the autosomal parameters.
3. The likelihood ratios LL_X1 - LL_X0 (constant sex bias) and
   LL_X2 - LL_X1 (epoch-varying sex bias) are the test statistics,
   each of which are compared to chi-squared with 1 degree of freedom.

Requirements
------------
- Hudson's `ms` on $PATH
- `dadi` python package (`pip install dadi`)
- `ms_functions.py` -- a stub is bundled in the repo root. The stub
  satisfies the import that ms_simulation_two_pops needs and writes a
  placeholder KimTree .dat file. That placeholder is NOT a valid KimTree
  input; do not pass it to `run_kimtree`. Replace ms_functions.py with
  a real converter if you need KimTree integration.

Run
---
    python examples/run_bottleneck_example.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import ms_simulation_two_pops as msim
from dadiLrtFunctions import read1DParams
from compute_lrt_stats import lrt


# --- demography ---------------------------------------------------------
# Times are durations of each epoch in generations, ordered past -> present.
# Sizes are the diploid Ne during the matching epoch.
# propFemales is the proportion of females in each epoch.
#
# Four epochs: ancient -> bottleneck -> recovery -> post-split
# Bottleneck epoch is male-biased (propFemales = 0.2);
# all others have no sex bias (propFemales = 0.5).
TIMES = [10000, 2000, 5000, 5000]
SIZES = [10000, 1000, 10000, 10000]
PROP_FEMALES = [0.5, 0.2, 0.5, 0.5]

# --- simulation parameters ---------------------------------------------
NUM_SAMPLES = 50  # haploid samples per population
NUM_REPS = 1000  # ms replicates
MU = 1.5e-8  # per-bp per-generation mutation rate
L = int(1e6)  # locus length (bp) per replicate
SEEDS = (1, 2, 3)
SIMNUM = 1
OUTDIR = os.path.join(HERE, "out")

os.makedirs(OUTDIR, exist_ok=True)
# fitThreeEpoch (called inside run_sb) writes its outputs to a `lrt_test`
# subdirectory next to the input SFS, so create it ahead of time.
os.makedirs(os.path.join(OUTDIR, "lrt_test"), exist_ok=True)


# --- Step 1: simulate joint and per-population SFS for autosomes and chrX -----------
for chromType in ("A", "X"):
    msim.run_ms_simulation(
        fnName=msim.ms_bottle_epoch_split,
        numSamples=NUM_SAMPLES,
        numReps=NUM_REPS,
        mu=MU,
        L=L,
        chromType=chromType,
        times=TIMES,
        sizes=SIZES,
        propFemales=PROP_FEMALES,
        seeds=SEEDS,
        simnum=SIMNUM,
        outdir=OUTDIR,
    )
# Produces, for each chromType:
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_ms.txt     -- raw ms output
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_joint.fs   -- joint 2D SFS
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_pop1.fs    -- marginal pop1 SFS
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_pop2.fs    -- marginal pop2 SFS


# --- step 2: fit demographic models and run the LRT ---------------------
POPNUM = 1  # which descendant population to analyze
fsfileA = "{}/sim_{}_A_pop{}.fs".format(OUTDIR, SIMNUM, POPNUM)
fsfileX = "{}/sim_{}_X_pop{}.fs".format(OUTDIR, SIMNUM, POPNUM)
outfileA = "{}/sim_{}_A_pop{}_threeEpoch.out".format(OUTDIR, SIMNUM, POPNUM)
modelfileA = "{}/sim_{}_A_pop{}_threeEpoch.fs".format(OUTDIR, SIMNUM, POPNUM)

msim.run_sb(fsfileA, fsfileX, outfileA, modelfileA)


# --- step 3: read log-likelihoods and run the LRT -----------------------
# Each X model .out file ends with a line of optimized params + log-likelihood;
# read1DParams (likType="pois") returns ll_opt from that last line.
xmodels = {}
for model in ("X0", "X1", "X2"):
    outfile = os.path.join(
        OUTDIR,
        "lrt_test",
        "sim_{}_X_pop{}_pois_three_epoch_{}.out".format(SIMNUM, POPNUM, model),
    )
    funcName = "three_epoch_{}".format(model)
    popt, ll_opt, theta, paramDict = read1DParams(funcName, outfile, likType="pois")
    xmodels[model] = {"outfile": outfile, "ll": ll_opt, "params": paramDict}

# X1 vs X0 (constant sex bias) and X2 vs X1 (epoch-varying sex bias),
# each 1 extra free parameter -> chi-squared with 1 degree of freedom.
stat_const, p_const = lrt(xmodels["X0"]["ll"], xmodels["X1"]["ll"], df=1)
stat_vary, p_vary = lrt(xmodels["X1"]["ll"], xmodels["X2"]["ll"], df=1)

# Best model: the most complex model whose entire nested chain is significant.
ALPHA = 0.05
if p_const < ALPHA and p_vary < ALPHA:
    best = "X2"
elif p_const < ALPHA:
    best = "X1"
else:
    best = "X0"


# --- step 4: report results ---------------------------------------------
print()
print("Autosomal fit (three_epoch):")
print("  {}".format(outfileA))
print()
print("X-chromosomal fits (log-likelihood from last line of each .out file):")
for model in ("X0", "X1", "X2"):
    print(
        "  {}: ll = {:.4f}  ({})".format(
            model, xmodels[model]["ll"], xmodels[model]["outfile"]
        )
    )
print()

# Nested-consistency guard: X0 is nested in X1 (c fixed at 0.75 vs free) and
# X1 is nested in X2, so a converged fit must satisfy ll_X0 <= ll_X1 <= ll_X2.
# A violation means the X optimizer did not converge -- the LRT would then be
# meaningless (the negative-clamp in lrt() would silently report stat=0, p=1),
# so warn loudly instead of printing a misleading result.
ll_X0, ll_X1, ll_X2 = (xmodels[m]["ll"] for m in ("X0", "X1", "X2"))
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
print()

c_x1 = xmodels["X1"]["params"].get("c")
print(
    "Estimated constant sex-bias c (X1 model): {:.4f}  (X0 null fixes c = 0.75)".format(
        c_x1
    )
)

if best == "X2":  # epoch-varying sex bias is significant, so report c1, c2
    print(
        "Estimated epoch-varying sex-bias (X2 model): c1 = {:.4f}, c2 = {:.4f}".format(
            xmodels["X2"]["params"].get("c1"), xmodels["X2"]["params"].get("c2")
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
print()

# Best-supported model = most complex model whose full nested chain is significant.
labels = {
    "X0": "no sex bias",
    "X1": "constant sex bias",
    "X2": "epoch-varying sex bias",
}
print("Best-supported model (alpha = {}): {} ({})".format(ALPHA, best, labels[best]))
print("Under the male-biased bottleneck above, expect X2 (epoch-varying sex bias),")
print("with the X1 estimate of c < 0.75.")
