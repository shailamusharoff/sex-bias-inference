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


# --- demography ---------------------------------------------------------
# Times are durations of each epoch in generations, ordered past -> present.
# Sizes are the diploid Ne during the matching epoch.
# propFemales is the proportion of females in each epoch.
#
# Four epochs: ancient -> bottleneck -> recovery -> post-split
# Bottleneck epoch is male-biased (propFemales = 0.2);
# all others have no sex bias (propFemales = 0.5).
TIMES        = [10000, 2000,  5000, 5000]
SIZES        = [10000, 1000, 10000, 10000]
PROP_FEMALES = [0.5,    0.2,   0.5,   0.5]

# --- simulation parameters ---------------------------------------------
NUM_SAMPLES = 50            # haploid samples per population
NUM_REPS    = 1000          # ms replicates
MU          = 1.5e-8        # per-bp per-generation mutation rate
L           = int(1e6)      # locus length (bp) per replicate
SEEDS       = (1, 2, 3)
SIMNUM      = 1
OUTDIR      = os.path.join(HERE, 'out')

os.makedirs(OUTDIR, exist_ok=True)
# fitThreeEpoch (called inside run_sb) writes its outputs to a `lrt_test`
# subdirectory next to the input SFS, so create it ahead of time.
os.makedirs(os.path.join(OUTDIR, 'lrt_test'), exist_ok=True)


# --- Step 1: simulate joint and per-population SFS for autosomes and chrX -----------
for chromType in ('A', 'X'):
    msim.run_ms_simulation(
        fnName=msim.ms_bottle_epoch_split,
        numSamples=NUM_SAMPLES, numReps=NUM_REPS, mu=MU, L=L,
        chromType=chromType,
        times=TIMES, sizes=SIZES, propFemales=PROP_FEMALES,
        seeds=SEEDS, simnum=SIMNUM, outdir=OUTDIR,
    )
# Produces, for each chromType:
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_ms.txt     -- raw ms output
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_joint.fs   -- joint 2D SFS
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_pop1.fs    -- marginal pop1 SFS
#   {OUTDIR}/sim_{SIMNUM}_{chromType}_pop2.fs    -- marginal pop2 SFS


# --- step 2: fit demographic models and run the LRT ---------------------
POPNUM = 1  # which descendant population to analyze
fsfileA    = '{}/sim_{}_A_pop{}.fs'.format(OUTDIR, SIMNUM, POPNUM)
fsfileX    = '{}/sim_{}_X_pop{}.fs'.format(OUTDIR, SIMNUM, POPNUM)
outfileA   = '{}/sim_{}_A_pop{}_threeEpoch.out'.format(OUTDIR, SIMNUM, POPNUM)
modelfileA = '{}/sim_{}_A_pop{}_threeEpoch.fs'.format(OUTDIR, SIMNUM, POPNUM)

msim.run_sb(fsfileA, fsfileX, outfileA, modelfileA)


# --- step 3: report where to find results -------------------------------
print()
print('Autosomal fit (three_epoch):')
print('  {}'.format(outfileA))
print()
print('X-chromosomal fits (last line of each = optimized params + log-likelihood):')
for model in ('X0', 'X1', 'X2'):
    path = os.path.join(
        OUTDIR, 'lrt_test',
        'sim_{}_X_pop{}_pois_three_epoch_{}.out'.format(SIMNUM, POPNUM, model),
    )
    print('  {}: {}'.format(model, path))
print()
print('LRT statistics (compute from the .out files):')
print('  constant sex bias:       2 * (LL_X1 - LL_X0)  vs  chi-sq(1)')
print('  epoch-varying sex bias:  2 * (LL_X2 - LL_X1)  vs  chi-sq(1)')
print()
print('Under the male-biased bottleneck above, expect the X1 estimate of c')
print('to be < 0.75 and the X1-vs-X0 LRT to be significant.')
