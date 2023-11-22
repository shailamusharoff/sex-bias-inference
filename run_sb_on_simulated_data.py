import os
import sys
import ms_simulation_two_pops as msim


import argparse
parser = argparse.ArgumentParser(description='Run SB method on simulated data')
parser.add_argument('--outdir', help='directory containing output (and input) files', required=True)
parser.add_argument('--simnum', type=int, help='simulation number. part of filename', required=True)
parser.add_argument('--popnum', type=int, help='population number: 1, 2, or 3', required=True)
args = parser.parse_args()

def call_run_sb(simnum, outdir, popnum):
    """ fit X and A with same three-epoch demographic model
        requires a directory lrt_test in outdir directory: where output is written """        
    fsfileA = '{}/sim_{}_A_pop{}.fs'.format(outdir, simnum, popnum)
    fsfileX = '{}/sim_{}_X_pop{}.fs'.format(outdir, simnum, popnum)
    outfileA = '{}/sim_{}_A_pop{}_threeEpoch.out'.format(outdir, simnum, popnum)    # fit A file 
    modelfileA = '{}/sim_{}_A_pop{}_threeEpoch.fs'.format(outdir, simnum, popnum)
    msim.run_sb(fsfileA, fsfileX, outfileA, modelfileA)

if __name__ == '__main__':
    call_run_sb(args.simnum, args.outdir, args.popnum)
