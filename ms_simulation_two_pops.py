import os
import sys
import dadi
from numpy import array
import ms_functions as msfunc
import simBottlenecks_functions as simb
import dadiLrtFunctions as lrt


def ms_split_bottle_split(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, msfile, reductionFactors=None, use_theta=True, segsites=None):
    """
    4 epochs and 3 populations
    Split into populations A and 3. Population 3 remains at constant size.
    Population A undergoes instantaneous size change (e.g. bottleleck) then splits into populations 1 and 2.
    s: number of segregating sites to generate
    """
    if(len(times) !=4 | len(sizes) !=4):
        sys.exit('times and sizes must be of length 4')    
    if not use_theta and segsites==None:
        sys.exit('ms_split_bottle_split: if use_theta is False, must provide number of segregating sites s')

    numTotal = 3 * numSamples              # total number of samples
    if reductionFactors is not None:
        sizes = sizes * reductionFactors   # effective sizes that account for propFemales
    N0 = sizes[-1]                         # N0 = Ncurrent
    timesAgo = [ times[-3] + times[-2] + times[-1], times[-2] + times[-1], times[-1], 0.]    # T3, T2, T1, 0
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    T3 = timesAgoMs[0]
    T2 = timesAgoMs[1]
    T1 = timesAgoMs[2]
    sizesMs = [ x / (1. * N0) for x in sizes ]
    thetaMs = 4 * N0 * mu * L
    N3 = sizesMs[0]
    N2 = sizesMs[1]
    N1 = sizesMs[2]

    # -ej t i j: at time t, all lineages from population i move to population j
    # -I npop n1 n2 n3: in a sample of n1+n2+n3 chroms, the first n1 are from pop1, the next n2 are from pop2, and the last n3 are from pop3
    # convention: most recent events on the left to more ancient events on the right. Then, timesAgoMs increases
    # ms 200 1000 -t theta -I 2 100 100 -ej T1 2 1 -eN T1 Nbottle -eN T2 Nprebottle -ej T3 3 1 -eN T3 Nanc
    # TODO could leave off last -eN at time T3 so N1, ancestral pop size, is twice the size of the split pops
    if use_theta:
        cmd = "ms {numTotal} {numReps} -t {thetaMs} -I 3 {numSamples} {numSamples} {numSamples} -ej {T1} 2 1 -eN {T1} {N2} -eN {T2} {N1} -ej {T3} 3 1 -eN {T3} {N1} -seeds {seeds[0]} {seeds[1]} {seeds[2]} > {msfile}".format(**locals())
    else:
        cmd = "ms {numTotal} {numReps} -s {segsites} -I 3 {numSamples} {numSamples} {numSamples} -ej {T1} 2 1 -eN {T1} {N2} -eN {T2} {N1} -ej {T3} 3 1 -eN {T3} {N1} -seeds {seeds[0]} {seeds[1]} {seeds[2]} > {msfile}".format(**locals())
    return cmd        
   

def ms_bottle_split(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, msfile, reductionFactors=None):
    """
    Returns ms command as a string
        
    Instantaneous size change followed by a concurrent split and instantaneous size change (three epochs)
    """
    if(len(times) !=3 | len(sizes) !=3):
        sys.exit('times and sizes must be of length 3')    

    numTotal = 2 * numSamples              # total number of samples
    if reductionFactors is not None:
        sizes = sizes * reductionFactors   # effective sizes that account for propFemales
    N0 = sizes[-1]                         # N0 = Ncurrent
    timesAgo = [ times[-2] + times[-1], times[-1], 0.]
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    sizesMs = [ x / (1. * N0) for x in sizes ]
    thetaMs = 4 * N0 * mu * L

    # ms 200 1000 -t theta -I 2 100 100 -ej T1/(4*N0) 1 2 -eN T1/(4*N0) N2/N0 -eN T2/(4*N0) N1/N0
    # note: T1 = timesAgoMs[1]; T2 = timesAgoMs[0]
    cmd = "ms {numTotal} {numReps} -t {thetaMs} -I 2 {numSamples} {numSamples} -ej {timesAgoMs[1]} 1 2 -eN {timesAgoMs[1]} {sizesMs[1]} -eN {timesAgoMs[0]} {sizesMs[0]} -seeds {seeds[0]} {seeds[1]} {seeds[2]} > {msfile}".format(**locals())  # TODO untested
    return cmd        
        
def ms_bottle_epoch_split(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, msfile, reductionFactors=None):
    """
    Returns ms command as a string

    Two instantaneous size changes followed by split (four epochs)
    """
    if(len(times) !=4 | len(sizes) !=4):
        sys.exit('times and sizes must be of length 4')    
        
    numTotal = 2 * numSamples              # total number of samples
    if reductionFactors is not None:
        sizes = sizes * reductionFactors   # effective sizes that account for propFemales
    N0 = sizes[-1]                         # N0 = Ncurrent
    timesAgo = [ times[-3] + times[-2] + times[-1], times[-2] + times[-1], times[-1], 0.]
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    sizesMs = [ x / (1. * N0) for x in sizes ]
    thetaMs = 4 * N0 * mu * L
     
    # ms 200 1000 -t theta -I 2 100 100 -ej T1/(4*N0) 1 2 -eN T2/(4*N0) N2/N0 -eN T3/(4*N0) N3/N0
    cmd = "ms {numTotal} {numReps} -t {thetaMs} -I 2 {numSamples} {numSamples} -ej {timesAgoMs[2]} 1 2 -eN {timesAgoMs[1]} {sizesMs[1]} -eN {timesAgoMs[0]} {sizesMs[0]} -seeds {seeds[0]} {seeds[1]} {seeds[2]} > {msfile}".format(**locals())
    return cmd    

def run_ms_simulation(fnName, numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, simnum, outdir, use_theta=True, segsites=None):
    """ 
    fnName: simulation function without quotes, e.g. ms_split_bottle_split . TODO change to func instead of fnName
    """
    if chromType == 'A':
       reductionFn = simb.fA
    elif chromType == 'X':
        reductionFn = simb.fX
    else:
        sys.exit('Error: chromType must be either A or X, got {}'.format(chromType))
    reductionFactors = array([ reductionFn(p) for p in propFemales] ) # from the coalescent

    # filenames
    outbase = "{0}/sim_{1}_{2}".format(outdir, simnum, chromType)
    msfile =  "{0}_ms.txt".format(outbase)
    fsfile =  "{0}_joint.fs".format(outbase)
    fsfile1 = "{0}_pop1.fs".format(outbase) 
    fsfile2 = "{0}_pop2.fs".format(outbase) 
    ktfile =  "{0}_joint.dat".format(outbase)

    # run ms simulation
    # hard-coded call: cmd = ms_bottle_epoch_split(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, msfile, reductionFactors)
    cmd = fnName(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, msfile, reductionFactors, use_theta, segsites)
    print cmd
    os.system(cmd)

    # read joint fs and get number of segregating sites
    ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=False)
    ms_fs.to_file(fsfile)
    S = int(ms_fs.S())
    if segsites is not None and S != segsites:
        print 'Error: segsites and S from data differ\n'
    
    # write KimTree .dat file for two simulated pops
    ms_data = msfunc.data_from_ms_file(msfile, average=False, write_kimtree=True, outfile=ktfile, segsites=S)   # this works regardless of whether segsites was in explicit argument to ms via -s or theta was provided via -t
    # to test this command without supplying S as an argument
    # ms_data = msfunc.data_from_ms_file(msfile='/Users/shaila/projects/adglm/data/sb/comparison_sim/lg_test/sim_1_A_ms.txt', average=False, write_kimtree=True, outfile='/Users/shaila/projects/adglm/data/sb/test/sim_1_A_kimtree_noS.dat)       

    # write dadi single-population sfs
    if len(ms_fs.shape) == 2:          # 2D fs
        fs1 = ms_fs.marginalize([1])   # pop1, which has index 0
        fs2 = ms_fs.marginalize([0])   # pop2, which has index 1
    elif len(ms_fs.shape) == 3:          # 3D fs
        fs1 = ms_fs.marginalize([1,2])   # pop1, which has index 0
        fs2 = ms_fs.marginalize([0,2])   # pop2, which has index 1
    else:
        sys.exit('Error: no case no marginalize this j')
    fs1.to_file(fsfile1)
    fs2.to_file(fsfile2)
    
def make_pop3_fs(outdir, numIter):
    """Running after simulations to make population 3 fs
    Call from interpeter: 
        make_pop3_fs(outdir='/Users/shaila/projects/adglm/data/sb/comparison_sim/smoothed_sfs_1e3', numIter=100)
        make_pop3_fs(outdir='/Users/shaila/projects/adglm/data/sb/comparison_sim/smoothed_sfs_1e3', numIter=100)
    """
    
    for simnum in range(1, numIter+1):
        for chromType in 'A', 'X':
            outbase = "{0}/sim_{1}_{2}".format(outdir, simnum, chromType)
            msfile =  "{0}_ms.txt".format(outbase)
            fsfile3 = "{0}_pop3.fs".format(outbase) 
            ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=False)

            # write dadi single-population sfs
            if len(ms_fs.shape) == 2:          # 2D fs
                fs3 = ms_fs.marginalize([2])   # pop3, which has index 2
            elif len(ms_fs.shape) == 3:          # 3D fs
                fs3 = ms_fs.marginalize([0,1])   # pop2, which has index 2
            else:
                sys.exit('Error: no case no marginalize this j')
            fs3.to_file(fsfile3)
    
    
def simSexBiasedBottleneckTwoPop(numSamples, numReps, mu, L, chromType, times, sizes, propFemales, seeds, simnum, outdir):
    """
    Older code that combines ms simulation and writing to files
    Simulates a bottleneck followed by instantaneous growth in the ancestral population, then a two-population split
    Four-epoch model: there is an epoch after end of bottleneck and before split
    
    Modification of bottleneck simulation currently in manuscript: ~/projects/adglm/results/sb/scripts/simBottlenecks_functions.py: simSexBiasedBottleneck(pklfile)
    
    Parameters: times are durations of epochs ordered from past to present, and sizes are population sizes during each epoch
    
    TimesAgoMs and sizesMs are the same length and each index corresponds to an event
    
    See function trueParams to convert to dadi parameters
    """
    if(len(times) !=4 | len(sizes) !=4):
        sys.exit('times and sizes must be of length 4')    
    if chromType == 'A':
       reductionFn = simb.fA
    elif chromType == 'X':
        reductionFn = simb.fX
    else:
        sys.exit('Error: chromType must be either A or X, got {}'.format(chromType))

    # filenames
    outbase = "{0}/sim_{1}_{2}".format(outdir, simnum, chromType)
    msfile =  "{0}_ms.txt".format(outbase)
    fsfile =  "{0}_joint.fs".format(outbase)
    fsfile1 = "{0}_pop1.fs".format(outbase) 
    fsfile2 = "{0}_pop2.fs".format(outbase) 
    ktfile =  "{0}_joint.dat".format(outbase)

    # ms command
    numTotal = 2 * numSamples     # total number of samples
    reductionFactors = array([ reductionFn(p) for p in propFemales] )                    # from the coalescent
    effSizes = sizes * reductionFactors   # effective sizes
    N0 = effSizes[-1]                     # N0 = Ncurrent
    timesAgo = [ times[-3] + times[-2] + times[-1], times[-2] + times[-1], times[-1], 0.]
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    sizesMs = [ x / (1. * N0) for x in effSizes ]
    thetaMs = 4 * N0 * mu * L
    
    # run ms command
    # -I 2 100 100: sample 100 chromosomes from each of two isolated populations. the first 100 chromosomes are from population and the second 100 are population 2
    # cmd example: ms 200 1000 -t theta -I 2 100 100 -ej T1/(4*N0) 1 2 -eN T2/(4*N0) N2/N0 -eN T3/(4*N0) N3/N0
    cmd = "ms {numTotal} {numReps} -t {thetaMs} -I 2 {numSamples} {numSamples} -ej {timesAgoMs[2]} 1 2 -eN {timesAgoMs[1]} {sizesMs[1]} -eN {timesAgoMs[0]} {sizesMs[0]} -seeds {seeds[0]} {seeds[1]} {seeds[2]}".format(**locals())
    cmd += ' > {}'.format(msfile)
    os.system(cmd)

    # write KimTree .dat file for two simulated pops
    ms_data = msfunc.data_from_ms_file(msfile, write_kimtree=True, outfile=ktfile)

    # write dadi sfs
    ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=True)
    ms_fs.to_file(fsfile)          # joint sfs; currently not used
    fs1 = ms_fs.marginalize([1])   # pop1
    fs2 = ms_fs.marginalize([0])   # pop2
    fs1.to_file(fsfile1)
    fs2.to_file(fsfile2)


def fitThreeEpoch(outfileA, infile, likType, funcName, optimizer='optimize_log_fmin', outBase=None, multinom=False):
    """
    Fits X models that are constrained based on A parameters.
    Differs from other functions because ci values are constrained to have biological ranges; also has a good timescale    
    Parameters
    	outfileA: output of auto model fit with param ests at end
	    likType: multinomial, poisson. TODO used only for file names
	    infile:  chrX fs file
  	  multinom: relevant for three_epoch_X_all which has this as an explict param
    	test of new function lrt:three_epoch_X1
    Output: written to directory lrt_test or lrt_test_optimize_log in the same directory as infile
    """

    if optimizer == 'optimize_log_fmin':
        optFn = dadi.Inference.optimize_log_fmin
        if outBase is None:
            outBase = os.path.split(infile)[0] + '/lrt_test/' + os.path.splitext(os.path.split(infile)[1])[0] 
    elif optimizer == 'optimize_log':
        optFn = dadi.Inference.optimize_log
        if outBase is None:
            outBase = os.path.split(infile)[0] + '/lrt_test_optimize_log/' + os.path.splitext(os.path.split(infile)[1])[0]         
    else:
        sys.exit('specified optimizer {} not supported'.format(optimizer))
        
    # read auto params from file. will fix all but c for this model
    AfuncName = 'three_epoch'
    popt, ll_opt, thetaA, paramDict = lrt.read1DParams(AfuncName, outfileA)
    nuB = paramDict['nuB']
    nuF = paramDict['nuF']    
    TB = paramDict['TB']
    TF = paramDict['TF']    

    # chrX optimization parameters and file names
    timescale = 1e-4   # this alone does do anything because I do not use the optimizer functon fit1DModel below. Need to set dadi.Integration.timescale_factor = timescale directly
    dadi.Integration.timescale_factor = timescale
    minGrid = 150      # larger grid?
    outfile = outBase + '_{}_{}.out'.format(likType, funcName)
    modelfile = outBase + '_{}_{}.dadi'.format(likType, funcName)
    logfile = outfile  # same file for both
    maxiter = None
    pts_l = [minGrid, minGrid+10, minGrid+20]
    perturb_fold = 1
    flush_delay = 0.5         # default

    # read in chrX data
    data = dadi.Spectrum.from_file(infile)      
    ns = data.sample_sizes

    # set params for each fn or eval lik
    if funcName == 'three_epoch_X0':                 # do not optimize because c = 0.75
        poptA = array([nuB,nuF,TB,TF,thetaA,0.75])   # all fixed params
        func = lrt.three_epoch_X0
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        model = func_ex(poptA, ns, pts_l)
        ll_opt = dadi.Inference.ll(model, data)        
        popt = lrt.getXparams(poptA, funcName)    # write popt, params in terms of chrX. poptA is auto terms followed by c's
        with open(outfile, 'a') as outF:
            outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt)
            outF.write(outstr)
        model.to_file(modelfile)

    elif funcName == 'three_epoch_X1' or funcName == 'three_epoch_X2':   # optimize chrX parameters to fit c's
        if funcName == 'three_epoch_X1':
            func = lrt.three_epoch_X1            
            func_ex = dadi.Numerics.make_extrap_log_func(func)
            params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
            upper_bound = [1, 10e2, 1, 1, 1e6, 1.11]   # only last bound matters, is for c
            lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.57]        
            fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
        else:
            func = lrt.three_epoch_X2
            func_ex = dadi.Numerics.make_extrap_log_func(func)
            params = array([nuB,nuF,TB,TF,thetaA,0.75,0.75])   # starting point for opt
            upper_bound = [1, 10e2, 1, 1, 1e6, 1.11, 1.11]   # only last two bounds matters, is for c
            lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.57, 0.57]        
            fixed_params = array([nuB,nuF,TB,TF,thetaA,None,None])  # c1 and c2 are last and free
        p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
        poptA = optFn(p0, data, func_ex, pts_l, lower_bound=lower_bound, upper_bound=upper_bound, verbose=len(params), maxiter=maxiter, output_file=logfile, flush_delay=flush_delay, fixed_params=fixed_params, multinom=False)
        model = func_ex(poptA, ns, pts_l)  # should be same as at end of optimization
        ll_opt = dadi.Inference.ll(model, data)
        popt = lrt.getXparams(poptA, funcName)    # write popt, params in terms of chrX. poptA is auto terms followed by c's
        with open(outfile, 'a') as outF:
            outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt)
            outF.write(outstr)
        model.to_file(modelfile)

def run_sb(fsfileA, fsfileX, outfileA, modelfileA):
    """Works. Fixed simulationBottlenecks_functions.py:fitThreeEpoch functions
    Fits autosomal and X-chromosomal models    
    Writes outfiles: <fsfileX without .fs>_pois_three_epoch_(X0,X1,X2).(out, dadi)
    Updated to use fitThreeEpoch function here, with constraints. Did not change main function because is used by other code, e.g. bottleneck simulations
    """
    (poptA, ll_optA, thetaA) = lrt.fitThreeEpoch(fsfileA, outfileA, modelfileA, isCluster=False)
    # simb.fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X0')
    # simb.fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X1')
    # simb.fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X2')
    fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X0')
    fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X1')
    fitThreeEpoch(outfileA, fsfileX, 'pois', 'three_epoch_X2')

def run_kimtree(datfileA, datfileX, treef, outdir, do_run=True):
    cmd = "/ye/zaitlenlabstore/shailam/sb/src/KimTree_2.0.1/src/kimtree -npilot 20 -lpilot 500 -burnin 10000 -length 20000 -thin 20 -file {0} -Xfile {1} -tree {2} -threads 2 -outputs {3}".format(datfileA, datfileX, treef, outdir)
    if do_run:
        os.system(cmd)
    return cmd
    
