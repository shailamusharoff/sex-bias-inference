import random
import os
import sys
import json
import string
import cPickle
import pprint
import dadi
import argparse
import numpy
from numpy import array
import sfsFunctions as f
import dadiLrtFunctions as lrt
import Plotting as myPlot
import pythonUtils as pyUtils
homeDir = os.path.expanduser('~')

### globals. remove this to have different values
mu = 1.5e-8
L = 1e4
allProps = ['0.1_0.1_0.1', '0.1_0.2_0.1', '0.1_0.3_0.1', '0.1_0.4_0.1', '0.1_0.5_0.1', '0.1_0.6_0.1', '0.1_0.7_0.1', '0.1_0.8_0.1', '0.1_0.9_0.1', '0.2_0.1_0.2', '0.2_0.2_0.2', '0.2_0.3_0.2', '0.2_0.4_0.2', '0.2_0.5_0.2', '0.2_0.6_0.2', '0.2_0.7_0.2', '0.2_0.8_0.2', '0.2_0.9_0.2', '0.3_0.1_0.3', '0.3_0.2_0.3', '0.3_0.3_0.3', '0.3_0.4_0.3', '0.3_0.5_0.3', '0.3_0.6_0.3', '0.3_0.7_0.3', '0.3_0.8_0.3', '0.3_0.9_0.3', '0.4_0.1_0.4', '0.4_0.2_0.4', '0.4_0.3_0.4', '0.4_0.4_0.4', '0.4_0.5_0.4', '0.4_0.6_0.4', '0.4_0.7_0.4', '0.4_0.8_0.4', '0.4_0.9_0.4', '0.5_0.1_0.5', '0.5_0.2_0.5', '0.5_0.3_0.5', '0.5_0.4_0.5', '0.5_0.5_0.5', '0.5_0.6_0.5', '0.5_0.7_0.5', '0.5_0.8_0.5', '0.5_0.9_0.5', '0.6_0.1_0.6', '0.6_0.2_0.6', '0.6_0.3_0.6', '0.6_0.4_0.6', '0.6_0.5_0.6', '0.6_0.6_0.6', '0.6_0.7_0.6', '0.6_0.8_0.6', '0.6_0.9_0.6', '0.7_0.1_0.7', '0.7_0.2_0.7', '0.7_0.3_0.7', '0.7_0.4_0.7', '0.7_0.5_0.7', '0.7_0.6_0.7', '0.7_0.7_0.7', '0.7_0.8_0.7', '0.7_0.9_0.7', '0.8_0.1_0.8', '0.8_0.2_0.8', '0.8_0.3_0.8', '0.8_0.4_0.8', '0.8_0.5_0.8', '0.8_0.6_0.8', '0.8_0.7_0.8', '0.8_0.8_0.8', '0.8_0.9_0.8', '0.9_0.1_0.9', '0.9_0.2_0.9', '0.9_0.3_0.9', '0.9_0.4_0.9', '0.9_0.5_0.9', '0.9_0.6_0.9', '0.9_0.7_0.9', '0.9_0.8_0.9', '0.9_0.9_0.9']


def fA(p):
    f = 4 * p * (1-p)
    return f

def fX(p):
    f =  (9 * p * (1-p)) / ((2-p) * 2)
    return f


def runWriteConstants():
    """
    production run on cluster
    outfile: gets pickle files written to it, one per line. later used as input
    """
    machType = 'scg3'           # determines paths
    numReps = int(1e5)          # can go up to 1e6 for better fs
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint

    numIters = int(len(chromTypes) * len(pfVals)**2)
    numSeeds = 3 * numIters
    seedList = random.sample(range(numIters*1000), numSeeds)   # heuristic on range

    ctr = 0
    for chromType in chromTypes:
        for pfConstant in pfVals:
            for pfBottle in pfVals:
                propFemales = array( [pfConstant, pfBottle, pfConstant] )
                seeds = seedList[ 3*ctr : (3*ctr)+3 ]
                writeConstantsToFile(propFemales, chromType, machType, numReps, seeds)
                ctr += 1


def readConstantsTest():
    """
    print out to see - pass.
    """

    machType = 'scg3'           # determines paths
    numReps = int(1e5)          # can go up to 1e6 for better fs
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint

    for chromType in chromTypes:
        for pfConstant in pfVals:
            for pfBottle in pfVals:
                propFemales = array( [pfConstant, pfBottle, pfConstant] )
                propList = '_'.join([str(x) for x in propFemales])
                resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
                resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
                pklfile = resBase + '.pkl'
                pklF = open(pklfile, 'rb')
                paramDict = cPickle.load(pklF)
                pklF.close()
                print pklfile
                pprint.pprint(paramDict)


def runWriteConstantsTest():
    propFemales = array([0.5, 0.1, 0.5])     # one per epoch
    chromType = 'A'   # is A or X; determines reduction factor
    machType = 'shimac'   # change to scg3 for production run
    numReps = int(1e3)    # debugging; increase for prodcution run
    seeds = [100,200, 400]
    writeConstantsToFile(propFemales, chromType, machType, numReps, seeds)

    
def writeConstantsToFile(propFemales, chromType, machType, numReps, seeds):
    """
    each sim has a persistent proportion of females, and one during the bottleneck.
    propFemales: numpy array
    chromType: X or A. no error checking
    seeds: list of three unique random seeds
    Note: written to dict which is not dynamic
    """

    if machType == 'scg3':
        homeDir = '/home/shailam'
    else:
        homeDir = '/Users/shaila'

    resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
    datadir = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
    mu = 1.5e-8
    L = 1e4
    numSamples = 100
    
    # times and absolute sizes going forward in time, in generations
    times = array([3800., 1120., 920.])      # epochs
    sizes = array([1.45e4, 1.86e3, 1e5])     # absolute
    
    # filenames
    propList = '_'.join([str(x) for x in propFemales])
    resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
    dataBase = '{}/bottleneck_{}_{}_{}reps'.format(datadir, chromType, propList, numReps)
    msfile = dataBase + '.ms'
    logfile = dataBase + '.log'        # not short: one line per rep
    fsfile = resBase + '.fs'
    pklfile = resBase + '.pkl'

    paramNameList = ['propFemales', 'chromType', 'machType', 'numReps', 'seeds', 'homeDir', 'resdir', 'datadir', 'mu', 'L', 'numSamples', 'numReps', 'times', 'sizes', 'propList', 'resBase', 'dataBase', 'msfile', 'fsfile', 'logfile', 'pklfile']
    paramDict = dict( [(x, locals()[x]) for x in paramNameList] )

    # write to a cPickle file
    pklF = open(pklfile, 'wb')
    cPickle.dump(paramDict, pklF)
    pklF.close()


def runUpdateConstantsToFile(infile, outfile):    
    """
    infile: one picklefile per line
    outfile: same as input, updated name
    updates picklefile paths
    """

    pklDict = {}
    oldDir = 'ms_bottlenecks'
    newDir = 'ms_bottlenecks_1000x'
    with open(outfile, 'w') as outF:
        for line in open(infile):
            oldPklfile = line.strip()
            propList, chromType, newPklfile = updateConstantsToFile(oldPklfile, oldDir, newDir)
            if propList not in pklDict:
                pklDict[propList] = {}
            pklDict[propList][chromType] = newPklfile
            outF.write(newPklfile + '\n')
    # write dictionary if picklefiles
    resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/'.format(homeDir)
    dictout = resdir + 'allParams.pkl'
    with open(dictout, 'wb') as pklF:
        cPickle.dump(pklDict, pklF)
            
def updateConstantsToFile(inPklfile, oldDir, newDir):
    """
    reads in old pickle files and updates paths of fsfile, resdir, resBase, pklfile
    """

    outPklfile = string.replace(inPklfile, oldDir, newDir)
    with open(inPklfile, 'rb') as pklF:
        paramDict = cPickle.load(pklF)
    
    # update fields
    paramDict['fsfile'] = string.replace(paramDict['fsfile'], oldDir, newDir)
    paramDict['resdir'] = string.replace(paramDict['resdir'], oldDir, newDir)
    paramDict['resBase'] = string.replace(paramDict['resBase'], oldDir, newDir)        
    paramDict['pklfile'] = string.replace(paramDict['pklfile'], oldDir, newDir)    

    # write to new pklfile
    with open(outPklfile, 'wb') as pklF:
        cPickle.dump(paramDict, pklF)
    return paramDict['propList'], paramDict['chromType'], paramDict['pklfile']

def trueParams(propFemales, chromType):
    """
    returns a string of true params for a single simulation
    Parameters: times are durations of epochs ordered from past to present, and sizes are population sizes during each epoch
    Parameters are stored this way so they can be easily converted to dadi or ms parameters
    """
    mu = 1.5e-8
    L = 1e4
    numSamples = 100
    # times and absolute sizes going forward in time, in generations
    times = array([3800., 1120., 920.])      # duration of epoch in generations
    sizes = array([1.45e4, 1.86e3, 1e5])     # population size during epoch
    
    if chromType == 'A':
        reductionFn = fA
    if chromType == 'X':
        reductionFn = fX

    reductionFactors = array([ reductionFn(p) for p in propFemales])     # from the coalescent
    effSizes = sizes * reductionFactors   # effective sizes

    # params fwd in time
    fwdParams = numpy.append(effSizes, times)
    
    # ms params
    N0 = effSizes[-1]          # N0 = Ncurrent
    timesAgo = [ times[2] + times[1], times[2], 0. ]   # NOTE this is hard-coded for a times array of length 3. Better: use - indexing from end. Most recent length is on the right and is not used. times is of length 2, so times[2] is the most recent epoch length in generations.
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    sizesMs = [ x / (1. * N0) for x in effSizes ]
    thetaMs = 4 * N0 * mu * L
    msParams = array( [timesAgoMs[1], sizesMs[1], timesAgoMs[0], sizesMs[0], thetaMs ])
    
    # dadi params
    Nref = effSizes[0]
    timesDadi = [ 0., times[1], times[2] ]      # zero in front is a NOTE bc it is not used
    timesDadi = [ x / (2. * Nref) for x in timesDadi ]
    sizesDadi = [ x / Nref for x in effSizes]   # converts to fold-size changes in units of Nref
    thetaDadi = 4 * Nref * mu * L
    dadiParams = array( [sizesDadi[1], sizesDadi[2], timesDadi[1], timesDadi[2], thetaDadi])

    # format output string
    outarr = numpy.concatenate([ array([mu, L]), propFemales, fwdParams, msParams, dadiParams])
    outarr = [chromType] + [ pyUtils.to_precision(x, 5) for x in outarr]
    outstr = '\t'.join(outarr) + '\n'
    return outstr
    

def writeTrueParams(outfile):
    """
    writes true ms params to file for all sim
    """
    chromTypes = ['A', 'X']
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint
    outF = open(outfile, 'w')
    header = 'chrom\tmu\tL\tp1\tp2\tp3\tN1\tN2\tN3\tt1\tt2\tt3\tms_t1\tms_n1\tms_t0\tms_n0\tms_theta\tdadi_nuB\tdadi_nuF\tdadi_tauB\tdadi_tauF\tdadi_theta\n'
    outF.write(header)

    for pfConstant in pfVals:
        for pfBottle in pfVals:
            for chromType in chromTypes:        
                propFemales = array( [pfConstant, pfBottle, pfConstant] )
                # print '---------- pF {}: {} ----------'.format(chromType, propFemales)
                outstr = trueParams(propFemales, chromType)
                outF.write(outstr)
    outF.close()


def p(Q):
    """
    returns value of proportion females (breeding ratio) corresponding to Q. What are the assumptions here?
    """
    return 2. - (9./8) * (1./Q)    

def calcPhatEpoch(outfileA, outfileX):
    """
    estimates effectives sizes and phat for each epoch
    input: output files from fitting containing estimated params
    """
    outfileDict = {'A':outfileA, 'X':outfileX}
    NeDict = {}

    # read parameters from X and A file
    funcName = 'three_epoch'
    chromType = ['A', 'X']
    for chrom in chromType:
        popt, ll_opt, theta, paramDict = lrt.read1DParams(funcName, outfileDict[chrom])
        Nanc = theta / (4 * mu * L)
        N1 = Nanc
        N2 = Nanc * paramDict['nuB']
        N3 = Nanc * paramDict['nuF']
        NeDict[chrom] = [N1, N2, N3]
        # print '{}: N1 = {:.1f}, N2 = {:.1f}, N3 = {:.1f}'.format(chrom, N1, N2, N3)

    Qarr = numpy.divide(NeDict['X'], NeDict['A'])
    parr = [ p(Q) for Q in Qarr]
    return parr


def estPtilde(infileA, infileX, modelfileA, modelfileX):
    muA = muX = mu   # might be bad syntactically
    LA = LX = L
    alpha = 1
    phat = lrt.estPnowrite(infileX, modelfileX, infileA, modelfileA, muA, LA, LX, alpha)
    return phat

def estQpiMs(logfileA, logfileX):
    """
    estimates Qpi from ms files
    only works on scg because that is where data files are, and because it calls datamash
    logfile:  ms stats output with pi in the second column for each ms iteration
    """
    colNum = 1  # 0-based
    piA = pyUtils.getMean(logfileA, colNum)
    piX = pyUtils.getMean(logfileX, colNum)    
    Qpi = piX / piA
    return Qpi

def estQpiSFS(infileA, infileX):
    """
    estimates Qpi from dadi input sfs files
    """
    fsA = dadi.Spectrum.from_file(infileA)    
    fsX = dadi.Spectrum.from_file(infileX)        
    piA = fsA.pi()
    piX = fsX.pi()
    Qpi = piX / piA
    return Qpi


def getResults(propFemales, resdir, datadir):
    """
    called by a run function
    TODO take a boolean to calc from ms files - might not have
    """
    propList = '_'.join([str(x) for x in propFemales])

    resBaseA = '{}/bottleneck_{}_{}_{}reps'.format(resdir, 'A', propList, numReps)
    resBaseX = '{}/bottleneck_{}_{}_{}reps'.format(resdir, 'X', propList, numReps)            
    dataBaseA = '{}/bottleneck_{}_{}_{}reps'.format(datadir, 'A', propList, numReps)
    dataBaseX = '{}/bottleneck_{}_{}_{}reps'.format(datadir, 'X', propList, numReps)            

    # proportion of females in each epoch
    parr = calcPhatEpoch(resBaseA + '.out', resBaseX + '.out')

    # ptilde: single value (ancestral?)
    ptilde = estPtilde(resBaseA + '.fs', resBaseX + '.fs', resBaseA + '.dadi', resBaseX + '.dadi')
    
    # Qpi from ms sims
    QpiMs = estQpiMs(dataBaseA + '.log', dataBaseX + '.log')
    ppiMs = p(QpiMs)

    # Qpi from sfs
    QpiSFS = estQpiSFS(resBaseA + '.fs', resBaseX + '.fs')
    ppiSFS = p(QpiSFS)

    outarr = propFemales.tolist() + parr + [ptilde, QpiMs, ppiMs, QpiSFS, ppiSFS]
    outstr = '\t'.join([str(x) for x in outarr]) + '\n'
    return outstr


def runGetResults(outfile):
    """
    for each set of input parameters, call fn
    """    
    numReps = int(1e5)          # can go up to 1e6 for better fs
    resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
    datadir = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)

    outF = open(outfile, 'w')
    header = 'p1\tp2\tp3\tp1hat\tp2hat\tp3hat\tptilde\tQpiMs\tppiMs\tQpiSFS\tppiSFS\n'
    outF.write(header)
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint
    chromTypes = ['A', 'X']
    for pfConstant in pfVals:
        for pfBottle in pfVals:
            propFemales = array( [pfConstant, pfBottle, pfConstant] )
            outstr = getResults(propFemales, resdir, datadir)
            outF.write(outstr)
    outF.close()


def bestOpt(fsfile, numOpts, funcName):
    """
    return filename of optmization run with best parameter
    """
    resBase = os.path.splitext(fsfile)[0]        
    likList = []
    optList = []
    for optNum in range(1,numOpts):   #NOTE bc opt num0 got renamed
        outfile = resBase + '_opt{}.out'.format(optNum)
        popt, ll_opt, theta, paramDict = lrt.read1DParams(funcName, outfile)     # for each file, read in lik
        likList.append(ll_opt)
        optList.append(optNum)
    # choose best lik
    idxBestLik = likList.index(max(likList))
    bestNum = optList[idxBestLik]
    bestOutfile = resBase + '_opt{}.out'.format(bestNum)    
    return bestOutfile


def runBestOpt(infile, outfile):
    """
    loop over file with input dadi fs filenames
    writes file names to outfile
    """
    funcName = 'three_epoch'
    numOpts = 10
    with open(outfile, 'w') as outF:
        for line in open(infile):
            fsfile = line.strip()
            bestOutfile = bestOpt(fsfile, numOpts, funcName)
            outF.write(bestOutfile + '\n')


def bestOptDict(fsfile, numOpts, funcName):
    """
    return dictionary of optmization run with best parameter
    """
    resBase = os.path.splitext(fsfile)[0]        
    likList = []
    optList = []
    for optNum in range(1,numOpts):   #NOTE bc opt num0 got renamed
        outfile = resBase + '_opt{}.out'.format(optNum)
        popt, ll_opt, theta, paramDict = lrt.read1DParams(funcName, outfile)     # for each file, read in lik
        likList.append(ll_opt)
        optList.append(optNum)
    # choose best lik
    idxBestLik = likList.index(max(likList))
    bestNum = optList[idxBestLik]
    bestOutfile = resBase + '_opt{}.out'.format(bestNum)    
    return bestNum, bestOutfile


def runBestOptDict(infile, outfile):
    """
    loop all pickle files with names of paths, etc
    writes dictionary with best opt to outfile
    """
    funcName = 'three_epoch'
    numOpts = 10
    optDict = {}   # key1 = propList, key2 = chrom, value = best opt file
    # loop over pickle files: have constants and file names for each simulation
    for line in open(infile):
        pklfile = line.strip()
        pklF = open(pklfile, 'rb')
        paramDict = cPickle.load(pklF)
        pklF.close()
    
        bestNum, bestOutfile = bestOptDict(paramDict['fsfile'], numOpts, funcName)
        propList = paramDict['propList']
        chromType = paramDict['chromType']        
        if propList not in optDict:
            optDict[propList] = {}
        optDict[propList][chromType] = {'bestNum': bestNum, 'bestOutfile': bestOutfile}

        # add to original pickle file
        paramDict['bestNum'] = bestNum
        paramDict['bestOutfile'] = bestOutfile        
        paramDict['bestModelfile'] = string.replace(bestOutfile, '.out', '.dadi')  
        with open(pklfile, 'wb') as pklF:
            cPickle.dump(paramDict, pklF)
        
    with open(outfile, 'wb') as outF:
        cPickle.dump(optDict, outF)
        

def getResultsDict(pklfileA, pklfileX):
    """
    A and X can have different optimizations considered
    paramDict has input params
    TODO take a boolean to calc from ms files - might not have
    """

    A = cPickle.load(open(pklfileA, 'rb'))
    X = cPickle.load(open(pklfileX, 'rb'))

    ## proportion of females in each epoch: best outfile from opt
    parr = calcPhatEpoch(A['bestOutfile'], X['bestOutfile'])   # (resBaseA + '.out', resBaseX + '.out')

    ## ptilde: single value (ancestral?): fsfile, best modelfile from opt
    ptilde = estPtilde(A['fsfile'], X['fsfile'], A['bestModelfile'], X['bestModelfile'])  # (resBaseA + '.fs', resBaseX + '.fs', resBaseA + '.dadi', resBaseX + '.dadi')
    
    ## Qpi from ms sims: logfile
    QpiMs = estQpiMs(A['logfile'], X['logfile'])  # (dataBaseA + '.log', dataBaseX + '.log')
    ppiMs = p(QpiMs)

    # Qpi from sfs: fsfile
    QpiSFS = estQpiSFS(A['fsfile'], X['fsfile'])   # (resBaseA + '.fs', resBaseX + '.fs')
    ppiSFS = p(QpiSFS)

    outarr = A['propFemales'].tolist() + parr + [ptilde, QpiMs, ppiMs, QpiSFS, ppiSFS]
    outstr = '\t'.join([str(x) for x in outarr]) + '\n'
    return outstr


            
def runGetResultsDict(infile, outfile):
    """
    input: dictionary with one picklefile containing params. something like allParams.pkl
    output: file with results
    """    

    # header to main output file
    outF = open(outfile, 'w')
    header = 'p1\tp2\tp3\tp1hat\tp2hat\tp3hat\tptilde\tQpiMs\tppiMs\tQpiSFS\tppiSFS\n'   # TODO update if necessary
    outF.write(header)
    
    # read in dictionary of all picklefiles
    with open(infile, 'rb') as pklF:
        pklDict = cPickle.load(pklF)

    # loop over propLists and open pickle files for X and A
    for k,v in sorted(pklDict.items()):
        pklfileA = v['A']
        pklfileX = v['X']        
        outstr = getResultsDict(pklfileA, pklfileX)
        outF.write(outstr)
    outF.close()
    
    
def simSexBiasedBottleneck(pklfile):
    """
    can have a persistent proportion of females, and one during the bottleneck.
    times, sizes, propFemales: lists of length the number of epochs
    """

    # read params from pickleFile: TODO broken
    pklF = open(pklfile, 'rb')
    inParams = cPickle.load(pklF)
    pklF.close()

    # scale sizes based on p's: then all will are effective sizes
    if inParams['chromType'] == 'A':
        reductionFn = fA
    elif inParams['chromType'] == 'X':
        reductionFn = fX
    else:
        print 'error: chromType must be either A or X, got {}'.format(inParams['chromType'])
    reductionFactors = array([ reductionFn(p) for p in inParams['propFemales']] )     # from the coalescent
    effSizes = inParams['sizes'] * reductionFactors   # effective sizes
    
    # ms parameters
    N0 = effSizes[-1]          # N0 = Ncurrent
    timesAgo = [ inParams['times'][2] + inParams['times'][1], inParams['times'][2], 0. ]
    timesAgoMs = [ x / (4. * N0) for x in timesAgo]
    sizesMs = [ x / (1. * N0) for x in effSizes ]
    thetaMs = 4 * N0 * inParams['mu'] * inParams['L']
    
    # run ms command and write fs to file
    cmd = 'ms {0} {1} -t {2} -eN {3} {4} -eN {5} {6} -seeds {7} {8} {9}'.format(inParams['numSamples'], inParams['numReps'], thetaMs, timesAgoMs[1], sizesMs[1], timesAgoMs[0], sizesMs[0], inParams['seeds'][0], inParams['seeds'][1], inParams['seeds'][2])
    
    cmd += ' > {}'.format(inParams['msfile'])
    os.system(cmd)
    ms_fs = dadi.Spectrum.from_ms_file(inParams['msfile'], mask_corners=True, average=True)
    ms_fs.to_file(inParams['fsfile'])
    
    # sample_stats, then remove ms file
    cmd = 'cat {} | {}/popgen/msdir/sample_stats > {}'.format(inParams['msfile'], homeDir, inParams['logfile'])
    os.system(cmd)
    

def runFitBottleneck():
    numReps = int(1e5)
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint
    for chromType in chromTypes:
        for pfConstant in pfVals:
            for pfBottle in pfVals:
                propFemales = array( [pfConstant, pfBottle, pfConstant] )
                propList = '_'.join([str(x) for x in propFemales])
                resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
                resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
                fitBottleneck(resBase, isCluster=True)


def poissonSample(infile, numSamples, outBase):
    """
    uses dadi Poisson sample function to make a bootstrap sample fs
    """

    fs = dadi.Spectrum.from_file(infile)
    for i in range(numSamples):
        fsSample = fs.sample()
        outfile = '{}_boot_{}.fs'.format(outBase, i)
        fsSample.to_file(outfile)
    

def fitThreeEpoch(outfileA, infile, likType, funcName, optimizer='optimize_log_fmin', outBase=None, multinom=False):
    """
    Fits X models that are constrained based on A parameters.
    newer fn with good timescale
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
            upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
            lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]        
            fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
            # model = func_ex(poptA, ns, pts_l)  # should be same as at end of optimization. TODO bug? should not be called here
            # ll_opt = dadi.Inference.ll(model, data)            
        else:
            func = lrt.three_epoch_X2
            func_ex = dadi.Numerics.make_extrap_log_func(func)
            params = array([nuB,nuF,TB,TF,thetaA,0.75,0.75])   # starting point for opt
            upper_bound = [1, 10e2, 1, 1, 1e6, 2, 2]   # only last two bounds matters, is for c
            lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1, 0.1]        
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
        
    elif funcName == 'three_epoch_X1_all':
        modelType = 'X1'
        func = lrt.three_epoch_X
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]        
        fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
        func_args = [multinom, modelType]
        
        p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
        poptA = optFn(p0, data, func_ex, pts_l,        
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      verbose=len(params),
                      func_args=func_args,
                      maxiter=maxiter, output_file=logfile,
                      flush_delay=flush_delay, fixed_params=fixed_params,
                      multinom=multinom)
        model = func_ex(poptA, ns, func_args[0], func_args[1], pts_l)  # should be same as at end of optimization
        funcName = 'three_epoch_X1'  # NOTE just for printing
        if multinom:
            ll_opt = dadi.Inference.ll_multinom(model, data)
            thetaOpt = dadi.Inference.optimal_sfs_scaling(model, data)  # only for multinom model
            popt = lrt.getXparams(poptA, funcName)    
            with open(outfile, 'a') as outF:
                outstr = lrt.format1DParams(funcName, popt, theta=thetaOpt, ll_opt=ll_opt)
                outF.write(outstr)
            model.to_file(modelfile)
        else:
            ll_opt = dadi.Inference.ll(model, data)            
            popt = lrt.getXparams(poptA, funcName)    
            with open(outfile, 'a') as outF:
                outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt)
                outF.write(outstr)
            model.to_file(modelfile)

    elif funcName == 'three_epoch_X1_P':   # testing new function which calls three_epoch_X1
        modelType = 'X1'
        func = lrt.three_epoch_X1_P
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]        
        fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
        
        p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
        poptA = optFn(p0, data, func_ex, pts_l,        
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      verbose=len(params),
                      maxiter=maxiter, output_file=logfile,
                      flush_delay=flush_delay, fixed_params=fixed_params,
                      multinom=multinom)
        model = func_ex(poptA, ns, pts_l)  # should be same as at end of optimization

        funcName = 'three_epoch_X1'  # NOTE just for converting
        ll_opt = dadi.Inference.ll(model, data)            
        popt = lrt.getXparams(poptA, funcName)    
        funcName = 'three_epoch_X1_P'
        with open(outfile, 'a') as outF:
            outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt, multinom=False)
            outF.write(outstr)
        model.to_file(modelfile)

    elif funcName == 'three_epoch_X1_P_v2':   # testing new function which has own param converter
        modelType = 'X1'
        func = lrt.three_epoch_X1_P_v2
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]        
        fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
        
        p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
        poptA = optFn(p0, data, func_ex, pts_l,        
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      verbose=len(params),
                      maxiter=maxiter, output_file=logfile,
                      flush_delay=flush_delay, fixed_params=fixed_params,
                      multinom=multinom)
        model = func_ex(poptA, ns, pts_l)  # should be same as at end of optimization
        ll_opt = dadi.Inference.ll(model, data)            
        popt = lrt.three_epoch_convert_params(poptA, modelType, retConstraints=True)   # new converting function
        funcName = 'three_epoch_X1_P'  # for printing. the v2 would replace this        
        with open(outfile, 'a') as outF:
            outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt, multinom=False)
            outF.write(outstr)
        model.to_file(modelfile)
        
    else:
        sys.exit('funcName invalid: {}').format(funcName)



def testFitThreeEpochX0():
    """
    testing new fn to eval X0 and comparing to opt
    """
    outfileA = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.5_0.5_0.5_100000reps.out'.format(homeDir)
    infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.5_0.5_0.5_100000reps.fs'.format(homeDir)
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X0')  # pois lik


def testPoissonSample():
    """
    run for only some fs of interest
    """

    numSamples = 2
    propsList = ['0.5_0.5_0.5']
    for props in propsList:
        infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_100000reps.fs'.format(homeDir, props)
        outBase = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/lrt_test/bottleneck_X_{}'.format(homeDir, props)
        poissonSample(infile, numSamples, outBase)

    
    
def fitThreeEpochX1(outfileA, infile, likType, funcName, optimizer='optimize_log_fmin'):
    """
    older fn with default timescale!! produced non-optimal fits.
    likType: multinomial, poisson
    infile:  chrX fs file
    test of new function lrt:three_epoch_X1
    """

    # param processing 
    if optimizer == 'optimize_log_fmin':
        optFn = dadi.Inference.optimize_log_fmin
        outBase = os.path.split(infile)[0] + '/lrt_test/' + os.path.splitext(os.path.split(infile)[1])[0] 
        
    elif optimizer == 'optimize_log':
        optFn = dadi.Inference.optimize_log
        outBase = os.path.split(infile)[0] + '/lrt_test_optimize_log/' + os.path.splitext(os.path.split(infile)[1])[0]         
    else:
        sys.exit('specified optimizer {} not supported'.format(optimizer))
    
    # read auto params from file
    AfuncName = 'three_epoch'
    popt, ll_opt, thetaA, paramDict = lrt.read1DParams(AfuncName, outfileA)
    
    # fix all but c for this model
    nuB = paramDict['nuB']
    nuF = paramDict['nuF']    
    TB = paramDict['TB']
    TF = paramDict['TF']    

    # optimize chrX
    timescaleFit = 1e-4   # TODO this doese not do anything because I do not use the optimizer functon fit1DModel below. Need to set dadi.Integration.timescale_factor = timescale directly
    numOpts = 10
    minGrid = 150


    if funcName == 'three_epoch_X0':
        params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]
        func = lrt.three_epoch_X0
        fixed_params = array([nuB,nuF,TB,TF,thetaA,0.75])  # c is last and free       #TODO does this work in opt with all fixed params? yes.
    elif funcName == 'three_epoch_X1':
        params = array([nuB,nuF,TB,TF,thetaA,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2]   # only last bound matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1]        
        func = lrt.three_epoch_X1
        fixed_params = array([nuB,nuF,TB,TF,thetaA,None])  # c is last and free
    elif funcName == 'three_epoch_X2':
        params = array([nuB,nuF,TB,TF,thetaA,0.75,0.75])   # starting point for opt
        upper_bound = [1, 10e2, 1, 1, 1e6, 2, 2]   # only last two bounds matters, is for c
        lower_bound = [1e-4, 1e-1, 1e-4, 1e-4, 1e2, 0.1, 0.1]        
        func = lrt.three_epoch_X2
        fixed_params = array([nuB,nuF,TB,TF,thetaA,None,None])  # c1 and c2 are last and free
    else:
        sys.exit('funcName invalid: {}').format(funcName)

    # optimize chrX multiple times
    for optNum in range(numOpts):
        outfile = outBase + '_{}_{}_opt{}.out'.format(likType, funcName, optNum)
        modelfile = outBase + '_{}_{}_opt{}.dadi'.format(likType, funcName, optNum)
        logfile = outfile  # using for both
        maxiter = None
        data = dadi.Spectrum.from_file(infile)
        ns = data.sample_sizes
        pts_l = [minGrid, minGrid+10, minGrid+20]
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        perturb_fold = 1
        p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
        flush_delay = 0.5         # default

        #        poptA = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
        poptA = optFn(p0, data, func_ex, pts_l,        
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      verbose=len(params),
                      maxiter=maxiter, output_file=logfile,
                      flush_delay=flush_delay, fixed_params=fixed_params,
                      multinom=False)
        # wrong -- this function is defined in terms of c1, c2, and the fixed auto params. either eval the same func with poptA or eval three_epoch function defined with out constraint with chrX param ests
        # popt = lrt.getXparams(poptA, funcName)    
        # model = func_ex(popt, ns, pts_l)      

        model = func_ex(poptA, ns, pts_l)  # should be same as at end of optimization
        ll_opt = dadi.Inference.ll(model, data)

        # write out params in terms of chrX
        popt = lrt.getXparams(poptA, funcName)            
        with open(outfile, 'a') as outF:
            outstr = lrt.format1DParams(funcName, popt, theta=None, ll_opt=ll_opt)
            outF.write(outstr)
        model.to_file(modelfile)



def runFitThreeEpochX0():
    """
    note: function is fitThreeEpochX1 but any of X0, X1, or X2 can be fit
    """
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.5_0.5_0.5_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.5_0.5_0.5_100000reps.fs'
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X0')  # pois lik
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X1')  # should not be sig improvement

    
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.7_0.7_0.7_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.7_0.7_0.7_100000reps.fs'
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X0')   # shoudl be sig worse than X1
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X1')   # shoudl be sig worse than X1
    
    
def runFitThreeEpochX1():
    """
    necessary
    """
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.7_0.7_0.7_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.7_0.7_0.7_100000reps.fs'

    fitThreeEpochX1(outfileA, infile, 'pois')

    
def runFitThreeEpochX2(fnArgs):
    if fnArgs is None:   # NOTE
        optimizer = 'optimize_log_fmin'        
    else:
        optimizer = fnArgs[0]
                           
    
    #-------------------------------------------------------#    
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.8_0.2_0.8_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.8_0.2_0.8_100000reps.fs'
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X0')
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X1')    
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X2', optimizer)   # should fit better than X0 or X1

    #-------------------------------------------------------#    
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.5_0.5_0.5_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.5_0.5_0.5_100000reps.fs'
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X2', optimizer)  # should not be sig improvement over X0 or X1

    #-------------------------------------------------------#       
    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.7_0.7_0.7_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.7_0.7_0.7_100000reps.fs'
    fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X2', optimizer)   # should not be sig improvement over X1



def runFitThreeEpoch(fnArgs):
    """
    running all models X0, X1, X2 for considered fs
    outfileA: output of fitting to autosomal fs
    infile:   chrX fs input file
    """

    if fnArgs is None:   # NOTE
        optimizer = 'optimize_log_fmin'        
    else:
        optimizer = fnArgs[0]
                           
    propsList = ['0.5_0.5_0.5', '0.8_0.8_0.8', '0.8_0.2_0.8', '0.7_0.7_0.7']
    for props in propsList:
        outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_{}_100000reps.out'.format(props)
        infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_100000reps.fs'.format(props)
        fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X0', optimizer)
        fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X1', optimizer)
        fitThreeEpochX1(outfileA, infile, 'pois', 'three_epoch_X2', optimizer)


def test_three_epoch_X():
    """
    <2015-10-23 Fri>    
    """

    # choose sim where X1 is the best model: p is all 0.7

    outfileA = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.7_0.7_0.7_100000reps.out'
    infile = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.7_0.7_0.7_100000reps.fs'
    optimizer = 'optimize_log_fmin'        

    # fit with multi and constraint
    outBase = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/test_three_epoch_X/bottleneck_X_0.7_0.7_0.7'
    fitThreeEpoch(outfileA, infile, 'multinom', 'three_epoch_X1_all', optimizer, outBase, multinom=True)
    # fit with pois and constraint
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X1_all', optimizer, outBase, multinom=False)    
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X1_P', optimizer, outBase, multinom=False)    # should be almost identical to output from fn above this one
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X1_P_v2', optimizer, outBase, multinom=False)    # improved code. should be identical to the one above. these are unit tests


    
def runFitThreeEpochBootstrap(fnArgs):
    """
    Updated version of runFitThreeEpoch: different args and uses default optimizer
        	outfileA: output of auto model fit with param ests at end
    results get put in same dir as infile
    """

    props, bootNum = fnArgs
    
    outfileA = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_{}_100000reps.out'.format(homeDir, props)
    infile = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_boot_{}.fs'.format(homeDir, props, bootNum)

    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X0')
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X1')
    fitThreeEpoch(outfileA, infile, 'pois', 'three_epoch_X2')

    timescaleFit = 1e-4
    resBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/lrt_test/bottleneck_X_{}_boot_{}'.format(homeDir, props, bootNum)  # TODO make this have three_epoch as funcName so has same format as others
    fitBottleneck(infile, isCluster=True, optNum=None, timescaleFit=timescaleFit, minGrid=150, resBase=resBase)   # fits a bottleneck model, unconstrained, to chrX fs
    
    # TODO need free opt: ?


def runPoissonSampleV1():
    """
    run for only some fs of interest
    """

    numSamples = 100
    propsList = ['0.5_0.5_0.5', '0.8_0.8_0.8', '0.8_0.2_0.8']
    for props in propsList:
        infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_100000reps.fs'.format(homeDir, props)
        outBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}'.format(homeDir, props)
        poissonSample(infile, numSamples, outBase)

        
def runPoissonSampleV2():
    """
    do for all fs so far. can loop over input fs so don't need to generate names?
    """
    numSamples = 100
    propsList = allProps   # global
    for props in propsList:
        infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_100000reps.fs'.format(homeDir, props)
        outBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}'.format(homeDir, props)
        poissonSample(infile, numSamples, outBase)

def runPoissonSampleAutoV2():
    """
    do for all fs so far. can loop over input fs so don't need to generate names?
    """
    numSamples = 100
    propsList = allProps   # global
    for props in propsList:
        infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_{}_100000reps.fs'.format(homeDir, props)
        outBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/auto_samples/bottleneck_A_{}'.format(homeDir, props)
        poissonSample(infile, numSamples, outBase)

        
    
def writeCmdsBootstrapV1(outfile):
    numSamples = 100
    propsList = ['0.5_0.5_0.5', '0.8_0.8_0.8', '0.8_0.2_0.8']

    with open(outfile, 'w') as outF:
        for props in propsList:
            for i in range(numSamples):
                cmdstr = '{} {}\n'.format(props, i)
                outF.write(cmdstr)
                
def writeCmdsBootstrapV2(outfile):
    numSamples = 100
    propsList = allProps
    with open(outfile, 'w') as outF:
        for props in propsList:
            for i in range(numSamples):
                cmdstr = '{} {}\n'.format(props, i)
                outF.write(cmdstr)

                

def collectResultsV1(fnArgs):
    """
    collects results of optimzations for first set of files run
    prints out header then estiamted parameters and file name
    funcName is either three_epoch_X0, three_epoch_X1, three_epoch_X2, or three_epoch
    """

    funcName, outfile = fnArgs
    likType = 'pois'
    if funcName == 'three_epoch_X0' or funcName == 'three_epoch_X1':
        header = 'props bootNum nuB nuF TB TF theta c ll_opt file\n'
    elif funcName == 'three_epoch_X2':
        header = 'props bootNum nuB nuF TB TF theta c1 c2 ll_opt file\n'
    else:    # free fit three_epoch
        header = 'props bootNum nuB nuF TB TF theta ll_opt file\n'

    propsList = ['0.5_0.5_0.5', '0.8_0.8_0.8', '0.8_0.2_0.8']
    numSamples = 100
    with open(outfile, 'w') as outF:
        outF.write(header)
        for props in propsList:
            for bootNum in range(numSamples):
                resBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/lrt_test/bottleneck_X_{}_boot_{}'.format(homeDir, props, bootNum)                
                if funcName == 'three_epoch':
                    outfile = resBase + '.out'    # free fit                
                else:
                    outfile = resBase + '_{}_{}.out'.format(likType, funcName)    # all other models
                if not os.path.isfile(outfile):   # if outfile not there bc not generated, skip
                    continue

                if funcName == 'three_epoch':                
                    popt, ll_opt, theta, paramDict, paramLine = lrt.read1DParams(funcName, outfile, retParamLine=True, likType='multinom')
                    if paramLine is None:
                        continue
                else:
                    popt, ll_opt, theta, paramDict, paramLine = lrt.read1DParams(funcName, outfile, retParamLine=True, likType='pois')
                    if paramLine is None:
                        continue
                outarr = [props, str(bootNum), paramLine, outfile]
                outstr = ' '.join(outarr) + '\n'
                outF.write(outstr)
    

def collectResultsV2(fnArgs):
    """
    collects results of optimzations for second set of files
    prints out header then estiamted parameters and file name
    funcName is either three_epoch_X0, three_epoch_X1, three_epoch_X2, or three_epoch
    """

    funcName, outfile = fnArgs
    likType = 'pois'
    if funcName == 'three_epoch_X0' or funcName == 'three_epoch_X1':
        header = 'props bootNum nuB nuF TB TF theta c ll_opt file\n'
    elif funcName == 'three_epoch_X2':
        header = 'props bootNum nuB nuF TB TF theta c1 c2 ll_opt file\n'
    else:    # free fit three_epoch
        header = 'props bootNum nuB nuF TB TF theta ll_opt file\n'

    propsList = allProps
    numSamples = 100
    with open(outfile, 'w') as outF:
        outF.write(header)
        for props in propsList:
            for bootNum in range(numSamples):
                resBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/lrt_test/bottleneck_X_{}_boot_{}'.format(homeDir, props, bootNum)                
                if funcName == 'three_epoch':
                    outfile = resBase + '.out'    # free fit                
                else:
                    outfile = resBase + '_{}_{}.out'.format(likType, funcName)    # all other models
                if not os.path.isfile(outfile):   # if outfile not there bc not generated, skip
                    continue

                if funcName == 'three_epoch':                
                    popt, ll_opt, theta, paramDict, paramLine = lrt.read1DParams(funcName, outfile, retParamLine=True, likType='multinom')
                    if paramLine is None:
                        continue
                else:
                    popt, ll_opt, theta, paramDict, paramLine = lrt.read1DParams(funcName, outfile, retParamLine=True, likType='pois')
                    if paramLine is None:
                        continue
                outarr = [props, str(bootNum), paramLine, outfile]
                outstr = ' '.join(outarr) + '\n'
                outF.write(outstr)

def QtoP(Q):
    p = (16. * Q - 9.) / (8. * Q)
    return p
                

def estQpiV2(outfile):
    """
    estimates pi from the bootstrap fs and writes to file
    auto in a different dir than chrX
    """
    likType = 'pois'
    propsList = allProps
    numSamples = 100
    header = 'props bootNum piA piX Qpi ppi\n'
    
    with open(outfile, 'w') as outF:
        outF.write(header)
        for props in propsList:
            for bootNum in range(numSamples):

                # read in chrX
                outBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}'.format(homeDir, props)
                fsfileX = '{}_boot_{}.fs'.format(outBase, bootNum)                

                # read in auto                
                outBase = '{}/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/auto_samples/bottleneck_A_{}'.format(homeDir, props)
                fsfileA = '{}_boot_{}.fs'.format(outBase, bootNum)                                
                if not os.path.isfile(fsfileX) or not os.path.isfile(fsfileA):
                    continue

                # estimate pi from each and calculate ratio, write to file
                fsX = dadi.Spectrum.from_file(fsfileX)
                piX = fsX.pi()
                fsA = dadi.Spectrum.from_file(fsfileA)
                piA = fsA.pi()
                Qpi = piX / piA
                ppi = QtoP(Qpi)
                outarr = [props, str(bootNum), str(piA), str(piX), str(Qpi), str(ppi)]
                outstr = ' '.join(outarr) + '\n'
                outF.write(outstr)
                
    
def plotDataThreeModels():
    """
    """
    figdir = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/figures'
    datadir = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x'


    ## data sim's under X2
    datafile = '{}/bottleneck_X_0.8_0.2_0.8_100000reps.fs'.format(datadir)
    data = dadi.Spectrum.from_file(datafile)

   # X2
    modelName = 'X2'
    modelfile = '{}/lrt_test/bottleneck_X_0.8_0.2_0.8_100000reps_pois_three_epoch_X2_opt6.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.8_0.2_0.8_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X2 model', 'data'), outfile=outfile, main=modelName)

    # all free
    modelName = 'free'
    modelfile = '{}/bottleneck_X_0.8_0.2_0.8_100000reps_opt9.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.8_0.2_0.8_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('free model', 'data'), outfile=outfile, main=modelName)


   # X1
    modelName = 'X1'   
    modelfile = '{}/lrt_test/bottleneck_X_0.8_0.2_0.8_100000reps_pois_three_epoch_X1_opt8.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.8_0.2_0.8_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X1 model', 'data'), outfile=outfile, main=modelName)

   # X0
    modelName = 'X0'
    modelfile = '{}/lrt_test/bottleneck_X_0.8_0.2_0.8_100000reps_pois_three_epoch_X0_opt5.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.8_0.2_0.8_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X0 model', 'data'), outfile=outfile, main=modelName)


    #### data sim's under X0
    datafile = '{}/bottleneck_X_0.5_0.5_0.5_100000reps.fs'.format(datadir)
    data = dadi.Spectrum.from_file(datafile)

   # X0
    modelName = 'X0'
    modelfile = '{}/lrt_test/bottleneck_X_0.5_0.5_0.5_100000reps_pois_three_epoch_X0_opt5.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.5_0.5_0.5_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X0 model', 'data'), outfile=outfile, main=modelName)
    
   # X1
    modelName = 'X1'   
    modelfile = '{}/lrt_test/bottleneck_X_0.5_0.5_0.5_100000reps_pois_three_epoch_X1_opt1.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.5_0.5_0.5_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X1 model', 'data'), outfile=outfile, main=modelName)


    # all free
    modelName = 'free'
    modelfile = '{}/bottleneck_X_0.5_0.5_0.5_100000reps_opt6.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.5_0.5_0.5_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('free model', 'data'), outfile=outfile, main=modelName)


    #### data sim's under X1
    datafile = '{}/bottleneck_X_0.7_0.7_0.7_100000reps.fs'.format(datadir)
    data = dadi.Spectrum.from_file(datafile)

   # X0
    modelName = 'X0'
    modelfile = '{}/lrt_test/bottleneck_X_0.7_0.7_0.7_100000reps_pois_three_epoch_X0_opt0.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.7_0.7_0.7_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X0 model', 'data'), outfile=outfile, main=modelName)
    
   # X1
    modelName = 'X1'   
    modelfile = '{}/lrt_test/bottleneck_X_0.7_0.7_0.7_100000reps_pois_three_epoch_X1_opt4.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.7_0.7_0.7_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('X1 model', 'data'), outfile=outfile, main=modelName)


    # all free
    modelName = 'free'
    modelfile = '{}/bottleneck_X_0.7_0.7_0.7_100000reps_opt1.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_X_0.7_0.7_0.7_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('free model', 'data'), outfile=outfile, main=modelName)
    

    
def plotVaryScaling():
    """
    2015-09-11: vary scaling of ms sims by Nanc and Ncurr
    """
    figdir = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/figures'
    datadir = '/Users/shaila/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x'

    datafile = '{}/bottleneck_A_0.8_0.2_0.8_100000reps.fs'.format(datadir)
    data = dadi.Spectrum.from_file(datafile)

    # all free
    modelName = 'free'
    modelfile = '{}/bottleneck_A_0.8_0.2_0.8_100000reps_opt4.dadi'.format(datadir)
    model = dadi.Spectrum.from_file(modelfile)
    outfile = '{}/bottleneck_A_0.8_0.2_0.8_model_{}_vs_data.pdf'.format(figdir, modelName)
    myPlot.plot_1d_comp_multinom(model, data, fs_labels=('free model', 'data'), outfile=outfile, main=modelName)

    
    

                
def write1000xFs(infile):
    """
    loop over files, read in fs, and write scaled ones to a new dir
    """
    outdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/'.format(homeDir)
    for line in open(infile):
        origFile = line.strip()
        newFile = outdir + os.path.split(origFile)[1]
        fs = dadi.Spectrum.from_file(origFile)
        newFs = fs * 1000
        dadi.Spectrum.to_file(newFs, newFile)
        

def fitBottleneckImproved(infile):
    """
    improved fitting: timestep, scale fs by 1000, lgr min grid, opt mult times
    """
    timescaleFit = 1e-4
    numOpts = 10
    for optNum in range(numOpts):
        fitBottleneck(infile, isCluster=True, optNum=optNum, timescaleFit=timescaleFit, minGrid=150)

def testFitBottleneckImproved():
    infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_0.1_0.9_0.1_100000reps.fs'.format(homeDir)
    fitBottleneckImproved(infile)


def testFitBottleneck():
    infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks/bottleneck_A_0.1_0.9_0.1_100000reps.fs'.format(homeDir)
    fitBottleneck(infile, isCluster=False)


def fitBottleneck(infile, isCluster=True, optNum=None, timescaleFit=None, minGrid=None, resBase=None):
    """
    calls lrt.fit1DModel with filenames and defaults
    """

    # constants used in simulating: will need to convert params
    L = 1e4
    numSamples = 100

    # file names
    funcName = 'three_epoch'
    if resBase is None:
        resBase = os.path.splitext(infile)[0]
    if optNum:   # TODO this does not match for optNum 0
        outfile = resBase + '_opt{}.out'.format(optNum)
        modelfile = resBase + '_opt{}.dadi'.format(optNum)
    else:
        outfile = resBase + '.out'
        modelfile = resBase + '.dadi'

    # nuB,nuF,TB,TF = params
    ## used for ms_bottleneck
    # upper_bound = [1, 10e4, 1, 1]
    # lower_bound = [1e-8, 1e-8, 1e-8, 1e-8]
    # params = array([0.1, 5, 0.1, 0.01])   

    ## used in ms_bottleneck_1000x
    params = array([0.128270, 6.896289, 0.038618, 0.031725])   # recent bottleneck, true params for auto. 
    upper_bound = [1, 10e2, 1, 1]
    lower_bound = [1e-4, 1e-1, 1e-4, 1e-4]
    maxiter = None
    if minGrid is None:
        minGrid = numSamples + 10
    popt, ll_opt, theta = lrt.fit1DModel(infile, outfile, modelfile, funcName, isCluster=isCluster, minGrid=minGrid, maxiter=maxiter, perturb_fold=1, lower_bound=lower_bound, upper_bound=upper_bound, params=params, fixed_params=None, logfile=None, timescale=timescaleFit)

    

def estPhat():
    """
    rough estimate: need logfile output
    """
    L = 1e4
    mu = 1.5e-8
    numReps = int(1e5)          # can go up to 1e6 for better fs
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint

    for pfConstant in pfVals:
        for pfBottle in pfVals:
            propFemales = array( [pfConstant, pfBottle, pfConstant] )
            propList = '_'.join([str(x) for x in propFemales])
            resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
            for chromType in chromTypes:
                resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
                fsfile = resBase + '.fs'
                fs = dadi.Spectrum.from_file(fsfile)
                fsDict[chromType] = fs
            outfile = '{}/figures/bottleneck_compXvsA_{}_{}reps.pdf'.format(resdir, propList, numReps)



def plotXvsAfs():
    """
    for a given set of proportions of female, plot X and A fs on same axes
    """

    machType = 'shimac'           # determines paths
    numReps = int(1e5)          # can go up to 1e6 for better fs
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint
    fsDict = {}

    for pfConstant in pfVals:
        for pfBottle in pfVals:
            propFemales = array( [pfConstant, pfBottle, pfConstant] )
            propList = '_'.join([str(x) for x in propFemales])
            resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
            for chromType in chromTypes:
                resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
                fsfile = resBase + '.fs'
                fs = dadi.Spectrum.from_file(fsfile)
                fsDict[chromType] = fs
            outfile = '{}/figures/bottleneck_compXvsA_{}_{}reps.pdf'.format(resdir, propList, numReps)
            main = 'bottleneck_compXvsA_{}_{}reps'.format(propList, numReps)
            myPlot.plot_1d_comp_multinom(fsDict['A'], fsDict['X'], fs_labels=('Auto', 'chrX'), outfile=outfile, main=main)


def plotXpropBottle():
    """
    for a given persistent pF, plot chrX fs on same axes
    """

    machType = 'shimac'           # determines paths
    numReps = int(1e5)          # can go up to 1e6 for better fs
    chromTypes = ['A', 'X']     # is A or X; determines reduction factor
    pfVals = numpy.arange( 0.1, 1.0, 0.1)   # does not contain right endpoint
    chromType = 'X'
    colorList =['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#565656', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221']         # python set of nine colors going from pink to green, dark gray in middle instead of white '#f7f7f7', 

    for pfConstant in pfVals:
        fsList = []
        labelList = []
        for pfBottle in pfVals:
            propFemales = array( [pfConstant, pfBottle, pfConstant] )
            propList = '_'.join([str(x) for x in propFemales])
            resdir = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks'.format(homeDir)
            resBase = '{}/bottleneck_{}_{}_{}reps'.format(resdir, chromType, propList, numReps)
            fsfile = resBase + '.fs'
            fs = dadi.Spectrum.from_file(fsfile)
            fsList.append(fs)
            labelList.append(str(pfBottle))    # could also round


        main = 'bottleneck_compXpropBottle_{}_{}reps'.format(propList, numReps)
        outfile = '{}/figures/bottleneck_compXpropBottle_{}_{}reps_counts.pdf'.format(resdir, propList, numReps)
        f.plotSFSlist(fsList, labelList, colorList, plotFractions=False, outfile=outfile, main=main)
        outfile = '{}/figures/bottleneck_compXpropBottle_{}_{}reps_density_semilogx.pdf'.format(resdir, propList, numReps)
        f.plotSFSlist(fsList, labelList, colorList, plotFractions=True, outfile=outfile, semilogType='x', main=main)



def runFIMv2():
    """
    on bootstrap
    """
    props = '0.8_0.2_0.8'
    funcName = 'three_epoch_X1'  # not best-fitting and correct model for this file: X1 is
    infile = '/home/shailam/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_0.8_0.2_0.8_boot_45.fs'
    outfileX = '/home/shailam/projects/hgdp-x/data/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/lrt_test/bottleneck_X_0.8_0.2_0.8_boot_45_pois_three_epoch_X1.out'
    outfileA = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_{}_100000reps.out'.format(homeDir, props)

    
    # nuB nuF TB TF theta c ll_opt
    # 0.122528 7.137566 0.072057 0.063519 4382.559884 0.791042 -410.35852464141260

    # fitting params
    likType = 'pois'
    timescale = 1e-4
    dadi.Integration.timescale_factor = timescale
    minGrid = 150
    pts_l = [minGrid, minGrid+10, minGrid+20]

    # chrX fs
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    if funcName == 'three_epoch_X0':     # do not opt
        func = lrt.three_epoch_X0
    elif funcName == 'three_epoch_X1':
        func = lrt.three_epoch_X1            
    elif funcName == 'three_epoch_X2':
        func = lrt.three_epoch_X2
    else:
        sys.exit('funcName invalid: {}').format(funcName)
    func_ex = dadi.Numerics.make_extrap_log_func(func)

    poptX, ll_optX, thetaX, paramDictX, paramLineX = lrt.read1DParams(funcName, outfileX, retParamLine=True, likType=likType)   
    AfuncName = 'three_epoch'
    poptA, ll_optA, thetaA, paramDictA = lrt.read1DParams(AfuncName, outfileA)
    
    cX = float(paramLineX.split(' ')[5])    # just one for X1
    poptOrig = poptA + [thetaA, float(cX)]

    # check - same as from file!! bc matched timescale
    model_test = func_ex(poptOrig, ns, pts_l)
    ll_test = dadi.Inference.ll(model_test, data)

    # estimate FIM
    uncerts_fim = dadi.Godambe.FIM_uncert(func_ex, pts_l, poptOrig, data, multinom=False)

    
    
        
def runFIM(infile, outfileA, funcName):
    """
    TODO something odd here with opt and getting same log likelihood -- timescale was different.
    
    run FIM to get CIs
    infile: original fs input file
    """
    
    props = '0.8_0.2_0.8'
    funcName = 'three_epoch_X1'  # not best-fitting and correct model for this file: X1 is
    infile = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_X_{}_100000reps.fs'.format(homeDir, props)
    outfileA = '{}/projects/hgdp-x/results/2015-02-23_male-biased-OOA/ms_bottlenecks_1000x/bottleneck_A_{}_100000reps.out'.format(homeDir, props)
    
    optNum = 0  # hard-coded, all fits were good
    outBase = os.path.split(infile)[0] + '/lrt_test/' + os.path.splitext(os.path.split(infile)[1])[0]   # still in resdir
    outfileX = outBase + '_{}_{}_opt{}.out'.format(likType, funcName, optNum)
    
    # fitting params
    likType = 'pois'
    # timescale = 1e-4
    timescale = 1e-3  # default
    dadi.Integration.timescale_factor = timescale
    minGrid = 150
    pts_l = [minGrid, minGrid+10, minGrid+20]

    # func name
    if funcName == 'three_epoch_X0':     # do not opt
        func = lrt.three_epoch_X0
    elif funcName == 'three_epoch_X1':
        func = lrt.three_epoch_X1            
    elif funcName == 'three_epoch_X2':
        func = lrt.three_epoch_X2
    else:
        sys.exit('funcName invalid: {}').format(funcName)
    func_ex = dadi.Numerics.make_extrap_log_func(func)

    # chrX fs
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
     
    # read best fit from chrX file
    # chrX data: only paramLine working, not popt or theta
    poptX, ll_optX, thetaX, paramDictX, paramLineX = lrt.read1DParams(funcName, outfileX, retParamLine=True, likType=likType)   

    # get FIM estimates: use my custom function. needs the correct value of c. here I have chrX params in front... matters? yes
    # I only opt c and c1 or c2 in my custom functions. the rest are fixed params from the auto opt. so I should be able to read those auto params from file, and append the c value to get poptA returned by the optimizer

    AfuncName = 'three_epoch'
    poptA, ll_optA, thetaA, paramDictA = lrt.read1DParams(AfuncName, outfileA)

    # In [67]: poptAuto
    # Out[67]: [0.122528, 7.137566, 0.057, 0.050246]

    # In [68]: popt
    # Out[68]: [0.122528, 7.137566, 0.072194, 0.06364, 4374.240242]

    # In [69]: paramLine
    # Out[69]: '0.122528 7.137566 0.072194 0.063640 4374.240242 0.789541 -445.66593717607969'
    # end of chrX file
    # 2268    , -445.666    , array([ 0.122528   ,  7.13757    ,  0.057      ,  0.050246   ,  5540.23    ,  0.789425   ])
    # nuB nuF TB TF theta c ll_opt
    # 0.122528 7.137566 0.072194 0.063640 4374.240242 0.789541 -445.66593717607969

    # end of auto file
    # 196     , -327.447    , array([ 0.122525   ,  7.13802    ,  0.0569937  ,  0.0502462  ])
    # nuB nuF TB TF theta ll_opt
    # 0.122528 7.137566 0.057000 0.050246 5540.234764 -327.44663763911751

    # In [102]: poptForFn
    # Out[102]: [0.122528, 7.137566, 0.057, 0.050246, 5540.234764, 0.789541]
    
    
    cX = float(paramLineX.split(' ')[5])    # just one for X1
    thetaX = float(paramLineX.split(' ')[4])    #     
    poptOrig = poptA + [thetaA, float(cX)]

    # check ll: not getting corect. work on popt.  -366.13489295737622
    model_test = func_ex(poptOrig, ns, pts_l)
    ll_test = dadi.Inference.ll(model_test, data)

    # nope: need auto params to begin.  -853.98437200728029
    poptXFull = poptX + [cX]
    model_test2 = func_ex(poptXFull, ns, pts_l)
    ll_test2 = dadi.Inference.ll(model_test2, data)

    # try three_epoch vanilla version for chrX: -365.9429060687421
    funcA = dadi.Demographics1D.three_epoch
    funcA_ex = dadi.Numerics.make_extrap_log_func(funcA)    
    modelA = funcA_ex(poptX[:4], ns, pts_l)
    ll_testA = dadi.Inference.ll_multinom(modelA, data)

    # try three_epoch poisson version: -366.13501082711082
    funcX = lrt.three_epoch_fixed_theta
    funcX_ex = dadi.Numerics.make_extrap_log_func(funcX)    
    modelX = funcX_ex( [poptX[4]] + poptX[:4], ns, pts_l)
    ll_testX = dadi.Inference.ll(modelX, data)
    
    
    uncerts_fim = dadi.Godambe.FIM_uncert(func_ex, pts_l, poptForFn, data, multinom=False)

    uncerts_fim = dadi.Godambe.FIM_uncert(func_ex, pts_l, poptOrig, data, multinom=False)

    uncerts_fim = dadi.Godambe.FIM_uncert(func_ex, pts_l, poptA, data, multinom=False)    
    
    # TODO get FIM estimates: use three_epoch. multi vesrsion - get same CI? also a pois version
    # lrt def three_epoch_fixed_theta(params, ns, pts):
