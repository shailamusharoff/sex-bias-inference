import dadi
import pylab
import os
import sys
import numpy as np
from numpy import array

def plotSFS(fsA, fsX, labelA, labelX, plotFractions, outfile=None, projDim=None, semilog=False):
    if projDim is not None:
        fsAOrig = fsA; fsXOrig = fsX
        fsA = dadi.Spectrum.project(fsAOrig, [projDim])
        fsX = dadi.Spectrum.project(fsXOrig, [projDim])

    fig = pylab.gcf()
    pylab.clf()
    ax = pylab.subplot(1,1,1)
    if semilog:
        plotFn = pylab.semilogy
    else:
        plotFn = pylab.plot
    if plotFractions:
        plotFn(fsA/fsA.S(), '-ob', ms=0.5, label=labelA)
        plotFn(fsX/fsX.S(), '-or', ms=0.5, label=labelX)
    else:
        plotFn(fsA, '-ob', ms=0.5, label=labelA)
        plotFn(fsX, '-or', ms=0.5, label=labelX)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size':8})
    maxDim = max(fsX.shape[0], fsA.shape[0])
    ax.set_xlim(0, maxDim-1)
    if outfile is not None:
        fig.savefig(outfile, dpi=200)
    pylab.show()


def plotSFSlist(fsList, labelList, colorList, plotFractions, outfile=None, projDim=None, semilogType=None, openFiles=False, main=None):
    """
    plot mult SFS based on list inputs. projDim is an integer
    projDim: an integer (not a list). project each fs to this dim
    semilogType = x | y | None | both
    openFiles: boolean. if True: fsList is a list of fileanames so open and read in first
    Currently only calls pylab.show if outfile is None

    """
    numFs = len(fsList)
    if len(labelList) != numFs or len(colorList) != numFs:
        sys.exit('Error: fsList, labelList, and colorList must all be the same length')

    if openFiles:
        fsFileList = fsList
        fsList = []
        for fsFile in fsFileList:
            fs = dadi.Spectrum.from_file(fsFile)
            fsList.append(fs)

    if projDim is not None:         # project each fs to projDim
        fsListOrig = fsList         # necessary?
        fsList = []
        for fs in fsListOrig:
            fsOrig = fs
            fsProj = dadi.Spectrum.project(fsOrig, [projDim])
            fsList.append(fsProj)

    fig = pylab.gcf()
    pylab.clf()
    ax = pylab.subplot(1,1,1)
    if semilogType == "y":
        plotFn = pylab.semilogy
    elif semilogType == "x":
        plotFn = pylab.semilogx
    elif semilogType == 'both':
        plotFn = pylab.loglog
    else:
        plotFn = pylab.plot

    for i in range(numFs):   # todo finish this
        fs = fsList[i]
        fsColor = colorList[i]
        fsLabel = labelList[i]
        if plotFractions:
            plotFn(fs/fs.S(), '-o', color=fsColor, ms=0.9, label=fsLabel, lw=1.2)
        else:
            plotFn(fs, '-o', color=fsColor, ms=0.9, label=fsLabel, lw=1.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size':10})


    if main is not None:
        pylab.suptitle(main)
    if outfile is not None:
        fig.savefig(outfile, dpi=200)
    else:
        pylab.show()


def plotMultSFS(plotFractions, chrom, semilog, siteTypes):
    """
    plot all auto SFS on one axis: counts if specified
    seems to be specific to one data structure?
    """
    if plotFractions == False:
        outfile = 'figures/sfs_chr{}_counts.png'.format(chrom)
    else:
        outfile = 'figures/sfs_chr{}_density.png'.format(chrom)
    colorList = ['green', 'red', 'blue', 'purple']
    fig = pylab.gcf()
    pylab.clf()
    ax = pylab.subplot(1,1,1)
    if semilog:
        plotFn = pylab.semilogy
    else:
        plotFn = pylab.plot
    for i in range(len(siteTypes)):
        siteType = siteTypes[i]
        fsColor = colorList[i]
        fs = fsDict['YRI'][chrom][siteType]
        if plotFractions:
            plotFn(fs/fs.S(), '-o', color=fsColor, ms=0.5, label=siteType)
        else:
            plotFn(fs, '-o', color=fsColor, ms=0.5, label=siteType)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size':8})


    if outfile is not None:
        fig.savefig(outfile, dpi=200)
    else:
        pylab.show()

def snmAnalytical(numChroms, theta=1):
    """
    Calculate expected SFS under the standard neutral model
    """
    sfsW1 = [1./i for i in range(1,numChroms)]
    sfsW1.insert(0, 0)
    sfsW1.append(0)
    sfsW1 = dadi.Spectrum(sfsW1, mask_corners=True)
    sfsW1_scaled = sfsW1 * theta
    return(sfsW1_scaled)


def fit_three_epoch(infile, outfile, modelfile=None, maxiter=100):
    """
    might now be obsolete: see dadiLrtFunctions:fitThreeEpoch, fit1DModel
    """
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [100, 110, 120]    ## larger grid set on 2014-03-06
    func = dadi.Demographics1D.three_epoch


    # from manual: params = (nuB,nuF,TB,TF)
    # nuB = Nbottle / Nancient
    # nuF = Ncurrent / Nancient
    # TB: length of bottle in 2*Nanc gens
    # TF: time since bottle recovery in 2*Nanc gens
    upper_bound = [1, 10, 1, 1]
    # true params for this SFS: [0.01111, 1, 6.25e-4, 0.014]
    params = array([0.001, 5, 0.1, 0.1])
    lower_bound = [1e-8, 1e-8, 1e-8, 1e-8]

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    ll_model = dadi.Inference.ll_multinom(model, data)
    p0 = dadi.Misc.perturb_params(params, fold=2, lower_bound=lower_bound, upper_bound=upper_bound)
    popt = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            verbose=len(params),
                                            maxiter=maxiter, output_file = outfile)
    model = func_ex(popt, ns, pts_l)
    ll_opt = dadi.Inference.ll_multinom(model, data)
    theta = dadi.Inference.optimal_sfs_scaling(model, data)

    if modelfile is not None:
        modelfile = open(modelfile, 'w')         # write SFS expected under demographic model and params to file
        model.to_file(modelfile)
        modelfile.close()

    outfile = open(outfile, 'a')
    outfile.write('Optimized parameters ' + repr(popt) + '\n')
    outfile.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(theta) + '\n')
    outfile.close()
    return (popt, ll_opt, theta)
