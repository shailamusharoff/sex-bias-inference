## dadi functions to carry out likelihood ratio test on a set of nested models
# /Users/shaila/projects/adglm/results/sb/scripts/dadiLrtFunctions.py

from optparse import OptionParser
import random
import numpy
from numpy import array
import dadi
import os
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
import sys
import math
import json
import pythonUtils as pyUtils
import pickle
import pprint
import bisect

# global
homeDir = os.path.expanduser('~')

#------------------ custom demographic functions ------------------#

def two_epoch_fixed_theta(params, ns, pts):
    """
    Instantaneous size change some time ago.

    params = (theta1, nu,T)
    ns = (n1,)   # this must be a list
    theta1: 4 Nref mu of the reference population
    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which size change happened (in units of 2*Na
       generations)
    n1: Number of samples in resulting Spectrum
    pts: Numbeur of grid points to use in integration.
    """
    theta1, nu,T = params

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, theta0=theta1)

    phi = dadi.Integration.one_pop(phi, xx, T, nu, theta0=theta1)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx,))
    return fs

def three_epoch_fixed_theta(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Time since bottleneck recovery (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    theta1, nuB,nuF,TB,TF = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta1)

    phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta1)
    phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta1)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def bottlegrowth_fixed_theta(params, ns, pts):
    """
    Instantanous size change followed by exponential growth.

    params = (nuB,nuF,T,theta1)
    ns = (n1,)

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contemporary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,T,theta1 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta1)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func, theta0=theta1)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def twoEpochGrowth(params, ns, pts):
    """
    Two size epochs followed by exponential growth.

    Update to bottlegrowth because growth starts at the same time the second size is reached. This function allows for a first size for some time, a second size for some time, and then exp growth
    Can be constrained as a bottleneck without recovery followed by exponential growth in optimization by setting the upper limit on the second epoch to be 1.

    params = (nuB, nuF, TB, TF)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary size to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Time in the past when growth began (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nuB,nuF,TB,TF = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)

    # exponential growth from size nuB to size nuF in TF generations
    nu_func = lambda t: nuB * numpy.exp(numpy.log(nuF / nuB) * t/TF)
    phi = Integration.one_pop(phi, xx, TF, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def threeEpochGrowth(params, ns, pts):
    """
    Three size epochs followed by exponential growth.

    Update to bottlegrowth because that only has two epochs before expontential growth.
    Can be constrained as a bottleneck followed by exponential growth in optimization by setting the upper limit on the second epoch to be 1.

    params = (nuB, nuF, nuC, TB, TF, TC)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of post-bottleneck size to ancient pop size
    nuC: Ratio of contemporary size to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Length of time of post-bottleneck size (in units of 2*Na generations)
    TC: Time in the past when growth began (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nuB,nuF,nuC,TB,TF,TC = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    # exponential growth from size nuF to size nuC in TC generations
    nu_func = lambda t: nuF * numpy.exp(numpy.log(nuC / nuF) * t/TC)
    phi = Integration.one_pop(phi, xx, TC, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def threeEpochGrowth_P(params, ns, pts):
    """
    Three size epochs followed by exponential growth. Fixed theta.

    Update to bottlegrowth because that only has two epochs before expontential growth.
    Can be constrained as a bottleneck followed by exponential growth in optimization by setting the upper limit on the second epoch to be 1.

    params = (nuB, nuF, nuC, TB, TF, TC)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of post-bottleneck size to ancient pop size
    nuC: Ratio of contemporary size to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Length of time of post-bottleneck size (in units of 2*Na generations)
    TC: Time in the past when growth began (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nuB,nuF,nuC,TB,TF,TC,theta1 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta1)

    phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta1)
    phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta1)

    # exponential growth from size nuF to size nuC in TC generations
    nu_func = lambda t: nuF * numpy.exp(numpy.log(nuC / nuF) * t/TC)
    phi = Integration.one_pop(phi, xx, TC, nu_func, theta0=theta1)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def prior_onegrow_mig_Eur((nu1F, nu2B, nu2F, Tp, T), n1, pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = yy = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the population growth event.
    phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)

    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = dadi.Integration.one_pop(phi, xx, T, nu=nu2_func)

    # Finally, calculate the spectrum.
    sfs = dadi.Spectrum.from_phi(phi, n1, (xx,))
    return sfs



def OutOfAfrica((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, 
                 mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs),
                 (n1,n2,n3), pts):
    """
    Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, TB, nu1=nuAf, nu2=nuB, 
                               m12=mAfB, m21=mAfB)

    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, 
                                 nu2=nuEu_func, nu3=nuAs_func, 
                                 m12=mAfEu, m13=mAfAs, m21=mAfEu,
                                 m23=mEuAs, m31=mAfAs, m32=mEuAs)

    fs = Spectrum.from_phi(phi, (n1,n2,n3), (xx,xx,xx))
    return fs

def OutOfAfricaTwoPop((nuAf, nuB, nuEu0, nuEu, 
                       mAfB, mAfEu, TAf, TB, TEuAs),
                 (n1,n2), pts):
    """
    2 populations only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, TB, nu1=nuAf, nu2=nuB, 
                               m12=mAfB, m21=mAfB)

    # phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)   # don't split into Eur and Asn

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    # nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    # phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, 
    #                             nu2=nuEu_func, nu3=nuAs_func, 
    #                             m12=mAfEu, m13=mAfAs, m21=mAfEu,
    #                             m23=mEuAs, m31=mAfAs, m32=mEuAs)
    phi = Integration.two_pops(phi, xx, TEuAs, nu1=nuAf, 
                                 nu2=nuEu_func,  
                                 m12=mAfEu, m21=mAfEu)

    # fs = Spectrum.from_phi(phi, (n1,n2,n3), (xx,xx,xx))
    fs = Spectrum.from_phi(phi, (n1,n2), (xx,xx))    
    return fs

# def OutOfAfricaAfr((nuAf, nuB, TAf, TB, TEuAs), n1, pts):
def OutOfAfricaAfr((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs), n1, pts):
    """
    1 population only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TB, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TEuAs, nu=nuAf)
    fs = Spectrum.from_phi(phi, n1, (xx,))    
    return fs

# def OutOfAfricaEur((nuAf, nuB, nuEu0, nuEu, TAf, TB, TEuAs), n1, pts):
def OutOfAfricaEur((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs), n1, pts):
    
    """
    1 population only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TB, nu=nuB)
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    phi = Integration.one_pop(phi, xx, TEuAs, nu=nuEu_func)
    fs = Spectrum.from_phi(phi, n1, (xx,))    
    return fs



# def OutOfAfricaAsn((nuAf, nuB, nuAs0, nuAs, TAf, TB, TEuAs), n1, pts):
def OutOfAfricaAsn((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs), n1, pts):

    """
    1 population only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TB, nu=nuB)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    phi = Integration.one_pop(phi, xx, TEuAs, nu=nuAs_func)
    fs = Spectrum.from_phi(phi, n1, (xx,))    
    return fs


def OOAEur((nuAf, nuB, nuEu0, nuEu, TAf, TB, TEuAs), n1, pts):
    """
    same as OutOfAfricaEur but has shorter signature with only necessary params
    1 population only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TB, nu=nuB)
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    phi = Integration.one_pop(phi, xx, TEuAs, nu=nuEu_func)
    fs = Spectrum.from_phi(phi, n1, (xx,))    
    return fs



def OOAAsn((nuAf, nuB, nuAs0, nuAs, TAf, TB, TEuAs), n1, pts):
    """
    same as OutOfAfricaAsn but has shorter signature with only necessary params
    
    1 population only from Out-of-Africa model from Gutenkunst (2009): This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The from dadi import line imports those modules from the dadi namespace into the local namespace, so we don't have to type dadi. to access them    
    """
    from dadi import Numerics, PhiManip, Integration, Spectrum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)
    phi = Integration.one_pop(phi, xx, TB, nu=nuB)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    phi = Integration.one_pop(phi, xx, TEuAs, nu=nuAs_func)
    fs = Spectrum.from_phi(phi, n1, (xx,))    
    return fs



# TODO finish and test
def gravel_eur_single_pop((nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu), ns, pts):
    """
    A one-population model used to model European out-of-Africa demography.
    Tennesen et al model
    timegrowthEu
    """

    # params used below: TAf, nuAf0, TEuAs, TB, timerowthEu, nuEu0, nuEu1, initialgrowth, nuEu2, nuEu1
    xx = Numerics.default_grid(pts)

    #first step: a single population
    phi = PhiManip.phi_1D(xx)

    #integrate for time TAf (with constant population)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf0)

    # TODO missing bottleneck integration here with nuB, TB
    
    initialgrowth=TEuAs+TB-timegrowthEu
    nuEu_func = lambda t: nuEu0*(nuEu1/nuEu0)**(min(t,initialgrowth)/initialgrowth)*(nuEu2/nuEu1)**(max(0,t-initialgrowth)/(TB+TEuAs-initialgrowth))

    # changed to one-pop integration
    #phi = Integration.two_pops(phi, xx, TB+TEuAs, nu1=nuAf_func, nu2=nuEu_func, m12=mAfB, m21=mAfB)
    # TODO what is T here?? If same as before, TB+TEuAs
    phi = Integration.one_pop(phi, xx, TB+TEuAs, nu=nuEu_func)


    # TODO did I leave out last growth event? probably fine because was super exponential... then need to adjust growth rate estiamtes
    
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def gravel_eur_single_pop_evalLik((nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu), ns, pts, LA, LX):
    """
    convenience function to evaluate likelihood gravel function
    """


def gravel_eur_single_pop_LRT((nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu), ns, pts, LA, LX, modelType):
    """
    chrX LRT on CEU gravel model extrapolated single-pop, more complex
    TB: Length of bottleneck (in units of 2*Na generations) -- guess
    """

    fitDict = {'A': {}, 'X2': {}, 'X1': {}, 'X0': {}}   # TODO put in wrapper function?
    # TODO update constants below

    # constants and file names
    LA = 3509211.      # autosomal capture region that passed mask filter: chr1-22: 3509211
    LX = 135064
    muA = 2.36e-8           # simon's value
    likType = 'multi'
    optNum = 0
    popName = 'CEU'
    siteType = 'syn'
    indir = '/Users/shaila/projects/hgdp-x/results/2014-07-01_1KG-syn-ns/VCFMultiPop/singlePopSFS/'
    outdir = '/Users/shaila/projects/hgdp-x/data/2014-07-01_1KG-syn-ns/1kG/VCFPerPop/{}/goodness-of-fit/'.format(popName)
    isCluster=False; minGrid=150; maxiter=100

    funcName = 'gravel_eur_single_pop'
    func = lrt.gravel_eur_single_pop
    func_ex = dadi.Numerics.make_extrap_log_func(func)

    ## plug in params from Simon's ESP paper
    # params = array([nuAf0=1.983, nuB=0.25, nuEu0=0.141, nuEu1=1.274, TAf=0.4054, TEuAs=0.0630, TB=0.1397, nuEu2=70.137, timegrowthEu=0.00171])
    # params = array([1.983, 0.25, 0.141, 1.274, 0.4054, 0.0630, 0.1397, 70.137, 0.00171])
    nuAf0=1.983; nuB=0.25; nuEu0=0.141; nuEu1=1.274; TAf=0.4054; TEuAs=0.0630; TB=0.1397; nuEu2=70.137; timegrowthEu=0.00171

    ##### Auto #####
    if modelType == 'A':   ## just eval lik
        chrom = 'Auto'
        base = '{}_{}_EXOMEonly_dadi_snp_reference_{}'.format(chrom, popName, siteType)

        params = array([nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu])
        # fixed_params = (nuAf0=1.983, TAf=0.4054, nuB=0.25, TB=0.1397, nuEu0=0.141, nuEu1=1.274, TEuAs=0.0630, nuEu2=70.137, timegrowthEu=0.00171)
        # gravel_eur_single_pop(params, ns)   # TODO adjust call

        infile = indir + base + '.dadi'
        outBase = outdir + base + '_model_{}_{}_opt{}'.format(modelType,funcName, optNum)
        popt, ll_opt, theta = evalLikelihood(func, params, infile, minPts, outBase)  # TODO test

        # plot data vs model
        outfile = outdir + base + '_{}_fixedParams_resid.png'.format(funcName)
        main = 'CEU: complex Tennesen model fixed for params'
        myPlot.plot_1d_comp_multinom(model, data, fs_labels=('model','data'), outfile=outfile, main=main)
        popt = params  # all params fixed
        fitDict[modelType] = {'infile': infile, 'modelfile': modelfile, 'popt': popt, 'll': ll, 'theta': theta}


    ########## chrX #########

    #### chrX: model 0 ####     ## just eval lik
    # X0: p=0.5, p constant
    # fix nu and scale tau by 4./3: just eval likeilhood
    if modelType == 'X0':
        chrom = 'X'
        # start optimizing here: expected chrX params under the null
        params = array([nuAf0, nuB, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, 4./3*TB, nuEu2, 4./3*timegrowthEu])    # fixed
        fixed_params = params
        base = '{}_{}_EXOMEonly_dadi_snp_reference_{}'.format(chrom, popName, siteType)  # TODO rename this bc reserved name
        infile = indir + base + '.dadi'
        outBase = outdir + base + '_model_{}_{}_opt{}'.format(modelType,funcName, optNum)
        popt, ll_opt, theta = evalLikelihood(func, params, infile, minPts, outBase)  # TODO test
        phat = nested.estPSimple(indir, outdir, popName, siteType, modelType, funcName, muA, LA, LX, 3, 0)   # TODO this has hard-coded file names; update
        fitDict[modelType] = {'infile': infile, 'modelfile': modelfile, 'popt': popt, 'll': ll, 'theta': theta, 'phat': phat}


    ### chrX: model 1 ###
    # X1: p free, p constant
    # TODO update to fit tau's: want them to be constrained to be some multiple. could to a grid search as in function gridSearchTwoEpochC() in this file
    if modelType == 'X1':
        chrom = 'X'
        params = array([nuAf0, nuB, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, 4./3*TB, nuEu2, 4./3*timegrowthEu])
        fixed_params = array([nuAf0, nuB, nuEu0, nuEu1, None, None, None, nuEu2, None])

    ### chrX: model 2 ###
    # X2: p=0.5 and bottleneck params nuB and TB can differ
    if modelType == 'X2':
        chrom = 'X'
        params = array([nuAf0, nuB, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, 4./3*TB, nuEu2, 4./3*timegrowthEu])
        fixed_params = array([nuAf0, None, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, None, nuEu2, 4./3*timegrowthEu])


    ### chrX: model 3 ###
    # X3: all is free
    if modelType == 'X3':
        chrom = 'X'
        params = array([nuAf0, nuB, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, 4./3*TB, nuEu2, 4./3*timegrowthEu])

    return fitDict

#    # TODO write dict to file: picke, and human readable
#    pklFile = indir + 'CEU_bottlegrowth_phat.pkl'
#    pklF = open(pklFile, 'wb')
#    pickle.dump(fitDict, pklF)
#    pklF.close()



################ Demographic functions for LRT: constrain X params ################

def three_epoch_X0(params, ns, pts):
    """
    modification of three_epoch for chrX LRT model 1: constant sex-bias
    to optimze this: nuB,nuF,TB_auto,TF_auto,theta_auto are all constant and only c is optimized over
     
    params = (nuB,nuF,TB,TF,theta,c)
    ns = (n1,)

    following are autosomal params:
    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB_auto: Length of bottleneck (in units of 2*Na generations) - auto
    TF_auto: Time since bottleneck recovery (in units of 2*Na generations) - auto
    theta_auto: mutation rate scaled by ancestral population size Na - auto
    c:  constant constraining other params

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    nu_i^X = nu_i^A
    tau_i^X = 4/3 * tau_i^A
    theta_i^X = 3/4 * theta_i^A
    """

    nuB,nuF,TB_auto,TF_auto,theta_auto,c = params

    # constraints
    TB = 4./3 * TB_auto
    TF = 4./3 * TF_auto
    theta = 3./4 * theta_auto
    
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta)

    phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
    phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def three_epoch_X1(params, ns, pts):
    """
    modification of three_epoch for chrX LRT model 1: constant sex-bias
    to optimze this: nuB,nuF,TB_auto,TF_auto,theta_auto are all constant and only c is optimized over
     
    params = (nuB,nuF,TB,TF,theta,c)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB_auto: Length of bottleneck (in units of 2*Na generations) - auto
    TF_auto: Time since bottleneck recovery (in units of 2*Na generations) - auto
    theta_auto: mutation rate scaled by ancestral population size Na - auto
    c:  constant constraining other params

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    nu_i^X = nu_i^A
    tau_i^X = 1/c * tau_i^A
    theta_i^X = c * theta_i^A
    """

    nuB,nuF,TB_auto,TF_auto,theta_auto,c = params

    # constraints
    TB = 1./c * TB_auto
    TF = 1./c * TF_auto
    theta = c * theta_auto
    
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta)

    phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
    phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def getXparams(poptA, funcName):
    """
    just for converting?
    """
    if funcName == 'three_epoch_X0':
        nuB,nuF,TB_auto,TF_auto,theta_auto,c = poptA    
        TB = 4./3 * TB_auto
        TF = 4./3 * TF_auto
        theta = 3./4 * theta_auto
        popt = array([nuB,nuF,TB,TF,theta,0.75])   # last param is fixed value of c

    elif funcName == 'three_epoch_X1':
        nuB,nuF,TB_auto,TF_auto,theta_auto,c = poptA    
        TB = 1./c * TB_auto
        TF = 1./c * TF_auto
        theta = c * theta_auto
        popt = array([nuB,nuF,TB,TF,theta,c])

    elif funcName == 'three_epoch_X2':
        nuB_auto,nuF_auto,TB_auto,TF_auto,theta_auto,c1,c2 = poptA    
        nuB = c2 / c1 * nuB_auto
        nuF = nuF_auto
        TB = 1./c1 * TB_auto
        TF = 1./c1 * TF_auto
        theta = c1 * theta_auto
        popt = array([nuB,nuF,TB,TF,theta,c1,c2])


    elif funcName == 'growth_X0':
        nu,T_auto,theta_auto = poptA
        c = 0.75
        T = 1./c * T_auto
        theta = c * theta_auto
        popt = array([nu,T,theta,c])
    elif funcName == 'growth_X1':
        nu,T_auto,theta_auto,c = poptA
        T = 1./c * T_auto
        theta = c * theta_auto
        popt = array([nu,T,theta,c])        
    elif funcName == 'growth_X2':
        nu_auto,T_auto,theta_auto,c1,c2 = poptA
        nu = c2 / c1 * nu_auto
        T = 1./c2 * T_auto   # TODO Check this
        theta = c1 * theta_auto
        popt = array([nu,T,theta,c1,c2])        
    elif funcName == 'threeEpochGrowth_X0':   # c is hardcoded
        nuB,nuF,nuC,TB,TF,TC,theta = poptA        
        c = 3./4
        TB *= 1./c
        TF *= 1./c
        TC *= 1./c        
        theta *= c
        popt = array([nuB,nuF,nuC,TB,TF,TC,theta,c])
    elif funcName == 'threeEpochGrowth_X1':
        nuB,nuF,nuC,TB,TF,TC,theta,c = poptA        
        TB *= 1./c
        TF *= 1./c
        TC *= 1./c        
        theta *= c        
        popt = array([nuB,nuF,nuC,TB,TF,TC,theta,c])        
    elif funcName == 'threeEpochGrowth_X2':
        nuB,nuF,nuC,TB,TF,TC,theta,c1,c2,c3,c4 = poptA        
        nuB *= c2/c1 
        nuF *= c3/c1
        nuC *= c4/c1
        TB *= 1./c1
        TF *= 1./c1
        TC *= 1./c1       
        theta *= c1       
        popt = array([nuB,nuF,nuC,TB,TF,TC,theta,c1,c2,c3,c4])        
    else:
        sys.exit('2: func name invalid: {}'.format(funcName))
    return popt
        

def three_epoch_X2(params, ns, pts):
    """
    modification of three_epoch for chrX LRT model 2: sex-bias of bottleneck differs from before and after
    to optimze this: nuB,nuF,TB_auto,TF_auto,theta_auto are all constant and only c is optimized over
     
    params = (nuB_auto,nuF_auto,TB_auto,TF_auto,theta_auto,c1,c2)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB_auto: Length of bottleneck (in units of 2*Na generations) - auto
    TF_auto: Time since bottleneck recovery (in units of 2*Na generations) - auto
    theta_auto: mutation rate scaled by ancestral population size Na - auto
    c:  constant constraining other params

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    nuB^X = c2 / c1 * nuB^A
    nuF^X = nuF^A
    tauB^X = 1/c1 * tauB^A
    tauF^X = 1/c1 * tauF^A
    theta_i^X = c1 * theta_i^A

    """

    nuB_auto,nuF_auto,TB_auto,TF_auto,theta_auto,c1,c2 = params

    # constraints
    nuB = c2 / c1 * nuB_auto
    nuF = nuF_auto
    TB = 1./c1 * TB_auto
    TF = 1./c1 * TF_auto
    theta = c1 * theta_auto
    
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta)

    phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
    phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch_X(params, ns, multinom, modelType, pts):
    """
    modification of three_epoch for chrX LRT model 1: constant sex-bias
    to optimze this: nuB,nuF,TB_auto,TF_auto,theta_auto are all constant and only c is optimized over
     
    params = (nuB,nuF,TB,TF,theta,c)
    ns = (n1,)
    multinom: True or False (then is Poisson)
    modelType: X0, X1, X2
    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB_auto: Length of bottleneck (in units of 2*Na generations) - auto
    TF_auto: Time since bottleneck recovery (in units of 2*Na generations) - auto
    theta_auto: mutation rate scaled by ancestral population size Na - auto
    c:  constant constraining other params

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    """

    # parameter parsing and constraints
    if modelType == 'X0':   # TODO don't need c here
        nuB,nuF,TB_auto,TF_auto,theta_auto = params
        c = 3./4
        TB = 1./c * TB_auto
        TF = 1./c * TF_auto
        theta = c * theta_auto
    elif modelType == 'X1':
        nuB,nuF,TB_auto,TF_auto,theta_auto,c = params
        TB = 1./c * TB_auto
        TF = 1./c * TF_auto
        theta = c * theta_auto
    elif modelType == 'X2':
        nuB_auto,nuF_auto,TB_auto,TF_auto,theta_auto,c1,c2 = params
        nuB = c2 / c1 * nuB_auto
        nuF = nuF_auto
        TB = 1./c1 * TB_auto
        TF = 1./c1 * TF_auto
        theta = c1 * theta_auto
    else:
        sys.exit('invalid modelType')

    # forward diffusion
    xx = Numerics.default_grid(pts)

    if multinom:
        phi = PhiManip.phi_1D(xx)
        phi = Integration.one_pop(phi, xx, TB, nuB)
        phi = Integration.one_pop(phi, xx, TF, nuF)

    else:
        phi = PhiManip.phi_1D(xx, theta0=theta)
        phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
        phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch_convert_params(params, modelType, retConstraints=False):
    """
    params: autosomal paramters with optional constraint parameters c at end
    retConstraints: True or False. Generally True for printing of estimated parameters and False for converting from autosomal to X params to pass to a demographic function
    """
    # parameter parsing and constraints
    
    if modelType == 'X0':   # TODO don't need c here
        nuB,nuF,TB_auto,TF_auto,theta_auto = params
        c = 3./4
        TB = 1./c * TB_auto
        TF = 1./c * TF_auto
        theta = c * theta_auto
        cList = [c]
    elif modelType == 'X1':
        nuB,nuF,TB_auto,TF_auto,theta_auto,c = params
        TB = 1./c * TB_auto
        TF = 1./c * TF_auto
        theta = c * theta_auto
        cList = [c]        
    elif modelType == 'X2':
        nuB_auto,nuF_auto,TB_auto,TF_auto,theta_auto,c1,c2 = params
        nuB = c2 / c1 * nuB_auto
        nuF = nuF_auto
        TB = 1./c1 * TB_auto
        TF = 1./c1 * TF_auto
        theta = c1 * theta_auto
        cList = [c1, c2]
    else:
        sys.exit('invalid modelType')

    params = array([nuB,nuF,TB,TF,theta])
    if retConstraints:
        params = numpy.append(params, cList)

    return params
        

def three_epoch_X_v2(params, ns, multinom, pts):
    """
    modification of three_epoch for chrX LRT model 1: constant sex-bias
    to optimze this: nuB,nuF,TB_auto,TF_auto,theta_auto are all constant and only c is optimized over
     
    The X chromosomal paramters are input
    params = (nuB,nuF,TB,TF,theta)
    ns = (n1,)
    multinom: True or False (then is Poisson)
    modelType: X0, X1, X2
    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB_auto: Length of bottleneck (in units of 2*Na generations) - auto
    TF_auto: Time since bottleneck recovery (in units of 2*Na generations) - auto
    theta_auto: mutation rate scaled by ancestral population size Na - auto
    c:  constant constraining other params

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nuB,nuF,TB,TF,theta = params  # pass in theta though multinom mode will not use

    # forward diffusion
    xx = Numerics.default_grid(pts)

    if multinom:
        phi = PhiManip.phi_1D(xx)
        phi = Integration.one_pop(phi, xx, TB, nuB)
        phi = Integration.one_pop(phi, xx, TF, nuF)

    else:
        phi = PhiManip.phi_1D(xx, theta0=theta)
        phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
        phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch_X1_P(params, ns, pts):
    fs = three_epoch_X(params, ns, multinom=False, modelType='X1', pts=pts)
    return fs

def three_epoch_X1_P_v2(params, ns, pts):
    params = three_epoch_convert_params(params, modelType='X1')
    fs = three_epoch_X_v2(params, ns, multinom=False, pts=pts)
    return fs


    
def growth_X(params, ns, multinom, modelType, pts):
    """
    TODO not tested
    Exponential growth beginning some time ago.

    params = (nu,T)
    ns = (n1,)
    multinom: True or False (then is Poisson)
    modelType: X0, X1
    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which growth began (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """

    # parameter parsing and constraints    
    if modelType == 'X0':
        nu,T_auto,theta_auto = params
        c = 0.75
        T = 1./c * T_auto
        theta = c * theta_auto
    elif modelType == 'X1':
        nu,T_auto,theta_auto,c = params
        T = 1./c * T_auto
        theta = c * theta_auto
    elif modelType == 'X2':
        nu_auto,T_auto,theta_auto,c1,c2 = params
        nu = c2 / c1 * nu_auto
        T = 1./c2 * T_auto  # TODO CHeck c2?
        theta = c1 * theta_auto
    else:
        sys.exit('invalid modelType')

    # forward diffusion
    xx = Numerics.default_grid(pts)
    if multinom:
        phi = PhiManip.phi_1D(xx)
        nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
        phi = Integration.one_pop(phi, xx, T, nu_func)
    else:
        phi = PhiManip.phi_1D(xx, theta0=theta)
        nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
        phi = Integration.one_pop(phi, xx, T, nu_func, theta0=theta)
        
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def growth_X0_P(params, ns, pts):
    fs = growth_X(params, ns, multinom=False, modelType='X0', pts=pts)
    return fs
def growth_X1_P(params, ns, pts):
    fs = growth_X(params, ns, multinom=False, modelType='X1', pts=pts)
    return fs
def growth_X2_P(params, ns, pts):
    fs = growth_X(params, ns, multinom=False, modelType='X2', pts=pts)
    return fs

def growth_X0_M(params, ns, pts):
    fs = growth_X(params, ns, multinom=True, modelType='X0', pts=pts)
    return fs
def growth_X1_M(params, ns, pts):
    fs = growth_X(params, ns, multinom=True, modelType='X1', pts=pts)
    return fs
def growth_X2_M(params, ns, pts):
    fs = growth_X(params, ns, multinom=True, modelType='X2', pts=pts)
    return fs


def threeEpochGrowth_X(params, ns, multinom, modelType, pts):
    """
    Three size epochs followed by exponential growth.
    params = (nuB, nuF, nuC, TB, TF, TC)
    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of post-bottleneck size to ancient pop size
    nuC: Ratio of contemporary size to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Length of time of post-bottleneck size (in units of 2*Na generations)
    TC: Time in the past when growth began (in units of 2*Na generations)
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    # parameter parsing and constraints
    if modelType == 'X0':   # TODO don't need c here
        nuB,nuF,nuC,TB,TF,TC,theta = params        
        c = 3./4
        TB *= 1./c
        TF *= 1./c
        TC *= 1./c        
        theta *= c
    elif modelType == 'X1':
        nuB,nuF,nuC,TB,TF,TC,theta,c = params        
        TB *= 1./c
        TF *= 1./c
        TC *= 1./c        
        theta *= c        
    elif modelType == 'X2':
        nuB,nuF,nuC,TB,TF,TC,theta,c1,c2,c3,c4 = params        
        nuB *= c2/c1 
        nuF *= c3/c1
        nuC *= c4/c1
        TB *= 1./c1
        TF *= 1./c1
        TC *= 1./c1       
        theta *= c1       
    else:
        sys.exit('invalid modelType')

    # forward diffusion
    xx = Numerics.default_grid(pts)

    if multinom:
        phi = PhiManip.phi_1D(xx)
        phi = Integration.one_pop(phi, xx, TB, nuB)
        phi = Integration.one_pop(phi, xx, TF, nuF)
        nu_func = lambda t: nuF * numpy.exp(numpy.log(nuC / nuF) * t/TC)
        phi = Integration.one_pop(phi, xx, TC, nu_func)

    else:
        phi = PhiManip.phi_1D(xx, theta0=theta)
        phi = Integration.one_pop(phi, xx, TB, nuB, theta0=theta)
        phi = Integration.one_pop(phi, xx, TF, nuF, theta0=theta)
        nu_func = lambda t: nuF * numpy.exp(numpy.log(nuC / nuF) * t/TC)
        phi = Integration.one_pop(phi, xx, TC, nu_func, theta0=theta)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def threeEpochGrowth_X0_P(params, ns, pts):
    fs = threeEpochGrowth_X(params, ns, multinom=False, modelType='X0', pts=pts)
    return fs
def threeEpochGrowth_X1_P(params, ns, pts):
    fs = threeEpochGrowth_X(params, ns, multinom=False, modelType='X1', pts=pts)
    return fs
def threeEpochGrowth_X2_P(params, ns, pts):
    fs = threeEpochGrowth_X(params, ns, multinom=False, modelType='X2', pts=pts)
    return fs



######### 
# Contains a single change in population size. Can make SFS immediately after. Does not store ms output file.
# defaults:   nu = 0.1                   # 10x contraction
# tauDefault = 0.005         # for Nref of 500: 100 gens ago
# parameters: nu1 = ratio of current epoch 3 to epoch 2 size; nu2 = ratio of epoch 2 to epoch 1
#             tau1 = end of bottleneck; tau2 = beginning of bottleneck
def threeEpoch(outdir, tau1, tau2, nu1, nu2, numIters = 1):
    thetaDefault = 50
    rhoDefault = 50
    chromList = ['A', 'X']
    props = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for currProp in props:
        for chrom in chromList:
            if chrom =='A':
                numChroms = 40
                f = 4 * currProp * (1-currProp)
                rho = f * rhoDefault
            else:
                numChroms = 30
                f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
                r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
                rho = f * r * rhoDefault      # rho = 4 Ne^X * 2/3 * per-site recomb rate. TODO Check that the python matches this formula
            tau = tauDefault / f
            theta = f * thetaDefault

            for i in range(numIters):
                base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, i)

                cmd = 'nice ms {0} 1000 -t {1} -r {2} 5000 -eN {3} {4}'.format(numChroms, theta, rho, tau, nu)
                if i == 0:
                    print 'p = {0}, {1}'.format(currProp, chrom)
                    print cmd

                # make spectrum from ms simulation without writing to file
                ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)

                # make dadi SFS
                outfile = base + '.dadi'
                dadi.Spectrum.to_file(ms_fs, outfile)

                # make prfreq format SFS
                outfile = base + '.prfreq'
                prf_fs = ms_fs[1:]          # discard first bin for prfreq format
                prf_fs = [str(x) for x in prf_fs]
                outfile = open(outfile, 'w')
                for i in prf_fs:
                    # outfile.write(str(int(i)) + '\n')
                    outfile.write(i + '\n')
                outfile.close()


# similar to singleSizeChangeOneIter
# tau1, nu1: most recent event. End of bottleneck, so expansion to original size.
# tau2, nu2: older event. Start of bottleneck, so reduction.
def simBottleneck(outdir, nu1, tauDefault1, nu2, tauDefault2, currProp, idx, indepSites=True, numChromsDefault=40, thetaDefault = 50, rhoDefault = 50, writeMsFile=False):
    chromList = ['A', 'X']
    for chrom in chromList:
        if chrom =='A':
            numChroms = numChromsDefault
            f = 4 * currProp * (1-currProp)
            rho = f * rhoDefault
        else:
            numChroms = int(numChromsDefault  * 0.75)    # sex un-biased sample
            f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
            r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
            rho = f * r * rhoDefault
        tau1 = tauDefault1 / f
        tau2 = tauDefault2 / f
        theta = f * thetaDefault
        base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, idx)

        if indepSites == True:
            cmd = 'ms {0} 1000 -t {1} -eN {2} {3} -eN {4} {5}'.format(numChroms, theta, tau1, nu1, tau2, nu2)
        else:
            cmd = 'ms {0} 1000 -t {1} -r {2} 5000 -eN {3} {4} -eN {5} {6}'.format(numChroms, theta, rho, tau1, nu1, tau2, nu2)

        if writeMsFile:
            msfile = base + '.ms'
            cmd += '> {}'.format(msfile)
            os.system(cmd)
            ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=False)
        else:             # make spectrum from ms simulation without writing to file
            ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)

        # make dadi SFS
        outfile = base + '.dadi'
        dadi.Spectrum.to_file(ms_fs, outfile)



# Contains a single change in population size and makes one file per iteration
# defaults:   nu = 0.1                   # 10x contraction
# tauDefault = 0.005         # for Nref of 500: 100 gens ago
## original verison assumes LX = LA
# LXtoLA = LX/LA
# thetaDefault is the autosomal theta for p = 0.5; tauDefault is also for auto p = 0.5
# TODO rewrite theta conversions more sensibly
# oneFilePerIter: if true, writes one dadi sfs file per ms iteration. numIters gets set to 1
# doAverage false -> sums over ms iterations to make fs
def singleSizeChange(outdir, nu = 0.1, tauDefault = 0.005, oneFilePerIter=False, numIters = 1, doAverage=True, indepSites=True, numChromsDefault=40, thetaDefault = 50, LXtoLA = 1, rhoDefault = 50, writeMsFile=False):
    thetaA = thetaDefault * 1.
    thetaX = LXtoLA * thetaDefault * 1.
    chromList = ['A', 'X']
    props = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for currProp in props:
        for chrom in chromList:
            if chrom =='A':
                numChroms = numChromsDefault
                f = 4 * currProp * (1-currProp)
                rho = f * rhoDefault
                theta = f * thetaA
            else:
                numChroms = int(numChromsDefault  * 0.75)    # sex un-biased sample
                f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
                r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
                rho = f * r * rhoDefault
                theta = f * thetaX

            tau = tauDefault / f
            if oneFilePerIter:
                for i in range(numIters):
                    base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, i)
                    if indepSites == True:
                        cmd = 'ms {} 1 -t {} -eN {} {}'.format(numChroms, theta, tau, nu)
                    else:
                        cmd = 'ms {} 1 -t {} -r {} 5000 -eN {} {}'.format(numChroms, theta, rho, tau, nu)
                    if i == 0:       ## print each unique command once
                        print 'p = {0}, {1}\n{2}'.format(currProp, chrom, cmd)
                    if writeMsFile:
                        msfile = base + '.ms'
                        cmd += '> {}'.format(msfile)
                        os.system(cmd)
                        ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=False)
                    else:             # make spectrum from ms simulation without writing to file
                        ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)
                    outfile = base + '.dadi'       # make dadi SFS
                    dadi.Spectrum.to_file(ms_fs, outfile)

            else:
                base = outdir + 'pf_{0}_{1}_ms'.format(currProp, chrom)
                if indepSites == True:
                    cmd = 'ms {} {} -t {} -eN {} {}'.format(numChroms, numIters, theta, tau, nu)
                else:
                    cmd = 'ms {} {} -t {} -r {} 5000 -eN {} {}'.format(numChroms, numIters, theta, rho, tau, nu)
                print 'p = {0}, {1}\n{2}'.format(currProp, chrom, cmd)
                if writeMsFile:
                    msfile = base + '.ms'
                    cmd += '> {}'.format(msfile)
                    os.system(cmd)
                    ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=doAverage)
                else:             # make spectrum from ms simulation without writing to file
                    ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=doAverage)
                outfile = base + '.dadi'       # make dadi SFS
                dadi.Spectrum.to_file(ms_fs, outfile)


                # make prfreq format SFS
#                outfile = base + '.prfreq'
#                prf_fs = ms_fs[1:]          # discard first bin for prfreq format
#                prf_fs = [str(x) for x in prf_fs]
#                outfile = open(outfile, 'w')
#                for i in prf_fs:
#                    outfile.write(i + '\n')
#                outfile.close()


# Contains a single change in population size. Can make SFS immediately after. Does not store ms output file.
# defaults:   nu = 0.1                   # 10x contraction
# tauDefault = 0.005         # for Nref of 500: 100 gens ago
# original verison assumes LX = LA; here LXtoLA = LX/LA
# thetaDefault is the autosomal theta for p = 0.5; tauDefault is also for auto p = 0.5
# TODO rewrite theta conversions more sensibly
def singleSizeChangeOneIter(outdir, nu, tauDefault, currProp, idx, indepSites=True, numChromsDefault=40, thetaDefault = 50, rhoDefault = 50, writeMsFile=False):
    chromList = ['A', 'X']
    for chrom in chromList:
        if chrom =='A':
            # numChroms = 40
            numChroms = numChromsDefault
            f = 4 * currProp * (1-currProp)
            rho = f * rhoDefault
        else:
            # numChroms = 30
            numChroms = int(numChromsDefault  * 0.75)    # sex un-biased sample
            f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
            r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
            rho = f * r * rhoDefault
        tau = tauDefault / f
        theta = f * thetaDefault

        base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, idx)

        if indepSites == True:
            cmd = 'ms {0} 1000 -t {1} -eN {2} {3}'.format(numChroms, theta, tau, nu)
        else:
            cmd = 'ms {0} 1000 -t {1} -r {2} 5000 -eN {3} {4}'.format(numChroms, theta, rho, tau, nu)

        if writeMsFile:
            msfile = base + '.ms'
            cmd += '> {}'.format(msfile)
            os.system(cmd)
            ms_fs = dadi.Spectrum.from_ms_file(msfile, mask_corners=True, average=False)
        else:             # make spectrum from ms simulation without writing to file
            ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)

        # make dadi SFS
        outfile = base + '.dadi'                 # TODO rename to _dadi_data_sfs.txt
        dadi.Spectrum.to_file(ms_fs, outfile)

# standard neutral model. Can make SFS immediately after. Does not store ms output file.
def snmOneIter(outdir, currProp, idx):
    thetaDefault = 50
    rhoDefault = 50
    chromList = ['A', 'X']
    for chrom in chromList:
        if chrom =='A':
            numChroms = 40
            f = 4 * currProp * (1-currProp)
            rho = f * rhoDefault
        else:
            numChroms = 30
            f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
            # r = (2 * currProp) / (1 + currProp)
            r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
            rho = f * r * rhoDefault
        theta = f * thetaDefault
        base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, idx)

        cmd = 'nice ms {0} 1000 -t {1} -r {2} 5000'.format(numChroms, theta, rho)

        # make spectrum from ms simulation without writing to file
        ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)

        # make dadi SFS
        outfile = base + '.dadi'                 # TODO rename to _dadi_data_sfs.txt
        dadi.Spectrum.to_file(ms_fs, outfile)


# standard neutral model. Can make SFS immediately after. Does not store ms output file.
def snmOneIter(outdir, currProp, idx):
    thetaDefault = 50
    rhoDefault = 50
    chromList = ['A', 'X']
    for chrom in chromList:
        if chrom =='A':
            numChroms = 40
            f = 4 * currProp * (1-currProp)
            rho = f * rhoDefault
        else:
            numChroms = 30
            f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
            # r = (2 * currProp) / (1 + currProp)
            r = 2/3.               # 2/3 of the X's underwent recombination: those from maternal transmission
            rho = f * r * rhoDefault
        theta = f * thetaDefault
        base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, idx)

        cmd = 'nice ms {0} 1000 -t {1} -r {2} 5000'.format(numChroms, theta, rho)

        # make spectrum from ms simulation without writing to file
        ms_fs = dadi.Spectrum.from_ms_file(os.popen(cmd), mask_corners=True, average=False)

        # make dadi SFS
        outfile = base + '.dadi'                 # TODO rename to _dadi_data_sfs.txt
        dadi.Spectrum.to_file(ms_fs, outfile)

# standard neutral model. Can make SFS immediately after. Does not store ms output file.
# to comare to ms: 5000 bp locus, 1000 indep iterations
def snmPoisOneIter(outdir, currProp, idx):
    # thetaDefault = 50
    # locusLen = 5000
    # numReps = 1000
    thetaDefault = 5000
    chromList = ['A', 'X']
    for chrom in chromList:
        if chrom =='A':
            numChroms = 40
            f = 4 * currProp * (1-currProp)
        else:
            numChroms = 30
            f =  (9 * currProp * (1-currProp)) / ((2-currProp) * 2)
        theta = f * thetaDefault
        base = outdir + 'pf_{0}_{1}_ms_{2}'.format(currProp, chrom, idx)

        idx = range(1, numChroms)
        poisParams = [theta/i for i in idx]
        sfs = [numpy.random.poisson(p, 1)[0] for p in poisParams]
        sfs.insert(0, 0)     # add zero to beginning of SFS: 0 derived alleles
        sfs.append(0)        # add zero to beginning of SFS: 2n derived alleles
        sfs = [str(x) for x in sfs]

        # for i in range(0, numChroms):
        #    poisParam = theta / i
        #    S_i = numpy.random.poisson(poisParam, 1)

        # make dadi SFS
        outfile = base + '.dadi'                 # TODO rename to _dadi_data_sfs.txt
        outF = open(outfile, 'w')
        header = "{0} unfolded\n".format(numChroms + 1)
        outF.write(header)
        sfsString = ' '.join(sfs) + '\n'
        outF.write(sfsString)
        maskLine = [1] + [0]*(numChroms-1) + [1]
        maskLine = [str(x) for x in maskLine]
        maskLine = ' '.join(maskLine)
        outF.write(maskLine + '\n')
        outF.close()

######## end paste from lrt-nested-pipeline.py


## model 2: all free params. Default.
# redundant? default min grid specified in caller
# TODO make min grid increment by proportions, not absolute amounts
def fitTwoEpoch(infile, outfile, modelfile, isCluster=True, nuFixed=None, tauFixed=None, minGrid=100, params=None):
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    # pts_l = [40,50,60]
    # pts_l = [100, 110, 120]    ## larger grid set on 2014-03-06
    pts_l = [minGrid, minGrid+10, minGrid+20]    ## user specified grid on 2014-03-12
    func = dadi.Demographics1D.two_epoch
    # params = array([1, 0.5])           # nu, T -- not true values. Should not matter for fixed values
    if params is None:
        params = array([5, 0.1])           # 2014-03-10 Starting params for growth
    elif len(params) != 2:
        sys.exit('Error: initial input parameters must be an array of length two for this model')
    fixed_params=[nuFixed, tauFixed]
    upper_bound = [1000, 10]           # TODO could reduce this?
    lower_bound = [1e-8, 1e-8]
    if nuFixed != None:
        if (nuFixed < lower_bound[0]) or (nuFixed > upper_bound[0]):
            print 'ERROR nuFixed {0} out of bounds: {1} {2}. exiting'.format(nuFixed, lower_bound[0], upper_bound[0])
            exit()
    if tauFixed != None:
        if (tauFixed < lower_bound[1]) or (tauFixed > upper_bound[1]):
            print 'ERROR tauFixed {0} out of bounds: {1} {2}. exiting'.format(tauFixed, lower_bound[1], upper_bound[1])
            exit()

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    ll_model = dadi.Inference.ll_multinom(model, data)
    p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bound)   # TODO why no upper bound?
    if isCluster == True:         # buffer IO
        flush_delay = 1        # could play with this
    else:
        flush_delay = 0.5      # default
    popt = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
                                      lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=len(params),
                                       maxiter=1000, output_file=outfile,
                                       flush_delay=flush_delay, fixed_params=fixed_params)

    outfile = open(outfile, 'a')
    outfile.write('Optimized parameters ' + repr(popt) + '\n')
    model = func_ex(popt, ns, pts_l)
    ll_opt = dadi.Inference.ll_multinom(model, data)
    theta = dadi.Inference.optimal_sfs_scaling(model, data)

    outfile.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(theta) + '\n')
    nuhat = popt[0]
    tauhat = popt[1]
    # outfile.write('{.7f} {.7f} {.7f} {.7f}'.format(nuhat, tauhat, theta, ll_opt))   # hard-coded for two params. TODO get the float formatting to work
    outfile.write('{} {} {} {}\n'.format(nuhat, tauhat, theta, ll_opt))   # hard-coded for two params
    outfile.close()

    modelfile = open(modelfile, 'w')         # write SFS expected under demographic model and params to file
    model.to_file(modelfile)
    modelfile.close()
    return (popt, ll_opt, theta)


# 2014-09-11 Adapted multi case to make pois case. Eventually make into one function
# todo change params to paramsInit
def fitTwoEpochPois(infile, outfile, modelfile, isCluster=True, nuFixed=None, tauFixed=None, thetaFixed=None, minGrid=100, params=None, maxiter=1000):
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [minGrid, minGrid+10, minGrid+20]
    func = two_epoch_fixed_theta
    if params is None:
        params = array([7000, 5, 0.1])           # correspond to a growth event
    elif len(params) != 3:
        sys.exit('Error: initial input parameters must be an array of length three for this model')
    fixed_params=[thetaFixed, nuFixed, tauFixed]
    upper_bound = [20000, 100, 10]           # TODO how to adjust upper bound for theta?
    lower_bound = [100, 1e-8, 1e-8]
    if thetaFixed != None:
        if (thetaFixed < lower_bound[0]) or (thetaFixed > upper_bound[0]):
            print 'ERROR thetaFixed {0} out of bounds: {1} {2}. exiting'.format(thetaFixed, lower_bound[0], upper_bound[0])
            exit()
    if nuFixed != None:
        if (nuFixed < lower_bound[1]) or (nuFixed > upper_bound[1]):
            print 'ERROR nuFixed {0} out of bounds: {1} {2}. exiting'.format(nuFixed, lower_bound[1], upper_bound[1])
            exit()
    if tauFixed != None:
        if (tauFixed < lower_bound[2]) or (tauFixed > upper_bound[2]):
            print 'ERROR tauFixed {0} out of bounds: {1} {2}. exiting'.format(tauFixed, lower_bound[2], upper_bound[2])
            exit()
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)
    if isCluster == True:         # buffer IO
        flush_delay = 1        # could play with this
    else:
        flush_delay = 0.5      # default
    popt = dadi.Inference.optimize_log(p0, data, func_ex, pts_l,
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=len(p0),
                                       multinom=False,          # specifies that theta is explicit
                                       fixed_params=fixed_params,
                                       maxiter=maxiter, output_file=outfile,
                                       flush_delay=flush_delay)
    model = func_ex(popt, ns, pts_l)
    ll_opt = dadi.Inference.ll(model, data)
    thetaHat = popt[0]
    nuHat = popt[1]
    tauHat = popt[2]
    outfile = open(outfile, 'a')
    outfile.write('Optimized parameters ' + repr(popt) + '\n')
    outfile.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(thetaHat) + '\n')
    outfile.write('{} {} {} {}\n'.format(nuHat, tauHat, thetaHat, ll_opt))   # hard-coded for two params
    outfile.close()

    modelfile = open(modelfile, 'w')         # write SFS expected under demographic model and params to file
    model = model / (thetaHat * 1.)     # so will have same scale as multinomial SFS
    model.to_file(modelfile)
    modelfile.close()
    return (popt, ll_opt, thetaHat)


## for each pair of auto, X:
# LL = LLA + LLX
# calc ll.2.1 == LL2 - LL1
# calc ll.1.0 == LL1 - LL0
def nestedModelLRT(isCluster, indir, baseA, baseX, allOutfile, minGrid=100):
    """
    This is just for two epoch!
    """
    allF = open(allOutfile, 'w')
    header = 'chrom model LL nu tau theta\n'
    allF.write(header)

    # auto: just run once
    chrom = 'A'
    infile = baseA + '.dadi'
    outfile = baseA + '-dadi-model2.txt'    # single line: demog param fitting
    modelfile = baseA + '_dadi_model2_sfs.txt'
    logfile = baseA + '-dadi-model2.log'    # for demog optimization
    (poptA, ll_optA, thetaA) = fitTwoEpoch(infile, logfile, modelfile, isCluster, nuFixed=None, tauFixed=None, minGrid=minGrid)         # retry a few times if hit param boundary

# TODO untested code: re-running if hit param boundary
#     hitParamBoundary = True
#     ctr = 0
#     maxIter = 5     # max num to try optimization
#     tol = 1e-5
#     if hitParamBoundary == True and ctr < maxIter:
#         (poptA, ll_optA, thetaA) = fitTwoEpoch(infile, logfile, modelfile, isCluster)         # retry a few times if hit param boundary
#         upper_bound = [1000, 10]           # TODO could reduce this?
#         lower_bound = [1e-8, 1e-8]
#         nuHat = poptA[0]
#         tauHat = poptA[1]
#         if (nuHat < lower_bound[0] + tol) or (nuHat > upper_bound[0] - tol) or (tauHat < lower_bound[1] + tol) or (tauHat > upper_bound[1] - tol):
#             if (nuHat < lower_bound[0]) or (nuHat > upper_bound[0]):
#                 print 'ERROR nuHat {0} out of bounds: {1} {2}. exiting'.format(nuHat, lower_bound[0], upper_bound[0])
#             if (tauHat < lower_bound[1]) or (tauHat > upper_bound[1]):
#                 print 'ERROR tauHat {0} out of bounds: {1} {2}. exiting'.format(tauHat, lower_bound[1], upper_bound[1])
#             ctr += 1
#             continue
#         else:
#             hitParamBoundary = False

    outstr = 'A 2 {0} {1} {2} {3} '.format(ll_optA, poptA[0], poptA[1], thetaA) + '\n'
    allF.write(outstr)

    modelNums = [2, 1, 0]
    for modelNum in modelNums:    # they are numbered 0, 1, 2
        chrom = 'X'
        infile = baseX + '.dadi'
        outfile = baseX + '-dadi-model' + str(modelNum) + '.txt'    # single line: demog param fitting
        modelfile = baseX + '_dadi_model' + str(modelNum) + '_sfs.txt'
        logfile = baseX + '-dadi-model' + str(modelNum) + '.log'    # for demog optimization

        if modelNum == 2:
            # model 2: run for auto, run for X with no params fixed
            (poptX, ll_optX, thetaX) = fitTwoEpoch(infile, logfile, modelfile, isCluster, nuFixed=None, tauFixed=None, minGrid=minGrid)
            LL2 = ll_optA + ll_optX

        elif modelNum == 1:
            # model 1: run for auto, run for X passing in 3/4 of nu auto
            nuFixed = poptA[0]
            (poptX, ll_optX, thetaX) = fitTwoEpoch(infile, logfile, modelfile, isCluster, nuFixed, tauFixed=None, minGrid=minGrid)
            LL1 = ll_optA + ll_optX

        elif modelNum == 0:
            # model 0: run for auto, run for X passing in 3/4 of nu auto AND 3/4 of tau auto
            nuFixed = poptA[0]
            tauFixed = poptA[1] * 4/3
            (poptX, ll_optX, thetaX) = fitTwoEpoch(infile, logfile, modelfile, isCluster, nuFixed, tauFixed, minGrid)
            LL0 = ll_optA + ll_optX

        else:
            print 'ERROR: invalid model number' + modelNum; exit()

        outstr = 'X {0} {1} {2} {3} {4} '.format(modelNum, ll_optX, poptX[0], poptX[1], thetaX) + '\n'
        allF.write(outstr)

    LL_2_1 = LL2 - LL1
    LL_1_0 = LL1 - LL0
    outstr = 'LL2 = {0}, LL1 = {1}, LL0 = {2}, ll.2.1 = {3}, ll.1.0 = {4}'.format(LL2, LL1, LL0, LL_2_1, LL_1_0)
    allF.write(outstr + '\n')
    allF.close()


def estP(infileX, modelfileX, logfileX, infileA, modelfileA, muA, LA, LX, alpha):
    """
    needs: X and A input fs; X and A model fit; additional params
    """
    logF = open(logfileX, 'a')   # append to existing chrX logfile

    # chrom='X'
    dataX = dadi.Spectrum.from_file(infileX)
    modelX = dadi.Spectrum.from_file(modelfileX)
    SX = dataX.S()
    FX = modelX.S()

    # chrom='Auto'
    dataA = dadi.Spectrum.from_file(infileA)
    modelA = dadi.Spectrum.from_file(modelfileA)
    SA = dataA.S()
    FA = modelA.S()

    # calculate muX based on muA and alpha
    muX = (2.*(2.+alpha)) / (3.*(1.+alpha)) * muA
    
    Q = 1. / (SA * LX * muX * FX) * (SX * LA * muA * FA)
    outstr = '(correct?) using alpha={}: Q = {}\n'.format(alpha, Q)
    logF.write(outstr)

    pHat = 2. - (9./8) * (SA * LX * muX * FX) / (SX * LA * muA * FA)
    outstr = 'using alpha={}: pHat = {}\n'.format(alpha, pHat)
    # print outstr
    logF.write(outstr)
    logF.close()
    return(pHat)

def estPnowrite(infileX, modelfileX, infileA, modelfileA, muA, LA, LX, alpha):
    """
    needs: X and A input fs; X and A model fit; additional params
    does not write to log file
    """
    # chrom='X'
    dataX = dadi.Spectrum.from_file(infileX)
    modelX = dadi.Spectrum.from_file(modelfileX)
    SX = dataX.S()
    FX = modelX.S()

    # chrom='Auto'
    dataA = dadi.Spectrum.from_file(infileA)
    modelA = dadi.Spectrum.from_file(modelfileA)
    SA = dataA.S()
    FA = modelA.S()

    # calculate muX based on muA and alpha    
    muX = (2.*(2.+alpha)) / (3.*(1.+alpha)) * muA

    Q = 1. / (SA * LX * muX * FX) * (SX * LA * muA * FA)
    outstr = 'using alpha={}: Q = {}\n'.format(alpha, Q)
    # print outstr

    pHat = 2. - (9./8) * (SA * LX * muX * FX) / (SX * LA * muA * FA)
    outstr = 'using alpha={}: pHat = {}\n'.format(alpha, pHat)
    # print outstr
    return(pHat)
    

def chooseBestOpt(filePrefix, funcName, numOpts):
    """
    Chooses best likelihood of set of optimization runs
    Improve: have some manual inspection based on demog params
    Assuming indices are 0..numOpts; CG file formats
    """
    ll_max = -1000000     # hardcoded small log likelihood
    outfileBest = None
    modelfileBest = None
    for optNum in range(numOpts):
        outfile = '{}{}.txt'.format(filePrefix, optNum)
        modelfile = '{}{}.dadi'.format(filePrefix, optNum)
        popt, ll_opt, theta, autoParamDict = read1DParams(funcName, outfile)
        if ll_opt is None:     # TODO better way to handle?
            continue
        ll_opt = float(ll_opt)
        if ll_opt > ll_max:
            ll_max = ll_opt
            outfileBest = outfile
            modelfileBest = modelfile
    if outfileBest is None:
        print 'error: no best outfile assigned'
    return (outfileBest, modelfileBest)



def readConstantsFromFile(jsonFile, debug=False):
    """
    read from json file and convert unicode to bytecode
    """
    inF = open(jsonFile, 'r')
    dataDict = json.load(inF)
    dataDictConverted = pyUtils.byteify(dataDict)
    if debug:
        pprint.pprint(dataDictConverted)
    return dataDictConverted


def modelX0Params(params, funcName, LA, LX, alpha):
    """
    calculates chrX params expected based on auto params, p=0.5, LX=LA, muX=muA
    T are times in dadi units, which is 2 * Nanc. the chrX expected value of the auto time because it is assumed NancX = 3/4 * NancA
    """

    alphaFactor = (2.*(2.+alpha)) / (3.*(1.+alpha))
    cFactor = 0.75 * alphaFactor * LX / LA
    
    if funcName == 'two_epoch':
        nu,T,theta = params
        theta *= cFactor
        T *= 4./3
        paramsExpected = [nu, T, theta]
        
    elif funcName == 'growth':
        nu,T ,theta = params
        theta *= cFactor
        T *= 4./3
        paramsExpected = [nu, T, theta]
    elif funcName == 'bottlegrowth':
        nuB,nuF,T ,theta = params
        theta *= cFactor         
        T *= 4./3
        paramsExpected = [nuB, nuF, T, theta]
    elif funcName == 'twoEpochGrowth':
        nuB,nuF,TB,TF ,theta = params
        theta *= cFactor         
        TB *= 4./3
        TF *= 4./3
        paramsExpected = [nuB, nuF, TB, TF, theta]
        
    elif funcName == 'three_epoch':
        nuB,nuF,TB,TF ,theta = params
        theta *= cFactor         
        TB *= 4./3
        TF *= 4./3
        paramsExpected = [nuB, nuF, TB, TF, theta]
    elif funcName == 'threeEpochGrowth':
        nuB,nuF,nuC,TB,TF,TC ,theta = params
        theta *= cFactor         
        TB *= 4./3
        TF *= 4./3
        TC *= 4./3
        paramsExpected = [nuB, nuF, nuC, TB, TF, TC, theta]
        
    elif funcName == 'gravel_eur_single_pop':
        nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu ,theta = params
        theta *= cFactor
        TAf *= 4./3
        TEuAs *= 4./3
        TB *= 4./3
        timegrowthEu *= 4./3
        paramsExpected = [nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu, theta]
    else:
        sys.exit('3: func name invalid: {}'.format(funcName))
    return paramsExpected

    

# DONE parallelize this by taking in optNum; make general enough for any data set
# TODO assuptions: LX=LA, muX=muA, multinomial model so theta is not an explict param
def lrt1DModel(jsonFile, autoOptNum=None, isCluster=True):
    """
    runs nested LRTs on any 1D model
    one case per demographic function
    """

    # read params from jsonFile: has auto and X file names, constants
    dataDict = readConstantsFromFile(jsonFile)
    # put in local file space - bad idea to do all
    infileX = dataDict['infileX']
    infileA = dataDict['infileA']
    funcName = dataDict['funcName']
    outBaseX = dataDict['outBaseX']

    ## special case: snm. just calculate p(Qpi) and p(Qtheta), write to file
    if funcName == 'snm':
        LX = dataDict['LX']
        LA = dataDict['LA']
        muA = dataDict['muA']
        alpha = dataDict['alpha']
        muX = (2.*(2.+alpha)) / (3.*(1.+alpha)) * muA

        infileA = dataDict['infileA']
        modelType = 'X0'
        outBase = outBaseX.replace('MODELTYPE', modelType)
        outfile = outBase + '.txt'
        logfile = outBase + '.log'

        fsX = dadi.Spectrum.from_file(infileX)
        fsA = dadi.Spectrum.from_file(infileA)
        thetaX = dadi.Spectrum.Watterson_theta(fsX)
        thetaA = dadi.Spectrum.Watterson_theta(fsA)
        piX = dadi.Spectrum.pi(fsX)   # expected value of pi from fs
        piA = dadi.Spectrum.pi(fsA)

        Qtheta = (thetaX / thetaA) / (LX * muX) * (LA * muA)
        Qpi = (piX / piA) / (LX * muX) * (LA * muA)         # TODO do I want to use mutation rate conversion for pi??
        ptheta = 2. - (9./8) * (1./Qtheta)
        ppi = 2. - (9./8) * (1./Qpi)

        # write estimates to logfile
        logF = open(logfile, 'a')   # append to existing chrX logfile
        outstr = 'Qtheta={} using alpha={}: pHat = {}\n'.format(Qtheta, alpha, ptheta)
        logF.write(outstr)
        outstr = 'Qpi={} using alpha={}: pHat = {}\n'.format(Qpi, alpha, ppi)
        logF.write(outstr)
        logF.close()
        return          # NOTE this is all for snm so skipping rest of function


    ### get best auto opt params ###
    if autoOptNum:       # read autoOptNum file if passed in [to test]
        outfileA = '{}{}.txt'.format(dataDict['filePrefix'], autoOptNum)
        modelfileA = '{}{}.dadi'.format(dataDict['filePrefix'], autoOptNum)
        # TODO check if file exists and if not set outfileA to None
    else:             # automatically choose opt with best likelihood.
        outfileA, modelfileA = chooseBestOpt(dataDict['filePrefix'], funcName, dataDict['numOpts'])
    if outfileA is None:
        sys.exit('no output files exist for autosomal models so cannot perform lrt.')

    popt, ll_opt, theta, autoParamDict = read1DParams(funcName, outfileA)  # need params with correct names from auto

    fitDict = {}     # key: chrX model name, value: model fit
    lrtModel = {}     # key: model name; value: list of fixed params defining model
    # for each demog model, defined nested X models based on parameters and make a dictionary
    # TODO make a function? takes funcName, returns lrtModel
    if funcName == 'two_epoch':
        header = 'nu T theta ll_opt\n'
        nu = autoParamDict['nu']
        T = autoParamDict['T']
        lrtModel['X0'] = array([nu, 4./3*T])
        lrtModel['X1'] = array([nu, None])
        lrtModel['X2'] = array([None, None])

    elif funcName == 'growth':
        header = 'nu T theta ll_opt\n'
        nu = autoParamDict['nu']
        T = autoParamDict['T']
        lrtModel['X0'] = array([nu, 4./3*T])
        lrtModel['X1'] = array([nu, None])
        lrtModel['X2'] = array([None, None])

    elif funcName == 'bottlegrowth':
        header = 'nuB nuF T theta ll_opt\n'
        nuB = autoParamDict['nuB']
        nuF = autoParamDict['nuF']
        T = autoParamDict['T']
        lrtModel['X0'] = array([nuB, nuF, 4./3*T])
        lrtModel['X1'] = array([nuB, nuF, None])
        lrtModel['X2'] = array([None, nuF, None])    # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None])

    elif funcName == 'three_epoch':
        header = 'nuB nuF TB TF theta ll_opt\n'
        nuB = autoParamDict['nuB']
        nuF = autoParamDict['nuF']
        TB = autoParamDict['TB']
        TF = autoParamDict['TF']
        lrtModel['X0'] = array([nuB, nuF, 4./3*TB, 4./3*TF])
        lrtModel['X1'] = array([nuB, nuF, None, None])
        lrtModel['X2'] = array([None, nuF, None, 4./3*TF])    # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None, None])

    elif funcName == 'twoEpochGrowth':
        header = 'nuB nuF TB TF theta ll_opt\n'
        nuB = autoParamDict['nuB']
        nuF = autoParamDict['nuF']
        TB = autoParamDict['TB']
        TF = autoParamDict['TF']
        lrtModel['X0'] = array([nuB, nuF, 4./3*TB, 4./3*TF])
        lrtModel['X1'] = array([nuB, nuF, None, None])
        lrtModel['X2'] = array([None, nuF, None, 4./3*TF])    # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None, None])

    elif funcName =='threeEpochGrowth':
        header = 'nuB nuF nuC TB TF TC theta ll_opt\n'
        nuB = autoParamDict['nuB']
        nuF = autoParamDict['nuF']
        nuC = autoParamDict['nuC']
        TB = autoParamDict['TB']
        TF = autoParamDict['TF']
        TC = autoParamDict['TC']        
        lrtModel['X0'] = array([nuB, nuF, nuC, 4./3*TB, 4./3*TF, 4./3*TC])
        lrtModel['X1'] = array([nuB, nuF, nuC, None, None, None])
        lrtModel['X2'] = array([None, nuF, nuC, None, 4./3*TF, 4./3*TC])    # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None, None, None, None])

    elif funcName == 'gravel_eur_single_pop':
        header = 'nuAf0 nuB nuEu0 nuEu1 TAf TEuAs TB nuEu2 timegrowthEu theta ll_opt\n'
        nuAf0 = autoParamDict['nuAf0']
        nuB = autoParamDict['nuB']
        nuEu0 = autoParamDict['nuEu0']
        nuEu1 = autoParamDict['nuEu1']
        nuEu2 = autoParamDict['nuEu2']
        TAf = autoParamDict['TAf']
        TEuAs = autoParamDict['TEuAs']
        TB = autoParamDict['TB']
        timegrowthEu = autoParamDict['timegrowthEu']
        lrtModel['X0'] = array([nuAf0, nuB, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, 4./3*TB, nuEu2, 4./3*timegrowthEu])
        lrtModel['X1'] = array([nuAf0, nuB, nuEu0, nuEu1, None, None, None, nuEu2, None])
        lrtModel['X2'] = array([nuAf0, None, nuEu0, nuEu1, 4./3*TAf, 4./3*TEuAs, None, nuEu2, 4./3*timegrowthEu])  # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None, None, None, None, None, None, None])

    else:
        sys.exit('2: func name invalid: {}'.format(funcName))

    ## fit all chrX models
    allF = open(dataDict['allOutfile'], 'w')
    header = 'chrom model LL theta phat popt\n'
    allF.write(header)
    outstr = 'A {} {} {} --- {}'.format(outfileA, ll_opt, theta, repr(popt)) + '\n'
    allF.write(outstr)   # write out auto demog model ests

    for modelType in sorted(lrtModel):    # sorted keys so run models in order of incr cplx
        fixed_params = lrtModel[modelType]
        outBase = outBaseX.replace('MODELTYPE', modelType)
        outfile = outBase + '.txt'
        modelfile = outBase + '.dadi'
        logfile = outBase + '.log'
        popt, ll_opt, theta = fit1DModel(infileX, outfile, modelfile, funcName=funcName, isCluster=isCluster, fixed_params=fixed_params, logfile=logfile)
        phat = estP(infileX, modelfile, logfile, infileA, modelfileA, dataDict['muA'], dataDict['LA'], dataDict['LX'], dataDict['alpha'])

        fitDict[modelType] = {'infile': infileA, 'modelfile': modelfile, 'popt': popt, 'll': ll_opt, 'theta': theta, 'phat': phat}
        outstr = 'model {}: pHat = {}, theta = {}'.format(modelType, phat, theta)
        print outstr
        outstr = 'X {} {} {} {} {}'.format(modelType, ll_opt, theta, phat, repr(popt)) + '\n'
        allF.write(outstr)
        allF.flush()

    # write dict to file: pickle, and human readable. can open easily to re-estimate p and convert to physical params
    paramF = open(dataDict['paramfile'], 'wb')
    pickle.dump(fitDict, paramF)
    paramF.close()

    # TODO these are two epoch only. update the likelihood differences: specific to each model. use a dictionary?
    # LL_2_1 = LL2 - LL1
    # LL_1_0 = LL1 - LL0
    # outstr = 'LL2 = {0}, LL1 = {1}, LL0 = {2}, ll.2.1 = {3}, ll.1.0 = {4}'.format(LL2, LL1, LL0, LL_2_1, LL_1_0)
    # allF.write(outstr + '\n')
    allF.close()


def get2DNestedModels(funcName, autoParamDict):  ## TODO NOT DONE
    """
    for each demog model, defined nested X models based on parameters and make a dictionary
    """
    lrtModel = {}     # key: model name; value: list of fixed params defining model
    if funcName == 'bottlegrowth':
        # params = (nuB,nuF,T)
        nuB = autoParamDict['nuB']
        nuF = autoParamDict['nuF']
        T = autoParamDict['T']
        lrtModel['X0'] = array([nuB, nuF, 4./3*T])
        lrtModel['X1'] = array([nuB, nuF, None])
        lrtModel['X2'] = array([None, nuF, None])    # allow for sex-biased bottleneck
        lrtModel['X3'] = array([None, None, None])
#    elif funcName == 'bottlegrowth_split_mig':
#        # params = (nuB,nuF,m,T,Ts)
#        nuB = autoParamDict['nuB']
#        nuF = autoParamDict['nuF']
#        m = autoParamDict['m']
#        Ts = autoParamDict['Tp']
#        T = autoParamDict['T']
        # TODO write in models

    elif funcName == 'prior_onegrow_mig':
        # params = (nu1F, nu2B, nu2F, m, Tp, T)
        nu1F = autoParamDict['nu1F']
        nu2B = autoParamDict['nu2B']
        nu2F = autoParamDict['nu2F']
        m = autoParamDict['m']
        Tp = autoParamDict['Tp']
        T = autoParamDict['T']

        lrtModel['X0'] = array([nu1F, nu2B, nu2F, m, 4./3*Tp, 4./3*T])
        lrtModel['X1'] = array([nu1F, nu2B, nu2F, m, None, None])
        lrtModel['X2'] = array([nu1F, None, nu2F, m, 4./3*Tp, None])  # allow for sex-biased bottleneck: nuB and time since bottleneck free
        lrtModel['X3'] = array([None, None, None, None, None, None])

    else:
        sys.exit('2: func name invalid: {}'.format(funcName))
    return lrtModel


def lrt2DModel(jsonFile, isCluster=True, autoOptNum=None):   ## TODO NOT DONE
    """
    runs nested LRTs on any 2D model
    one case per demographic function
    """
    # read params from jsonFile: has auto and X file names, constants
    dataDict = readConstantsFromFile(jsonFile)
    # put in local file space - prob a bad idea to do all
    infileX = dataDict['infileX']
    infileA = dataDict['infileA']
    funcName = dataDict['funcName']
    outBaseX = dataDict['outBaseX']

    ### get best auto opt params ###
    # TODO read autoOptNum file if passed in

    # automatically choose opt with best likelihood.
    outfileA, modelfileA = chooseBestOpt(dataDict['filePrefix'], funcName, dataDict['numOpts'])
    if outfileA is None:
        sys.exit('no output files exist for autosomal models so cannot perform lrt.')

    popt, ll_opt, theta, autoParamDict = read1DParams(funcName, outfileA)  # need params with correct names from auto

    fitDict = {}     # key: chrX model name, value: model fit
    lrtDict = get2DNestedModels(funcName, autoParamDict)

    ## fit all chrX models
    allF = open(dataDict['allOutfile'], 'w')
    header = 'chrom model LL theta phat popt\n'
    allF.write(header)
    outstr = 'A {} {} {} --- {}'.format(outfileA, ll_opt, theta, repr(popt)) + '\n'
    allF.write(outstr)   # write out auto demog model ests

    for modelType in sorted(lrtModel):    # sorted keys so run models in order of incr cplx
        fixed_params = lrtModel[modelType]
        outBase = outBaseX.replace('MODELTYPE', modelType)
        outfile = outBase + '.txt'
        modelfile = outBase + '.dadi'
        logfile = outBase + '.log'
        popt, ll_opt, theta = fit2DModel(infileX, outfile, modelfile, funcName=funcName, isCluster=isCluster, fixed_params=fixed_params, logfile=logfile)
        phat = estP(infileX, modelfile, logfile, infileA, modelfileA, dataDict['muA'], dataDict['LA'], dataDict['LX'], dataDict['alpha'])

        fitDict[modelType] = {'infile': infileA, 'modelfile': modelfile, 'popt': popt, 'll': ll_opt, 'theta': theta, 'phat': phat}
        outstr = 'model {}: pHat = {}, theta = {}'.format(modelType, phat, theta)
        print outstr
        outstr = 'X {} {} {} {} {}'.format(modelType, ll_opt, theta, phat, repr(popt)) + '\n'
        allF.write(outstr)
        allF.flush()

    # write dict to file: pickle, and human readable. can open easily to re-estimate p and convert to physical params
    paramF = open(dataDict['paramfile'], 'wb')
    pickle.dump(fitDict, paramF)
    paramF.close()

    # TODO these are two epoch only. update the likelihood differences: specific to each model. use a dictionary?
    # LL_2_1 = LL2 - LL1
    # LL_1_0 = LL1 - LL0
    # outstr = 'LL2 = {0}, LL1 = {1}, LL0 = {2}, ll.2.1 = {3}, ll.1.0 = {4}'.format(LL2, LL1, LL0, LL_2_1, LL_1_0)
    # allF.write(outstr + '\n')
    allF.close()



# adding bottlegrowth
def fitThreeEpoch(infile, outfile, modelfile, isCluster=True, nuBFixed=None, nuFFixed=None, tauBFixed=None, tauFFixed=None, minGrid=100, maxiter=100, funcName='threeEpoch'):  # DEFAULT PARAMS TODO remove? TODO make min grid increment by proportions, not absolute amounts
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [minGrid, minGrid+10, minGrid+20]    ## user specified grid on 2014-03-12
    if funcName == 'threeEpoch':
        func = dadi.Demographics1D.three_epoch
    elif funcName == 'bottlegrowth':
        func = dadi.Demographics1D.bottlegrowth
    else:
        sys.exit('func name invalid to fit three epoch model: {}'.format(funcName))

    # from manual: params = (nuB,nuF,TB,TF)
    # nuB = Nbottle / Nancient
    # nuF = Ncurrent / Nancient
    # TB: length of bottle in 2*Nanc gens
    # TF: time since bottle recovery in 2*Nanc gens
    upper_bound = [1, 10e3, 1, 1]
    # true params for this SFS: [0.01111, 1, 6.25e-4, 0.014]
    params = array([0.1, 5, 0.5, 0.1])
    lower_bound = [1e-8, 1e-8, 1e-8, 1e-8]


    fixed_params=[nuBFixed, nuFFixed, tauBFixed, tauFFixed]
    if nuBFixed != None:
        if (nuBFixed < lower_bound[0]) or (nuBFixed > upper_bound[0]):
            print 'ERROR nuBFixed {0} out of bounds: {1} {2}. exiting'.format(nuBFixed, lower_bound[0], upper_bound[0])
            exit()
        if (nuFFixed < lower_bound[1]) or (nuFFixed > upper_bound[1]):
            print 'ERROR nuFFixed {0} out of bounds: {1} {2}. exiting'.format(nuFFixed, lower_bound[1], upper_bound[1])
            exit()

    if tauBFixed != None:
        if (tauBFixed < lower_bound[2]) or (tauBFixed > upper_bound[2]):
            print 'ERROR tauBFixed {0} out of bounds: {1} {2}. exiting'.format(tauBFixed, lower_bound[2], upper_bound[2])
            exit()
        if (tauFFixed < lower_bound[3]) or (tauFFixed > upper_bound[3]):
            print 'ERROR tauFFixed {0} out of bounds: {1} {2}. exiting'.format(tauFFixed, lower_bound[3], upper_bound[3])
            exit()

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    ll_model = dadi.Inference.ll_multinom(model, data)
    p0 = dadi.Misc.perturb_params(params, fold=2, upper_bound=upper_bound)   # TODO why no upper bound?
    if isCluster == True:         # buffer IO
        flush_delay = 1        # could play with this
    else:
        flush_delay = 0.5      # default
    popt = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
                                      lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=len(params),
                                       maxiter=maxiter, output_file=outfile,
                                       flush_delay=flush_delay, fixed_params=fixed_params)
    # popt_sm = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
#                                     lower_bound=lower_bound,
#                                      upper_bound=upper_bound,
#                                      verbose=len(params),
#                                      maxiter=10, output_file=outfile,
#                                      flush_delay=flush_delay)
    # model_sm = func_ex(popt_sm, ns, pts_l)
    # ll_sm = dadi.Inference.ll_multinom(model_sm, data)

    outfile = open(outfile, 'a')
    outfile.write('Optimized parameters ' + repr(popt) + '\n')
    model = func_ex(popt, ns, pts_l)
    ll_opt = dadi.Inference.ll_multinom(model, data)
    theta = dadi.Inference.optimal_sfs_scaling(model, data)

    outfile.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(theta) + '\n')
    outfile.write('{} {} {} {} {} {}\n'.format(popt[0], popt[1], popt[2], popt[3], theta, ll_opt))   # hard-coded for two params
    outfile.close()
    modelfile = open(modelfile, 'w')         # write SFS expected under demographic model and params to file
    model.to_file(modelfile)
    modelfile.close()
    return (popt, ll_opt, theta)



# list of Demographics1D.py fns: two_epoch, growth, bottlegrowth, three_epoch
# TODO take multi or pois as a param
# TODO check if fixed values outside of bounds as a vector, not one at at time
# TODO option to make min grid increment by proportions, not absolute amounts
# could use in a bottle call: nuBFixed=None, nuFFixed=None, tauBFixed=None, tauFFixed=None,
# TODO BUG params, fixed_params, lower_bound, and upper_bound not used!!
def fit1DModel(infile, outfile, modelfile, funcName, isCluster=True, minGrid=100, maxiter=100, perturb_fold = 2, lower_bound=None, upper_bound=None, params=None, fixed_params=None, logfile=None, timescale=None):
    # TODO make reasonable upper and lower bounds
    # TOOD set defaults if not passed in: check values passed in
    """
    one case per demographic function
    """
    if logfile is None:     # for back compatability to when I wrote evthign to outfile
        logfile = outfile

    if timescale:
        dadi.Integration.timescale_factor = timescale
    # if minGrid > 60:   # HARDCODED NOTE
        # dadi.Integration.timescale_factor = 1e-4   # re-sets default which is ok for up to 60 grid points. might need larger for very large grid sizes

    # if funcName == 'snm':   # special case: can't optimize. Just generate snm fs with appropriate value of theta
    #  func = dadi.Demographics1D.snm
    if funcName == 'two_epoch':             
        numParams = 2                             # nu,T = params
        func = dadi.Demographics1D.two_epoch
        if upper_bound is None:        
            upper_bound = [10e4, 10.]
        if lower_bound is None:        
            lower_bound = [1e-8, 1e-8]
        if params is None:
            params = array([5., 0.1])

    elif funcName == 'growth':              
        numParams = 2                             # nu,T = params
        func = dadi.Demographics1D.growth
        if upper_bound is None:
            upper_bound = [10e4, 10.]
        if lower_bound is None:                    
            lower_bound = [1e-8, 1e-8]
        if params is None:        
            params = array([5., 0.1])

    elif funcName == 'bottlegrowth':    
        numParams = 3                               # nuB,nuF,T = params
        func = dadi.Demographics1D.bottlegrowth
        if upper_bound is None:
            upper_bound = [1., 10e4, 1.]
        if params is None:        
            params = array([0.1, 5., 0.5])
        if lower_bound is None:                    
            lower_bound = [1e-8, 1e-8, 1e-8]

    elif funcName == 'three_epoch':
        numParams = 4                                 # nuB,nuF,TB,TF = params
        func = dadi.Demographics1D.three_epoch
        if upper_bound is None:
            upper_bound = [1., 10e4, 1., 1]        # NOTE this constrains second size to be smaller or sames as ancestral size
        if params is None:
            params = array([0.1, 5., 0.5, 0.1])
        if lower_bound is None:
            lower_bound = [1e-8, 1e-8, 1e-8, 1e-8]
    elif funcName == 'twoEpochGrowth':
        numParams = 4                           # nuB,nuF,TB,TF = params
        func = twoEpochGrowth
        if upper_bound is None:
            upper_bound = [1., 10e4, 1., 1.]        # NOTE this constrains second size to be smaller or sames as ancestral size
        if params is None:        
            params = array([0.1, 5., 0.5, 0.1])
        if lower_bound is None:                    
            lower_bound = [1e-8, 1e-8, 1e-8, 1e-8]
    elif funcName == 'threeEpochGrowth':
        numParams = 6                            # nuB,nuF,nuC,TB,TF,TC = params
        func = threeEpochGrowth
        if upper_bound is None:
            upper_bound = [1., 10e4, 10e6, 1., 1., 1.]
        if params is None:        
            params = array([0.1, 5., 50, 0.5, 0.1, 0.1])
        if lower_bound is None:                    
            lower_bound = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    elif funcName == 'gravel_eur_single_pop':            # params = (nuB, nuF, nuC, TB, TF)
        numParams = 5
        func = gravel_eur_single_pop        # params (nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu)
        # start opt at Simon's CEU estimates
        nuAf0=1.983; nuB=0.25; nuEu0=0.141; nuEu1=1.274; TAf=0.4054; TEuAs=0.0630; TB=0.1397; nuEu2=70.137; timegrowthEu=0.00171
        if params is None:        
            params = array([nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu])
        if upper_bound is None:
            upper_bound = [1e4, 1e2, 1e4, 1e4, 3, 3, 3, 1e4, 3]
        if lower_bound is None:                    
            lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 0, 0, 0, 1e-2, 0]

    else:
        sys.exit('1: func name invalid: {}'.format(funcName))

    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [minGrid, minGrid+10, minGrid+20]
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    ll_model = dadi.Inference.ll_multinom(model, data)   # lik of starting point - necessary?
    p0 = dadi.Misc.perturb_params(params, fold=perturb_fold, lower_bound=lower_bound, upper_bound=upper_bound)
    if isCluster == True:         # buffer IO
        flush_delay = 1           # could play with this  # TODO make arg?
    else:
        flush_delay = 0.5         # default
    # TODO hardcoded for multi: implement choice of poisson
    popt = dadi.Inference.optimize_log_fmin(p0, data, func_ex, pts_l,
                                      lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=len(params),
                                       maxiter=maxiter, output_file=logfile,
                                       flush_delay=flush_delay, fixed_params=fixed_params)

    if logfile != outfile:
        outfile = open(outfile, 'w')
    else:
        outfile = open(outfile, 'a')     # back compat in case logfile is also output file
    model = func_ex(popt, ns, pts_l)
    ll_opt = dadi.Inference.ll_multinom(model, data)
    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    # these were previously written out
    # outfile.write('Optimized parameters ' + repr(popt) + '\n')
    # outfile.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(theta) + '\n')

    # replaced prev code with a function call
    outstr = format1DParams(funcName, popt, theta, ll_opt)
    outfile.write(outstr)
    outfile.close()

    model.to_file(modelfile)       # filename fine, doesn't need to be filehandle
    return (popt, ll_opt, theta)   # TODO maybe put in a param dict specific to each demog fn, or named list/tuple?



###### TODO Write an LRT for bottleneck
## for each pair of auto, X:
# LL = LLA + LLX
# calc ll.2.1 == LL2 - LL1
# calc ll.1.0 == LL1 - LL0
def nestedThreeEpochModelLRT(isCluster, indir, baseA, baseX, allOutfile, minGrid=100):
    allF = open(allOutfile, 'w')
    header = 'chrom model LL nuB nuF tauB tauF theta\n'
    allF.write(header)

    # auto: just run once
    chrom = 'A'
    infile = baseA + '.dadi'
    outfile = baseA + '-dadi-model2.txt'    # single line: demog param fitting
    modelfile = baseA + '_dadi_model2_sfs.txt'
    logfile = baseA + '-dadi-model2.log'    # for demog optimization
    (poptA, ll_optA, thetaA) = fitThreeEpoch(infile, logfile, modelfile, isCluster, nuBFixed=None, nuFFixed=None, tauBFixed=None, tauFFixed=None, minGrid=minGrid)
    outstr = 'A 2 {0} {1} {2} {3} {4} {5} '.format(ll_optA, poptA[0], poptA[1], poptA[2], poptA[3], thetaA) + '\n'
    allF.write(outstr)

    modelNums = [2, 1, 0]
    for modelNum in modelNums:    # they are numbered 0, 1, 2
        chrom = 'X'
        infile = baseX + '.dadi'
        outfile = baseX + '-dadi-model' + str(modelNum) + '.txt'    # single line: demog param fitting
        modelfile = baseX + '_dadi_model' + str(modelNum) + '_sfs.txt'
        logfile = baseX + '-dadi-model' + str(modelNum) + '.log'    # for demog optimization

        if modelNum == 2:
            # model 2: run for auto, run for X with no params fixed
            (poptX, ll_optX, thetaX) = fitThreeEpoch(infile, logfile, modelfile, isCluster, nuBFixed=None, nuFFixed=None, tauBFixed=None, tauFFixed=None, minGrid=minGrid)
            LL2 = ll_optA + ll_optX

        elif modelNum == 1:
            # model 1: run for auto, run for X passing in 3/4 of nu auto
            nuBFixed = poptA[0]
            nuFFixed = poptA[1]

            (poptX, ll_optX, thetaX) = fitThreeEpoch(infile, logfile, modelfile, isCluster, nuBFixed, nuFFixed, tauBFixed=None, tauFFixed=None, minGrid=minGrid)    # TODO constrain tF so times balance
            LL1 = ll_optA + ll_optX

        elif modelNum == 0:
            # model 0: run for auto, run for X passing in 3/4 of nu auto AND 3/4 of tau auto
            nuBFixed = poptA[0]
            nuFFixed = poptA[1]
            tauBFixed = poptA[2] * 4/3
            tauFFixed = poptA[3] * 4/3

            (poptX, ll_optX, thetaX) = fitThreeEpoch(infile, logfile, modelfile, isCluster, nuBFixed, nuFFixed, tauBFixed, tauFFixed, minGrid=minGrid)    # TODO constrain tF so times balance
            LL0 = ll_optA + ll_optX

        else:
            print 'ERROR: invalid model number' + modelNum; exit()

        outstr = 'X {0} {1} {2} {3} {4} {5} {6} '.format(modelNum, ll_optX, poptX[0], poptX[1], poptX[2], poptX[3], thetaX) + '\n'
        allF.write(outstr)

    LL_2_1 = LL2 - LL1
    LL_1_0 = LL1 - LL0
    outstr = 'LL2 = {0}, LL1 = {1}, LL0 = {2}, ll.2.1 = {3}, ll.1.0 = {4}'.format(LL2, LL1, LL0, LL_2_1, LL_1_0)
    allF.write(outstr + '\n')
    allF.close()



########### dadi convenience functions #########

def make_data_dict_comments(filename, return_comments=False):
    """
    Parse SNP file and store info in a properly formatted dictionary.

    filename: Name of file to work with.

    return_comments: If true, the return value is (fs, comments), where
         comments is a list of strings containing the comments
         from the file (without #'s).

    This is specific to the particular data format described on the wiki.
    Modification for other formats should be straightforward.

    The file can be zipped (extension .zip) or gzipped (extension .gz). If
    zipped, there must be only a single file in the zip archive.

    """
    if os.path.splitext(filename)[1] == '.gz':
        import gzip
        f = gzip.open(filename)
    elif os.path.splitext(filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError('Must be only a single data file in zip '
                             'archive: %s' % filename)
        f = archive.open(namelist[0])
    else:
        f = open(filename)

    # Skip to the header
    comments = []             # this part differs from Misc.py definition
    header = f.readline()
    while header.startswith('#'):
        comments.append(header[1:].strip())
        header = f.readline()

    allele2_index = header.split().index('Allele2')

    # Pull out our pop ids
    pops = header.split()[3:allele2_index]

    # The empty data dictionary
    data_dict = {}

    # Now walk down the file
    for line in f:
        if line.startswith('#'):
            continue
        # Split the into fields by whitespace
        spl = line.split()

        data_this_snp = {}

        # We convert to upper case to avoid any issues with mixed case between
        # SNPs.
        data_this_snp['context'] = spl[0].upper()
        data_this_snp['outgroup_context'] = spl[1].upper()
        data_this_snp['outgroup_allele'] = spl[1][1].upper()
        data_this_snp['segregating'] = spl[2].upper(),spl[allele2_index].upper()

        calls_dict = {}
        for ii,pop in enumerate(pops):
            calls_dict[pop] = int(spl[3+ii]), int(spl[allele2_index+1+ii])
        data_this_snp['calls'] = calls_dict

        # We name our SNPs using the final columns
        snp_id = '_'.join(spl[allele2_index+1+len(pops):])

        data_dict[snp_id] = data_this_snp

    if not return_comments:
        return data_dict
    else:
        return data_dict, comments


def flattenGridOutputTwoEp(outfile, xopt, fopt, grid, fout, thetas):
    """
    vectorize or flatten arrays to lists and write to file to read into R
    for now: specific to two-epoch model
    """

    foutList = fout.ravel()  # to list
    nuList = grid[0].ravel()
    tauList = grid[1].ravel()
    thetaList = thetas.ravel()
    outF = open(outfile, 'w')
    header = 'nu tau theta ll\n'
    outF.write(header)
    for i in range(len(foutList)):
        outF.write( '{} {} {} {}\n'.format(nuList[i], tauList[i], thetaList[i], foutList[i]) )
    outF.close()


def flattenGridOutput(outfile, xopt, fopt, grid, fout, thetas, funcName):
    """
    vectorize or flatten arrays to lists and write to file to read into R
    note: need to transpose in R
    will fail if fn not in getFuncByName
    Inputs:
    xopt  1D array with best point
    fopt  value of function at opt point
    grid  tuple of length n. representation of evaluation grid
    fout  n-dim array. function value at grid points. likelihoods
    thetas n-length list of thetas
    """
    func, numParams, paramNames = getFuncByName(funcName)   

    foutList = [str(x) for x in fout.ravel()]  # TODO increase number of sig figs
    thetaList = [str(x) for x in thetas.ravel()]
    paramList = [None] * numParams
    for i in range(numParams):    
        paramList[i] = [str(x) for x in grid[i].ravel()]
        
    outF = open(outfile, 'w')
    for i in range(numParams):    
        outF.write(paramNames[i] + ' ' + ' '.join(paramList[i]) + '\n')
    outF.write('theta ' + ' '.join(thetaList) + '\n')    
    outF.write('ll ' + ' '.join(foutList) + '\n')        
    outF.close()
    

    
#--------- dadi snp files ---------#

def maskSnpFile(snpfile, bedfile, outfile, keepBed, idx0Chrom):
    """
    Writes out a new snp file which does not contain any snps found in bed file. Should be able to handle files with multiple chromosomes

    bedfile: contains a mask. bedfile assumptions
     (1) sorted in increasing order by starts (second col) and intervals are non-overlapping which can be achieved with bedtools merge.
     (2) no header
     (3) start is a 0-based position, end is 1-based (UCSC)

    snpfile: dadi format snp file
    keepBed: True or False. If true, keep snps in bed file; if False, exclude snps in bed file
    idx0Chrom = 0-based index of chrom column. pos idx assumed to be one more
    """

    # read bedfile into two arrays
    maskDict = {}   # key=chrom, value = two lists
    inF = open(bedfile, 'r')
    for line in inF:
        line = line.strip()
        fields = line.split('\t')
        chrom = fields[0]; start = fields[1]; end = fields[2]
        if chrom not in maskDict:
            maskDict[chrom] = {'starts': [], 'ends': []}   # TODO inefficient? if have list of chroms can do before loop
        maskDict[chrom]['starts'].append(int(start))
        maskDict[chrom]['ends'].append(int(end))
    inF.close()

    # PASS unittest for leo's filter 7 files: is br-1 == bl?
    # br = bisect.bisect_right(maskDict['X']['starts'][:5], 92788590)
    # bl = bisect.bisect_left(maskDict['X']['ends'][:5], 92788590)

    inF = open(snpfile)
    outF = open(outfile, 'w')
    while True:               # write header to outfile
        header = inF.readline()
        if not header.startswith('#'):
            break
        else:
            outF.write(header)
    outF.write(header)        # write dadi header to outfile

    # write out if chrom, pos statisfies keepBed
    for line in inF:
        line = line.strip()
        fields = line.split('\t')
        chrom = fields[idx0Chrom]
        pos = int(fields[idx0Chrom + 1])
        startIdx = bisect.bisect_right(maskDict[chrom]['starts'], pos)  # returns slice so just after bed start
        endIdx = bisect.bisect_left(maskDict[chrom]['ends'], pos)   # returns slice so just after bed end
        if (startIdx - 1) == endIdx:   # SNP falls in a bed interval. subtract 1 because this makes intervals line up. see lab notebook 3/25/15
            inBed = True
        else:
            inBed = False
        if inBed and keepBed:
            outF.write(line + '\n')
        elif not inBed and not keepBed:
            outF.write(line + '\n')
        else:
            continue
    inF.close()
    outF.close()



def makeSinglePopFs(infile, outBase, popNames, projDims, writeDim=False):
    """
    from a dadi snp file, make all single-pop fs and write to file

    infile: dadi snp file
    outBase: base for output file names. will have projDim and popName appended to it
    popNames: list of population names
    projDims: list of projection dimensions in same order as popNames
    writeDim: if True, dimension is part of outfile name. for back compat.
    """

    d = dadi.Misc.make_data_dict(infile)
    for i in range(len(popNames)):
        popName = popNames[i]
        projDim = projDims[i]        
        if writeDim:
            outfile = '{}_{}_dim{}.fs'.format(outBase, popName, projDim)
        else:
            outfile = '{}_{}.fs'.format(outBase, popName, projDim)            
        fs = dadi.Spectrum.from_data_dict(d, [popName], [projDim], polarized=False)
        fs.to_file(outfile)
    

# TODO finish writing. try taking in a serialized version of dictionary!
def edit_trinucl_context(infile, triFile, idx0Tri, idx0Chrom, outfile):
    """
    adds trinucleotide context to a dadi snp file
    infile: dadi snp file
    triFile:   has trinucl context. col1=chrom, col2=pos, col idx0+1 has trinucl context
    # stopped because there is no writer for data_dict s    """

    # read trinucl into a dictionary
    triDict = {}   # key = chrom, val = pos, val = annot
    with open(triDict, 'r') as inF:
        header = inF.readline()
        outgrp = header.split()[idx0Tri]   # name of outgroup
        for line in inF:
            line = line.strip().split()
            chrom = line[0]
            pos = line[1]


    # get type of polarization

    # go line by line in dadi snp file and edit


#--------- ms simulations ---------#
def two_epoch_mscore((nu, T)):
    """
    similar to code Ryan uses in the YRI_CEU example. single population, single size change
    scaled such that Nref is Nanc, so the theta estimated from dadi can be used
    """
    command = "-eN 0 %(nu)f -eN %(T)f 1."
    # There are several factors of 2 necessary to convert units between dadi
    # and ms.
    sub_dict = {'nu':nu, 'T':T/2}
    return command % sub_dict

def two_epoch_mscore_Ncurr((nu, T)):
    """
    my version, scaled to Ncurr. Single eN command
    """
    nuMs = 1/nu
    timeMs = T/(2*nu)
    mscore = "-eN {} {}".format(timeMs, nuMs)
    return mscore


def growth_mscore((nu, T)):
    """
    scaled to Nanc. is this sufficient or does the starting size need to specificed?
attempt 1: get theta which is a factor of nu too large
    "-eG %(T)f 0.0"   removed this because still scales overall in terms of Ncurr

attempt 2: params hit a bound
              "-eN %(T)f 1."   # sets growth rate to zero and size to 1, so is Nanc
attempt 3: added 1st -eG and "-eN 0 %(nu)f "\
    can try -N ... 0 : 
    """

    # growth rate
    alpha2 = numpy.log(nu) / T
    
    command = "-G %(alpha2)f "\
              "-eG %(T)f 0.0 "\
              "-eN %(T)f 1.0"              
              
    sub_dict = {'alpha2':2*alpha2, 'T':T/2, 'T':T/2}              
    return command % sub_dict


def three_epoch_mscore((nuB,nuF,TB,TF)):
    """
    similar to code Ryan uses in the YRI_CEU example. single population, single size change scaled such that Nref is Nanc, so the theta estimated from dadi can be used
    """
    command = "-eN 0 %(nuF)f -eN %(TF)f %(nuB)f -eN %(Tsum)f 1."
    # There are several factors of 2 necessary to convert units between dadi
    # and ms.
    sub_dict = {'nuF':nuF, 'nuB':nuB, 'TF':TF/2, 'Tsum':(TF+TB)/2}
    return command % sub_dict


def writeSeedsToFile(numIters, outfile):
    """
    numIters: integer. total number of iterations and unique seeds to write to outfile
    Writes three seeds per iter, ints for ms
    """
    seeds = {}
    for i in range(3 * numIters):    # need three seeds per iter
        # set seed: make sure it is different from previously used seeds
        seed = int(random.uniform(0,4000000))
        while seed in seeds:
            seed = int(random.uniform(0,4000000))
        seeds[seed] = 1
    seedList = []
    for seed in seeds:   # iterate over keys
        seedList.append(seed)
    if (len(seedList) % 3) != 0:
        sys.exit('error: seedList length is not a multiple of three. exiting')

    outF = open(outfile, 'w')
    for i in range(numIters):
        seedTriplet = seedList[i*3:(i*3)+3]
        outstr = [str(x) for x in seedTriplet]
        outF.write(' '.join(outstr) + '\n')
    outF.close()

def ms_command_seed(theta, ns, core, iter, recomb=0, rsites=None, seeds=None):
    """
    Generate ms command for simulation from core.
    Extend Ryan's command to take a list of three random seeds: checks that they are unique (might not be necessary)

    theta: Assumed theta
    ns: Sample sizes
    core: Core of ms command that specifies demography.
    iter: Iterations to run ms
    recomb: Assumed recombination rate
    rsites: Sites for recombination. If None, default is 10*theta.
    seeds: list of three seeds to be passed to ms
    """
    if len(ns) > 1:
        ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f -I %(numpops)i "\
                "%(sample_sizes)s %(core)s"
    else:
        ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f  %(core)s"

    if seeds:
        ms_command += " -seeds %(seeds)s"

    if recomb:
        ms_command = ms_command + " -r %(recomb)f %(rsites)i"
        if not rsites:
            rsites = theta*10
    sub_dict = {'total_chrom': numpy.sum(ns), 'iter': iter, 'theta': theta,
                'numpops': len(ns), 'sample_sizes': ' '.join(map(str, ns)),
                'core': core, 'recomb': recomb, 'rsites': rsites}

    if seeds:
        if len(seeds) != 3 or len(set(seeds)) != 3:
            sys.exit('Error: seeds must be a unique list of length 3. Got: {}'.format(seeds))
        seed1, seed2, seed3 = seeds
        sub_dict['seeds'] = '{} {} {}'.format(seed1, seed2, seed3)

    return ms_command % sub_dict



########### Developed for most recent LRT with grid search on c, alpha ########
# hard coded for two demog params
# TODO error checking here that params are read in and not missing / file was not empty / file was not missing last few formatted lines
#
def readTwoEpochParams(outfile, isGrid=False):
    """
    reads demog params nu, tau, theta, ll (, c, alpha) from file
    only reads last line of file

    isGrid: True if param file to be read is from a grid search on c and alpha (just poisX1 right now), False otherwise
    """
    outF = open(outfile, 'r')
    lines = outF.readlines()
    paramLine = lines[-1].strip()
    outF.close()
    if isGrid:
        expectedNumParams = 6
    else:
        expectedNumParams = 4

    # if line has the word array in or is not four floats separated by spaces, file is corrupt
    if 'array' in paramLine or len(paramLine.split()) != expectedNumParams:
        errstr = 'Error: this does not have the required line format of {} fields. If contains array, is a dadi optimization line. Line: {}'.format(expectedNumParams, paramLine)
        sys.exit(errstr)

    if isGrid:
        nuhat, tauhat, theta, ll_opt, c, alpha = [float(x) for x in paramLine.split(' ')]   # hard-coded for two params. from dadiLrtFunctions:gridSearchTwoEpochC
        maxParams = (theta, nuhat, tauhat)
        return (c, alpha, maxParams, ll_opt)   # NOTE multiple returns
    else:
        nuhat, tauhat, theta, ll_opt = [float(x) for x in paramLine.split(' ')]   # hard-coded for two params. from dadiLrtFunctions:fitTwoEpoch
        popt = (nuhat, tauhat)
        return (popt, ll_opt, theta)


def read1DParams(funcName, outfile, retParamLine=False, likType='multinom'):
    """
    ### TODO param string format for three_epoch_X1 and X2 break line format expected by read1DParams: ll_opt is last, and theta should be next-to-last but c  or c1,c2 are last. this means the incorrect thing is returned by popt <2015-09-26 Sat>. Is this now fixed?

    working on fixing: for anything with constraints, pass in likType as pois so the param string format is read
    
    reads demog params specified by model from file
    only reads last line of file and gets params position, not by name
    returns: popt, ll_opt, theta
    currently: only checks for number of params, does not return names; unpacking done in function-sp way
    future? return a dictionary for popt, or namped tuple
    one case per demographic function
    """

    try:
        outF = open(outfile, 'r')
    except IOError:    # TODO is this what I want??
        outstr = 'read1DParams: file with parameters not found and cannot be read from outfile={}'.format(outfile)
        print outstr
        return (None, None, None, None)
    lines = outF.readlines()
    if len(lines) < 2:
        outstr = 'read1DParams: file with parameters does not have enough lines outfile={}'.format(outfile)
        print outstr
        if retParamLine:
            return (None, None, None, None, None)
        else:
            return (None, None, None, None)        
        
    headerLine = lines[-2].strip()
    headerFields = headerLine.split(' ')   # might not need this because get paramNames below
    paramLine = lines[-1].strip()
    paramFields = [float(x) for x in paramLine.split(' ')]    
    outF.close()

    # get expected number of params and param names based on function
    func, numParams, paramNames = getFuncByName(funcName)
    if likType == 'multinom':
        expectedNumFields = numParams + 2  # add one each for theta, ll_opt
    else:
        expectedNumFields = numParams + 1  # add one for ll_opt bc theta already an explicit param
            
    # if line has the word array in or is not correct number of floats separated by spaces, file is corrupt
    if 'array' in paramLine or len(paramFields) != expectedNumFields:
        errstr = 'Error: this does not have the required line format of {} fields. If contains array, is a dadi optimization line. Line: {}'.format(expectedNumFields, paramLine)
        sys.exit(errstr)

    if likType == 'multinom':        
        ll_opt = paramFields[-1:][0]
        theta = paramFields[-2:-1][0]
        popt = paramFields[:-2]
        paramDict = dict(zip(paramNames, popt))
    else:
        ll_opt = paramFields[-1:][0]
        popt = paramFields[:-1]     # take off on for ll_opt bc theta is a paramField
        paramDict = dict(zip(paramNames, popt))
        theta = paramDict['theta']        

    if retParamLine:
        return (popt, ll_opt, theta, paramDict, paramLine)        
    else:
        return (popt, ll_opt, theta, paramDict)


## convert params: two epoch, so only one time param
def convertTwoEpochParams(Nanc, tau, g):
    timeYears = 2 * Nanc * tau * g          # in years
    return timeYears

# more general: for single-pop demog histories
# converts from genetic to physical units in generations
# params is in the same order as the demographic function: ordered tuple
def convert1DParams(funcName, params, Nanc):
    """
    one case per demographic function
    """

    if funcName == 'two_epoch':
        nu,T = params
        Tgen = 2 * Nanc * T
        paramsPhys = (nu, Tgen)
    elif funcName == 'growth':
        nu,T = params
        Tgen = 2 * Nanc * T
        paramsPhys = (nu, Tgen)
    elif funcName == 'bottlegrowth':
        nuB,nuF,T = params
        Tgen = 2 * Nanc * T
        paramsPhys = (nuB, nuF, Tgen)
    elif funcName == 'three_epoch':
        nuB,nuF,TB,TF = params
        TBgen = 2 * Nanc * TB
        TFgen = 2 * Nanc * TF
        paramsPhys = (nuB, nuF, TBgen, TFgen)
    elif funcName == 'threeEpochGrowth':
        nuB,nuF,nuC,TB,TF,TC = params
        TBgen = 2 * Nanc * TB
        TFgen = 2 * Nanc * TF
        TCgen = 2 * Nanc * TC        
        paramsPhys = (nuB, nuF, nuC, TBgen, TFgen, TCgen)
    else:
        sys.exit('3: func name invalid: {}'.format(funcName))
    return paramsPhys


def convert1DParamsGeneral(params, funcName, mu, L, timeInGens=True, yearsPerGen=25.):
    """
    converts from genetic to physical units in generations
    returns list same length as params with Nanc added to end

    """

    if funcName == 'two_epoch':
        nu,T,theta = params
        Nanc = theta / (4. * mu * L)
        if timeInGens:
            T *= 2. * Nanc
        else:
            T *= 2. * Nanc * yearsPerGen            
        paramsExpected = [nu, T, theta]
        
    elif funcName == 'growth':
        nu,T ,theta = params
        Nanc = theta / (4. * mu * L)
        if timeInGens:
            T *= 2. * Nanc
        else:
            T *= 2. * Nanc * yearsPerGen                    
        paramsExpected = [nu, T, theta]
    elif funcName == 'bottlegrowth':
        nuB,nuF,T ,theta = params
        Nanc = theta / (4. * mu * L)         
        if timeInGens:
            T *= 2. * Nanc
        else:
            T *= 2. * Nanc * yearsPerGen                    
        paramsExpected = [nuB, nuF, T, theta]
    elif funcName == 'twoEpochGrowth':
        nuB,nuF,TB,TF ,theta = params
        Nanc = theta / (4. * mu * L)         
        if timeInGens:
            TB *= 2. * Nanc
            TF *= 2. * Nanc
        else:
            TB *= 2. * Nanc * yearsPerGen
            TF *= 2. * Nanc * yearsPerGen        
        paramsExpected = [nuB, nuF, TB, TF, theta]
        
    elif funcName == 'three_epoch':
        nuB,nuF,TB,TF ,theta = params
        Nanc = theta / (4. * mu * L)         
        if timeInGens:
            TB *= 2. * Nanc
            TF *= 2. * Nanc
        else:
            TB *= 2. * Nanc * yearsPerGen
            TF *= 2. * Nanc * yearsPerGen        
        paramsExpected = [nuB, nuF, TB, TF, theta]
    elif funcName == 'threeEpochGrowth':
        nuB,nuF,nuC,TB,TF,TC ,theta = params
        Nanc = theta / (4. * mu * L)         
        if timeInGens:
            TB *= 2. * Nanc
            TF *= 2. * Nanc
            TC *= 2. * Nanc
        else:            
            TB *= 2. * Nanc * yearsPerGen
            TF *= 2. * Nanc * yearsPerGen
            TC *= 2. * Nanc * yearsPerGen
        paramsExpected = [nuB, nuF, nuC, TB, TF, TC, theta]
        
    elif funcName == 'gravel_eur_single_pop':
        nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu ,theta = params
        Nanc = theta / (4. * mu * L)
        if timeInGens:        
            TAf *= 2. * Nanc
            TEuAs *= 2. * Nanc
            TB *= 2. * Nanc
            timegrowthEu *= 2. * Nanc
        else:            
            TAf *= 2. * Nanc * yearsPerGen
            TEuAs *= 2. * Nanc * yearsPerGen
            TB *= 2. * Nanc * yearsPerGen
            timegrowthEu *= 2. * Nanc * yearsPerGen        
        paramsExpected = [nuAf0, nuB, nuEu0, nuEu1, TAf, TEuAs, TB, nuEu2, timegrowthEu, theta]
    else:
        sys.exit('3: func name invalid: {}'.format(funcName))

    paramsExpected += [Nanc]
    return paramsExpected




# TODO add sig figs for each field?
def format1DParams(funcName, popt, theta, ll_opt, multinom=True):
    """
    convenience function for single-pop demographic histories
    returns a string with newlines to write directly to file
    one case per demographic function

    multinom: if true, theta is not in popt written out separatly

later encoding: funcName does not need to be hardcoded below because calls getFuncByName
    
    """
    # if funcName == 'snm':
    #    header = 'popt theta ll_opt\n'    # NOTE popt should be blank
    if funcName == 'two_epoch':
        header = 'nu T theta ll_opt\n'
    elif funcName == 'growth':
        header = 'nu T theta ll_opt\n'
    elif funcName == 'bottlegrowth':
        header = 'nuB nuF T theta ll_opt\n'
    elif funcName == 'three_epoch':
        header = 'nuB nuF TB TF theta ll_opt\n'
    elif funcName == 'three_epoch_X0' or funcName == 'three_epoch_X1':
        header = 'nuB nuF TB TF theta c ll_opt\n'  # theta not at end
    elif funcName == 'three_epoch_X2':
        header = 'nuB nuF TB TF theta c1 c2 ll_opt\n'  # theta not at end
    elif funcName == 'twoEpochGrowth':
        header = 'nuB nuF TB TF theta ll_opt\n'
    elif funcName =='threeEpochGrowth':
        header = 'nuB nuF nuC TB TF TC theta ll_opt\n'
    elif funcName == 'gravel_eur_single_pop':
        header = 'nuAf0 nuB nuEu0 nuEu1 TAf TEuAs TB nuEu2 timegrowthEu theta ll_opt\n'

    else:
        try:
            func, numParams, paramNames = getFuncByName(funcName)
            header = ' '.join(paramNames) + 'll_opt\n'
        except ValueError:
            print 'could not getFuncByName'
            # TODO does this exit?
            
    outstr = header
    for i in range(len(popt)):
        outstr = outstr + ' {:7f}'.format(popt[i])
    if funcName == 'three_epoch_X0' or funcName == 'three_epoch_X1' or funcName == 'three_epoch_X2' or not multinom:    # theta explicit param
        outstr = outstr + ' {:.14f}\n'.format(ll_opt)   # TODO sig figs: more than is written elsewhere
    else:            
        outstr = outstr + ' {:7f} {:.14f}\n'.format(theta, ll_opt)   # TODO sig figs: more than is written elsewhere
    return outstr



def format1DParamsUnused(funcName, params, theta, ll):
    """
    convenience function for single-pop demographic histories
    Depricated for format1DParams
    """
    outstr = 'll = {}, theta = {}'.format(ll, theta)
    if funcName == 'two_epoch':
        nu,T = params
        outstr += ', nu = {}, T = {}'.format(nu, T)
    elif funcName == 'growth':
        nu,T = params
        outstr += ', nu = {}, T = {}'.format(nu, T)
    elif funcName == 'bottlegrowth':
        nuB,nuF,T = params
        outstr += ', nuB = {}, nuF = {}, T = {}'.format(nuB, nuF, T)
    elif funcName == 'three_epoch':
        nuB,nuF,TB,TF = params
        outstr += ', nuB = {}, nuF = {}, TB = {}, TF = {}'.format(nuB, nuF, TB, TF)
    elif funcName == 'twoEpochGrowth':
        nuB,nuF,TB,TF = params
        outstr += ', nuB = {}, nuF = {}, TB = {}, TF = {}'.format(nuB, nuF, TB, TF)
    elif funcName == 'threeEpochGrowth':
        nuB,nuF,nuC,TB,TF,TC = params
        outstr += ', nuB = {}, nuF = {}, nuC = {}, TB = {}, TF = {}, TC = {}'.format(nuB, nuF, nuC, TB, TF, TC)
    else:
        sys.exit('4: func name invalid: {}'.format(funcName))
    return outstr




## estimate Nanc from fourfold data and model sfs from synon data
# chrom must be X or chrX for X;
def estNanc4D(doEstNanc, chrom, popName, optNum, fourDfile, synModelFile, mu, LfourD):
    if doEstNanc:
        fourDfs = dadi.Spectrum.from_file(fourDfile)
        synModelfs = dadi.Spectrum.from_file(synModelFile)
        thetaFourD = (fourDfs.sum() * 1.) / synModelfs.sum()
        Nanc = thetaFourD / (4* mu * LfourD)
        # print 'Estimated Nanc from 4D data: {}'.format(Nanc)
    else:
        if chrom=='X' or chrom=='chrX':
            Nanc = 7500
        else:
            Nanc = 10000
        # print 'Fixing Nanc to be {} for chrom {}:'.format(Nanc, chrom)   # debug
    return Nanc

def getMultiLikTwoEpoch(infile, outfile, modelfile, nuFixed, tauFixed, minPts):
    """
    get multi two epoch lik of params
    infile contains data SFS, modelfile gets model SFS written to it, outfile gets params written to it
    """
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [minPts, minPts+10, minPts+20]
    params = (nuFixed, tauFixed)
    func = dadi.Demographics1D.two_epoch
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    dadi.Spectrum.to_file(model, modelfile)   # write to file
    ll_opt = dadi.Inference.ll_multinom(model, data)
    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    popt = (nuFixed, tauFixed)
    outF = open(outfile, 'w')
    outF.write('Optimized parameters ' + repr(popt) + '\n')
    outF.write('Optimized log-likelihood: ' + str(ll_opt) + '; theta: ' + str(theta) + '\n')
    outF.write('{:.5f} {:.5f} {:.5f} {:.14f}\n'.format(nuFixed, tauFixed, theta, ll_opt))   # TODO incr sig figs?
    outF.close()
    return (popt, ll_opt, theta)


def getFuncByName(funcName):
    """
    returns demogrphic function, number of params, and param names based on name
    one case per demographic function
    """
    if funcName == 'two_epoch':
        func = dadi.Demographics1D.two_epoch
        paramNames = ['nu', 'T']

    elif funcName == 'growth':
        func = dadi.Demographics1D.growth
        paramNames = ['nu', 'T']

    elif funcName == 'growth_X0_P':
        func = growth_X0_P
        paramNames = ['nu', 'T', 'theta']
    elif funcName == 'growth_X1_P':
        func = growth_X1_P
        paramNames = ['nu', 'T', 'theta', 'c']
    elif funcName == 'growth_X2_P':
        func = growth_X2_P
        paramNames = ['nu', 'T', 'theta', 'c1', 'c2']

    elif funcName == 'growth_X0_M':
        func = growth_X0_M
        paramNames = ['nu', 'T']
    elif funcName == 'growth_X1_M':
        func = growth_X1_M
        paramNames = ['nu', 'T', 'c']
    elif funcName == 'growth_X2_M':
        func = growth_X2_M
        paramNames = ['nu', 'T', 'c1', 'c2']

        
    elif funcName == 'bottlegrowth':
        func = dadi.Demographics1D.bottlegrowth
        paramNames = ['nuB', 'nuF', 'T']

    elif funcName == 'three_epoch':
        func = dadi.Demographics1D.three_epoch
        paramNames = ['nuB', 'nuF', 'TB', 'TF']
    elif funcName == 'three_epoch_X0' or funcName == 'three_epoch_X0_P':
        func = three_epoch_X0
        paramNames = ['nuB', 'nuF', 'TB', 'TF', 'theta', 'c']
    elif funcName == 'three_epoch_X1' or funcName == 'three_epoch_X1_P':        
        func = three_epoch_X1
        paramNames = ['nuB', 'nuF', 'TB', 'TF', 'theta', 'c']
    elif funcName == 'three_epoch_X2' or funcName == 'three_epoch_X2_P':
        func = three_epoch_X2
        paramNames = ['nuB', 'nuF', 'TB', 'TF', 'theta', 'c1', 'c2']

        
    elif funcName == 'twoEpochGrowth':
        func = twoEpochGrowth
        paramNames = ['nuB', 'nuF', 'TB', 'TF']

    elif funcName == 'threeEpochGrowth':            # params = (nuB', ' nuF', ' nuC', ' TB', ' TF)
        func = threeEpochGrowth
        paramNames = ['nuB', 'nuF', 'nuC', 'TB', 'TF', 'TC']
    elif funcName == 'threeEpochGrowth_P':            # params = (nuB', ' nuF', ' nuC', ' TB', ' TF)
        func = threeEpochGrowth_P
        paramNames = ['nuB', 'nuF', 'nuC', 'TB', 'TF', 'TC', 'theta']
        
    elif funcName == 'gravel_eur_single_pop':
        func = gravel_eur_single_pop
        paramNames = ['nuAf0', 'nuB', 'nuEu0', 'nuEu1', 'TAf', 'TEuAs', 'TB', 'nuEu2','timegrowthEu']

    else:
        sys.exit('1: func name invalid: {}'.format(funcName))

    numParams = len(paramNames)
    return func, numParams, paramNames


def evalLikelihood(funcName, params, infile, outBase=None, minGrid=None, pts_l=None, timescale=None, likType='multi', theta=None):
    # infile, outfile, modelfile, nuFixed, tauFixed, minPts):
    #     params = (nuFixed, tauFixed)
    """
    NOTE TODO Poisson not completed so likType not used
    get multinomal likelihood of paramaters for a given demographic model
    infile contains data SFS, modelfile gets model SFS written to it, outfile gets params written to it
    outBase: prefix for output files: modelfile, outfile, logfile. if None, do not write anything out.
    func: name of demographic function
    minGrid: integer
    pts_l: a list of grid points

    
    likType: can be multi or pois
    theta: required if likType is pois
    """
    if timescale:
        dadi.Integration.timescale_factor = timescale
    func, numParams, paramNames = getFuncByName(funcName)
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes

    if pts_l and minGrid:
        raise valueError('only one of pts_l and minGrid can be specified, not both')
    if pts_l is None:
        if minGrid is None:
            minGrid = max(ns) + 10    # NOTE heuristic
        pts_l = [minGrid, minGrid+10, minGrid+20]

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(params, ns, pts_l)
    if likType == 'multi':
        ll_opt = dadi.Inference.ll_multinom(model, data)
        theta = dadi.Inference.optimal_sfs_scaling(model, data)
    else:
        ll_opt = dadi.Inference.ll(model, data)        
        if theta is None:
            theta = params[-1]   # this does not work for some fixed_theta fns

    # write params ll to file
    if outBase:
        outfile = outBase + '.txt'
        modelfile = outBase + '.dadi'
        logfile = outBase + '.log'    # not used yet
        model.to_file(modelfile)

        outstr = format1DParams(funcName, params, theta, ll_opt)
        outF = open(outfile, 'w')
        outF.write(outstr)
        outF.close()

    return (params, ll_opt, theta)



# TODO re-name vars to be simpler
# TODO verNum currently unused
# TODO alphaRange: a list of either 1 or three elements.
# alphaVals: a list of alpha values to try. passed in by callee, default is 3
def gridSearchTwoEpochC(nuA, tauA, thetaA, infile, outfile, modelfile, gridfile, verNum, startVal, endVal, numPts, muX, muA, LfourDX, LfourDA, alphaVals=[3]):
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    minPts = 200   # hardcoded
    pts_l = [minPts, minPts+10, minPts+20]
    func = two_epoch_fixed_theta
    func_ex = dadi.Numerics.make_extrap_func(func)  # could change to log version
    maxParams = None
    maxLL = None
    maxC = None
    maxAlpha = None
    llVals = []
    cVals = numpy.linspace(startVal, endVal, num=numPts, endpoint=True)
    gridF = open(gridfile, 'w')
    gridF.write('c alpha theta nu tau ll\n')
    for c in cVals:
        for alpha in alphaVals:
            muX = (2.*(2.+alpha)) / (3.*(1.+alpha)) * muA
            nuX = nuA     # fixing this to be the same for model 1
            tauX = 1./c * tauA
            thetaX = thetaA * c * muX / muA * LfourDX / LfourDA
            params = (thetaX, nuX, tauX)
            model = func_ex(params, ns, pts_l)
            ll = dadi.Inference.ll(model, data)
            llVals.append(ll)
            gridF.write( '{:.10f} {} {:.5f} {:.5f} {:.5f} {:.14f}\n'.format(c, alpha, params[0], params[1], params[2], ll))
            if maxParams is None:   # NOTE just done first time. unnecssary to check all other times
                maxParams = params
                maxLL = ll
                maxC = c
                maxAlpha = alpha
            if ll > maxLL:
                maxParams = params
                maxLL = ll
                maxC = c
                maxAlpha = alpha
    gridF.close()
    model = func_ex(maxParams, ns, pts_l)      # make a model sfs and write to file
    model = model / (maxParams[0] * 1.)        # so will have same scale as multinomial SFS
    model.to_file(modelfile)

    # write params to file in the fmt expected by readTwoEpochParams: written by fitTwoEpoch
    outF = open(outfile, 'w')
    outF.write('nu tau theta ll c alpha\n{} {} {} {} {} {}\n'.format(maxParams[1], maxParams[2], maxParams[0], maxLL, maxC, maxAlpha))
    outF.close()

    return(maxC, maxAlpha, maxParams, maxLL)



def gridSearch(infile, outfile, upper_bound, lower_bound, num_vals, funcName, isLog=True, useExtrap=True):
    """
    does not use scipy.optimize.brute because that can only take slice objects and could not do points which are equally spaced in log space but unequally spaced in linear space

    lower_bound: list of lower bounds
    upper_bound: list of upper bounds
    num_vals: list of number of values desired for each parameter

    isLog: if True, param grid is equidistant in log space. If false, in linear space.
    """

    # make a datastructure with params for each point
    ubList = [ math.log(x) for x in upper_bound ]    
    lbList = [ math.log(x) for x in lower_bound ]
    paramVals = []
    if isLog:
        for i in range(len(lbList)):
            # paramVals0 = numpy.linspace(lbList[i], ubList[i], num_vals[i])
            # linearPts = [math.exp(x) for x in paramVals0]
            # alternate
            linearPts = numpy.logspace(lbList[i], ubList[i], num_vals[i], base=math.e).tolist()   # in linear space
            paramVals.append(linearPts)
    else:
        sys.exit('Non-log version of gridSearch not yet implemented. Can use dadi.Inference.optimize_grid')
    
    # make list of points: tuple version of grid
    paramGrid = []
    import itertools
    for element in itertools.product(*paramVals):
        paramGrid.append(array(element))

    minGrid = ns[0] + 60     # make a fine grid
    dadi.Integration.timescale_factor = 1e-4
    func, numParams, paramNames = getFuncByName(funcName)   # fn fails if name invalid
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    header = '\t'.join(paramNames) + ' theta\tll\n'
    
    # eval each point and write output to file
    data = dadi.Spectrum.from_file(infile)
    ns = data.sample_sizes
    pts_l = [minGrid, minGrid+10, minGrid+20]
    pts = minGrid
    outF = open(outfile, 'w')
    outF.write(header)
    # can use regular function or extrapolating function: difference?
    llOptList = []
    for params in paramGrid:
        if useExtrap:
            model = func_ex(params, ns, pts_l)
        else:
            model = func(params, ns, pts)            
        ll_opt = dadi.Inference.ll_multinom(model, data)
        llOptList.append(ll_opt)
        theta = dadi.Inference.optimal_sfs_scaling(model, data)
        paramStr = '\t'.join( [str(x) for x in params] )
        outF.write( '{}\t{}\t{}\n'.format(paramStr, theta, ll_opt) )
    outF.close()
    
    # get index of max lik: since these are log likelihoods, take maximum
    maxIdx = llOptList.index(max(llOptList))
    xopt = paramGrid[maxIdx]
    fopt = llOptList[maxIdx]
    
    # assign xopt and fopt
    print 'MLE point on grid and likelihood: {} {}'.format(xopt, fopt)
    return xopt, fopt
