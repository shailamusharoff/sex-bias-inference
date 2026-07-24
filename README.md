# sex-bias-inference README
The python source code for our sex-bias inference method and its documentation

Musharoff S, Shringarpure S, Bustmante CD, Ramachandran S (2019) The inference of sex-biased human demography from whole-genome data. PLOS Genetics 15(9): e1008293. https://doi.org/10.1371/journal.pgen.1008293

Please contact the first author for software: sam442@cornell.edu

All code in the vendor source directory was written by other people. 

## Setup instructions w/ UV 
- Clone the repo with `git clone git@github.com:shailamusharoff/sb-private.git`
- `cd sb-private`
- Install (uv)[https://docs.astral.sh/uv/] if you haven't already 
- `uv venv` to create a virtual environment 
- `source .venv/bin/activate` to activate the venv
- `uv pip install -r requirements.txt` to install the requirements
- Build ms with ` cd vendor/msdir && ./clms && cp ms ../../.venv/bin/ms && cd ../..` and check the install with `which ms` (should point to ~/wherever-you-have-this-repo/.venv/bin/ms if install worked)
- run `python examples/run_bottleneck_example.py` to run an example bottleneck with a population ratio of 80/20 males/females. 

## Brief Documentation 
We're assuming that you've read the paper and understand the big picture. This is just providing documentation for the functions in this repo that you may wish to use directly. 
- `fitThreeEpoch()`
    Fits X models that are constrained based on A parameters.
    newer fn with good timescale
    Parameters
    	outfileA: output of auto model fit with param ests at end
	    likType: multinomial, poisson. TODO used only for file names
	    infile:  chrX fs file
  	  multinom: relevant for three_epoch_X_all which has this as an explict param
    	test of new function lrt:three_epoch_X1
    Output: written to directory lrt_test or lrt_test_optimize_log in the same directory as infile
