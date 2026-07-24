# sex-bias-inference README
The python source code for our sex-bias inference method and its documentation

Musharoff S, Shringarpure S, Bustmante CD, Ramachandran S (2019) The inference of sex-biased human demography from whole-genome data. PLOS Genetics 15(9): e1008293. https://doi.org/10.1371/journal.pgen.1008293

Please contact the first author with questions: sam442@cornell.edu

Note: the "vendor" source directory contains code written by others, specifically the program ms, which is used to simulate data in the example.

## Setup instructions with virtual environment 
- Clone the repo with `git clone git@github.com:shailamusharoff/sb-private.git`
- `cd sb-private`
- Install [uv](https://docs.astral.sh/uv/) if you haven't already 
- `uv venv` to create a virtual environment 
- `source .venv/bin/activate` to activate the venv
- `uv pip install -r requirements.txt` to install the requirements
- Build ms with ` cd vendor/msdir && ./clms && cp ms ../../.venv/bin/ms && cd ../..` and check the install with `which ms` (should point to the location of this repo which contains .venv/bin/ms if install worked)
- run `python examples/run_bottleneck_example.py` to run an example bottleneck with a population ratio of 80/20 males/females. 

## Brief Documentation 
This section is to give you an overview of the structure of this repo. More detailed documentation can be found in comments around specified functions and by looking in `./examples`, and the paper cited above contains a full description of the method.

### Core Pipeline in `ms_simulation_two_popps.py`
- `run_ms_simulation(...)`: Simulates an X chromosomal and an autosomal site frequency spectrum (SFS) via `ms` and writes `.fs` SFS files to `outdir`. 
- `run_sb(...)`: fits autosomal model and nested X chromosomal models (X0 with no sex bias, X1 with constant sex bias, X2 with varying sex bias) and produces the sex bias Likelihood Ratio Test (LRT) outputs. 
- `run_kimtree(...)`: optional, runs external KimTree binary for alternate analysis method. 

### Building Demographic Models for Simulation
- `ms_bottle_epoch_split`: 4 epochs (two size changes, then population split) 
- `ms_bottle_split`: 3 epochs (bottleneck then population split) 
- `ms_split_bottle_split`: population split, bottleneck, then population split 

All take in the same arguments and return the corresponding ms command string. The length of times/sizes/propFemales must align with epoch count. 

### Supporting functions that can be called directly 
- `fitThreeEpoch(...)`: fits one constrained X model, and `run_sb()` calls it three times, but you can call it once if you want a single model 
- `lrt.fitThreeEpoch(...)`: Standalone 3 epoch autosomal fit. 
- `lrt.read1DParams`: parse original params, LL, and theta from a fit's `.out` file. Used to compute LRT programatically. 
- `plotSFS(...)`: Plot A vs X spectra from `.fs` files. 
