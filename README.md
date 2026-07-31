# sex-bias-inference README
The python source code for our sex-bias inference method and its documentation

Musharoff S, Shringarpure S, Bustamante CD, Ramachandran S (2019) The inference of sex-biased human demography from whole-genome data. PLOS Genetics 15(9): e1008293. https://doi.org/10.1371/journal.pgen.1008293

Please contact the first author with questions: sam442@cornell.edu

Thank you to Zander Schatzberg for help with modernizing this code!

Note: the "vendor" source directory contains code written by others, specifically Hudson's program `ms`, which is used to simulate data in the example.

## Setup instructions with virtual environment + Running Example 
- Clone the repo with `git clone git@github.com:shailamusharoff/sb-private.git`
- `cd sb-private`
- Install [uv](https://docs.astral.sh/uv/) if you haven't already 
- `uv venv` to create a virtual environment 
- `source .venv/bin/activate` to activate the venv
- `uv pip install -r requirements.txt` to install the requirements
- Build ms with ` cd vendor/msdir && ./clms && cp ms ../../.venv/bin/ms && cd ../..` and check the install with `which ms` (should point to the location of this repo which contains .venv/bin/ms if the install worked). This will throw some warnings, but they aren't problematic. 
- run `python examples/run_bottleneck_example.py` to run an example with a male-biased bottleneck (proportion of females 0.2) and no sex-bias outside the bottleneck (proportion of females 0.5).

## Brief Documentation 
This section is an overview of the structure of this repo. More detailed documentation can be found in comments around specified functions and by looking in `./examples`, and the paper cited above contains a full description of the method.

The input data are an autosomal and X-chromosmal 1-dimensional (1D) data-format frequency spectrum files (`.fs`). See the dadi documention for this format: https://github.com/RyanGutenkunst/dadi

The pipeline below gives the option of simulating data with the program ms.

In either case, a demographic model must be specified; the default model in the example below is a three-epoch model.

The output consists of likelihood ratio tests for sex-bias along with expected site freqeuency spectra for each model.

### Core Pipeline in `ms_simulation_two_pops.py`
- `run_ms_simulation(...)`: [start here if simulating data] Simulates an X chromosomal and an autosomal site frequency spectrum (SFS) via `ms` and writes `.fs` SFS files to `outdir`. 
- `run_sb(...)`: [start here if using your own data] Fits autosomal model and nested X chromosomal models (X0 with no sex bias, X1 with constant sex bias, X2 with varying sex bias) and produces the sex bias Likelihood Ratio Test (LRT) outputs. Both input files, fsfileA (autosomal) and fsfileX (X-chromosomal), must be 1D dadi-format frequency spectrum files (`.fs`).
- `run_kimtree(...)`: [optional] Runs external KimTree binary for alternate analysis method. 

### Simulating data with ms
- `ms_bottle_epoch_split`: 4 epochs (two size changes, then population split) 
- `ms_bottle_split`: 3 epochs (bottleneck then population split) 
- `ms_split_bottle_split`: population split, bottleneck, then population split 

All take in the same arguments and return the corresponding ms command string. The length of times/sizes/propFemales must align with epoch count. 

### Supporting functions that can be called directly 
- `fitThreeEpoch(...)`: fits one constrained X model. `run_sb()` calls it three times, once for each relevant X model, but you can call it once if you want to fit a single model. 
- `lrt.fitThreeEpoch(...)`: standalone 3-epoch autosomal fit. 
- `lrt.read1DParams`: parse original params, LL, and theta from a fit's `.out` file. Used to compute LRT programmatically. 
- `plotSFS(...)`: Plot autosomal (A) vs X chromosomal (X) site frequency spectra from `.fs` files. 
