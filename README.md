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

