"""
STUB for the original ms_functions module.

ms_simulation_two_pops imports `ms_functions as msfunc` at module top
and calls `msfunc.data_from_ms_file(..., write_kimtree=True, ...)` from
inside run_ms_simulation to write a KimTree-format .dat file alongside
the dadi .fs files. The original ms_functions.py was not included in
this repository, which made the entire ms_simulation_two_pops module
unimportable.

This stub provides the minimum surface needed to make imports succeed.
It is sufficient for the sex-bias LRT pipeline (`run_sb`), which only
consumes the .fs files; the .dat output exists in name only and should
NOT be passed to `run_kimtree`.

If you need real KimTree integration, replace this file with a converter
that reads the ms output and writes the format consumed by KimTree 2.0.
The relevant references:
  - Hudson's ms output format: section "Output format of ms" in
    https://home.uchicago.edu/~rhudson1/source/mksamples.html
  - KimTree 2.0 input format: README / user manual at
    https://www1.montpellier.inra.fr/CBGP/software/kimtree/
"""
import os
import sys


def data_from_ms_file(msfile, average=False, write_kimtree=False,
                      outfile=None, segsites=None):
    """
    Stub matching the original signature.

    Parameters mirror what ms_simulation_two_pops passes in:
      msfile         path to Hudson ms output (unused by this stub)
      average        boolean (unused by this stub)
      write_kimtree  if True and outfile is given, write a placeholder
                     marker file so the absence is obvious downstream
      outfile        placeholder .dat path
      segsites       number of segregating sites (unused by this stub)

    Returns:
      None. The original returned a parsed-data structure; nothing in
      the SB LRT pipeline reads that return value, so None is safe.
    """
    if write_kimtree and outfile is not None:
        # Write a placeholder so anyone who later inspects the file sees
        # clearly that the KimTree write was stubbed -- much better than
        # an empty file that fails silently inside `kimtree`.
        with open(outfile, 'w') as out:
            out.write('# PLACEHOLDER: ms_functions.data_from_ms_file is a stub.\n')
            out.write('# No real KimTree .dat content was written.\n')
            out.write('# Replace ms_functions.py with a real converter to use this file with `kimtree`.\n')
            out.write('# Source ms file: {}\n'.format(os.path.abspath(msfile) if msfile else 'n/a'))
            out.write('# segsites:        {}\n'.format(segsites))
        print('[ms_functions stub] KimTree .dat write skipped (placeholder at {})'.format(outfile),
              file=sys.stderr)
    return None
