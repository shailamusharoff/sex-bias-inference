import sys
import bisect
import math

# FIX: open() calls below were leaking file handles; switched to `with` blocks
# so files close deterministically even on exceptions.

def test():
    print('import successful')


def readHapmap(infile, hasHeader=True):
    """
    read hapmap map into a data stucture. list of ordered tuples

    infile: input file for a single chrom from Hapmap. assumes file is sorted by ascending position and has unique entries
    hasHeader: boolean. if True one header line is skipped
    returns: list of tuples
    """
    posMap = []

    # FIX: use `with` so file closes on exception (was leaking on parse errors).
    with open(infile, 'r') as inF:
        if hasHeader:
            header = inF.readline()
        for line in inF:
            line = line.strip()
            phys, combined, genetic = line.split(' ')
            posMap.append((int(phys), float(genetic)))  # append tuple to end of list

    return posMap

# DONE deal with boundary conditions for physPos.
# check if i is 0, length of phys, or unfined
def estPhysPos(posMap, phys, genet, physPos):
    """
    get approx the genetic pos of a physical snp using a linear regression
    posMap: list of tuples (physical, genetic)   e.g. Hapmap
    physPos: physical position in bp

    returns: estimated genetic position in cM
    note: not using raise ValueError because happens if i is 0 or len(phys), which I am setting manually
    """

    i = bisect.bisect_right(phys, physPos)
    if i == 0:             # boundary condition
        geneticPos = -1
    elif i == len(phys):   # boundary condition
        geneticPos = -1
    else:                  # do linear regression to estimate value
        posInterval = (posMap[i-1], posMap[i])
        phys1 = posInterval[0][0]
        genetic1 = posInterval[0][1]
        phys2 = posInterval[1][0]
        genetic2 = posInterval[1][1]
        # FIX: guard against duplicate physical positions (phys2 == phys1) which
        # would raise ZeroDivisionError. Docstring says the map should be unique,
        # but we now fail loudly with a useful error rather than crashing on /0.
        if phys2 == phys1:
            raise ValueError(
                'Duplicate physical position {} in genetic map (entries {} and {})'.format(
                    phys1, i-1, i))
        slope = (genetic2 - genetic1) / (phys2 - phys1)
        geneticPos = genetic1 + slope * (physPos - phys1)
        if geneticPos < genet[0] or geneticPos > genet[len(genet)-1]:
            # FIX: warnings now go to stderr so they don't get swallowed by stdout
            # capture or interleaved with data output.
            print('Warning: predicted genetic position is beyond the bounds of genetic map\n',
                  file=sys.stderr)

    return(geneticPos)


def byteify(input):
    """
    http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
    converts unicode to byte strings. for use with json dictionaries
    """
    # FIX: Python 2 -> Python 3 port. The original purpose of this function
    # was to convert json.load's `unicode` output to Py2 `str` (bytes) so
    # callers could index dicts with bytes keys. In Py3, json.load already
    # returns `str`-keyed dicts and downstream callers (e.g.
    # readConstantsFromFile -> dataDict['infileX']) index with `str`, so the
    # right Py3 behavior is to leave the structure unchanged. Encoding str
    # to bytes here would silently break all those `dataDict['...']` lookups.
    return input


def to_precision(x,p):
    """
    From http://randlet.com/blog/python-significant-figures-format/
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

    
################## functions to work on files  ##################

def getMean(logfile, colNum):
    """
    logfile: file with value of interest in a given column
    colNum: 0-based column number index
    output: single mean
    """
    ctr = 0
    currSum = 0
    # FIX: use `with` so the file handle closes deterministically (was leaking).
    with open(logfile) as fh:
        for line in fh:
            ctr += 1
            currSum += float(line.split('\t')[colNum])
    # FIX: guard against an empty input file; previously raised an opaque
    # ZeroDivisionError. Now we surface the actual problem.
    if ctr == 0:
        raise ValueError('getMean: no data lines in {}'.format(logfile))
    meanVal = currSum / ctr
    return meanVal



def convertBkgdfileToBed(chrom, infile, outfile):
    """
    custom function to convert a mcvicker background selection score file to a bed file

    infile: background selection score from 
    http://www.phrap.org/othersoftware.html

    outfile: tab-delimited bedfile of scores
    
    input for chrom 22:
    835 18021
    830 18467
    824 18015

    output: bed file. s0-based start, 1-based end. so, end is the first 0-based base not in the range.
    22 0 18021 835
    11 18021 36488 830
    """

    startPos = 0
    with open(outfile, 'w') as outF:
        for line in open(infile, 'r'):
            line = line.strip()
            B, length = line.split()
            endPos = startPos + int(length)

            # write line
            outstr = '{}\t{}\t{}\t{}\n'.format(chrom, startPos, endPos, B)
            outF.write(outstr)
            
            # update both
            startPos = endPos
