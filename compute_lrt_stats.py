from scipy.stats import chi2


def lrt(ll_null, ll_alt, df=1):
    stat = 2.0 * (ll_alt - ll_null)
    if stat < 0:
        stat = 0.0
    return stat, chi2.sf(stat, df)
