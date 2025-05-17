import os
import pandas as pd
import numpy as np
from scipy.special import xlogy


def read_context_data(data, ref=None, dtype="count"):
    d = pd.read_csv(data, header=None, sep=" ")
    d.columns = ["context", dtype]

    k = d.context.str.len().unique().item()
    assert (
        len(d.context.str.len().unique()) == 1
    )  # require all contexts to have equal length
    assert k % 2 != 0  # require odd context length

    k_radius = (k - 1) / 2
    d["m_base"] = d.context.str[int(k_radius)]

    if ref:
        d = d[d["m_base"].str.upper() == ref.upper()]
    else:
        assert (
            len(d.m_base.unique()) == 1
        )  # require middle base to be the same if no ref provided

    assert (
        d["m_base"].unique().item() in "ACGT"
    )  # check that middle base belongs to ACGT
    d.drop(["m_base"], axis=1, inplace=True)

    return d, k


def write_encoded_data(data, outdir, output_prefix, k, encoding):
    if not outdir:
        outdir = "./"

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{output_prefix}{k}mers_{encoding}bitOHE.tsv")
    data.to_csv(outpath, sep="\t", index=False)


def log_loss(p_preds, m, u):
    """
    Calculates log likelihood
        p_preds: rate predictions
        m: mutated counts
        u: unmutated counts
    """
    ll = 0
    for idx, p in enumerate(p_preds):
        if idx % 10000 == 0:
            print(ll)
        if idx == 1:
            print(ll)
        ll += xlogy(m[idx], p.item()) + xlogy(
            u[idx], (1 - p.item())
        )  # p = predicted rate, m = mut count, u = unmut count
    return ll


def nagelkerke_r2(N, ll0, ll):
    """
    Calculates nagelkerke_r2
        N: number of data points (sum of weights -> sum(u + m))
        ll0: null model log likelihood
        ll: model log likelihood
    """
    nk_r2 = (1 - np.exp((2 * (ll0 - ll)) / N)) / (1 - np.exp((2 * ll0) / N))
    return nk_r2
