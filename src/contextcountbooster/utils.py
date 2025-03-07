import os
import pandas as pd

def read_context_data(data, ref = None, dtype = "count"):
    
    d = pd.read_csv(data, header = None, sep = " ")
    d.columns = ["context", dtype]
    
    k = d.context.str.len().unique().item()
    assert(len(d.context.str.len().unique()) == 1) # require all contexts to have equal length
    assert(k % 2 != 0) # require odd context length
    
    k_radius = (k-1)/2
    d['m_base'] = d.context.str[int(k_radius)]

    if ref: 
        d = d[d["m_base"].str.upper() == ref.upper()]
    else: 
        assert(len(d.m_base.unique()) == 1) # require middle base to be the same if no ref provided
    
    assert(d["m_base"].unique().item() in "ACGT") # check that middle base belongs to ACGT
    d.drop(['m_base'], axis=1, inplace=True)

    return d, k

def write_encoded_data(data, outdir, data_type, k, ref, val_frac, encoding):
    if not outdir:
        outdir = "./"

    if data_type == "train":
        pct = f"{(int((1-val_frac)*100))}pct"
    elif data_type == "val":
        pct = f"{(int(val_frac*100))}pct"
    else: 
        pct = ""
    
    os.makedirs(outdir, exist_ok=True) 
    outpath = os.path.join(outdir, 
                           f"{data_type}{pct}_ref{ref}_{k}mers_{encoding}bitOHE.tsv")
    data.to_csv(outpath, sep='\t', index = False)