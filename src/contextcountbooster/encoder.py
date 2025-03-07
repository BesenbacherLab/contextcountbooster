import pandas as pd
import numpy as np
from contextcountbooster.utils import read_context_data
from contextcountbooster.utils import write_encoded_data


encoding_7bit = {"A": "1000111", # encodes iupac: ACGTMRW; where M = AC; R = AG; W = AT
                "C": "0100100", 
                "G": "0010010", 
                "T": "0001001"}

encoding_4bit = {"A": "1000", # encodes iupac: ACGT
                 "C": "0100", 
                 "G": "0010", 
                 "T": "0001"}


class OneHotEncoder:
    def __init__(self,
                 counts, 
                 weights, 
                 output_dir = None,
                 encoding = 7,
                 ref = None, 
                 pseudocount = 1,
                 train_val_split = False,
                 val_frac = 0.1,
                 ):
        
        # read in count data
        self.counts, k1 = read_context_data(counts, ref, dtype = "count")
        self.weights, k2 = read_context_data(weights, ref, dtype = "weight")
        assert(k1 == k2)
        self.k = k1
        self.output_dir = output_dir
        self.encoding = encoding
        self.ref = ref
        self.pseudocount = pseudocount
        self.train_val_split = train_val_split
        self.val_frac = val_frac


    def encode(self):
        
        # combine counts and weights
        d = self.combine_data()

        # add pseudocount and pseudoweight to each observation
        pseudoweight = self.get_pseudocount(d)
        d["count"] = d["count"] + self.pseudocount
        d["weight"] = d["weight"] + pseudoweight
        
        # one hot encode
        if self.encoding == 7: 
            encoder = encoding_7bit
        else: 
            encoder = encoding_4bit
        encoding = [[x for k in context for x in encoder[k]] for context in d.context.to_list()]
        df_encoding = pd.DataFrame(encoding, columns = ["p" + str(x) + "_b" + str(y) for x in range(1, self.k+1, 1) for y in range(1, self.encoding+1, 1)])

        if self.train_val_split: 
            prng = np.random.RandomState(0)

            counts = d["count"].to_list() # n_good
            weights = d["weight"].to_list() # m_total
            uncounted = [x - y for x, y in zip(weights, counts)] # n_bad
            n_samples = [int(x*self.val_frac) for x in weights] # n_sample
            
            # sample validation set counts
            val_count = prng.hypergeometric(ngood = counts, nbad = uncounted, nsample = n_samples)
            d_val = pd.DataFrame({"context": d["context"], 
                                  "count": val_count,
                                  "weight": n_samples, 
                                  "freq": val_count/n_samples})

            # remove validation counts from training set
            d["count"] = d["count"] - val_count
            d["weight"] = d["weight"] - n_samples
            d['freq'] = (d["count"])/(d["weight"])

            # write output
            write_encoded_data(pd.concat([d.reset_index(drop=True), df_encoding], axis=1), 
                               self.output_dir, 
                               "train", 
                               self.k, 
                               self.ref, 
                               self.val_frac, 
                               self.encoding)
            write_encoded_data(pd.concat([d_val.reset_index(drop=True), df_encoding], axis=1), 
                               self.output_dir, 
                               "val", 
                               self.k, 
                               self.ref, 
                               self.val_frac, 
                               self.encoding)

        else: 
            d['freq'] = (d["count"])/(d["weight"])
            write_encoded_data(pd.concat([d.reset_index(drop=True), df_encoding], axis=1), 
                               self.output_dir, 
                               "test", 
                               self.k, 
                               self.ref, 
                               self.val_frac, 
                               self.encoding)

        
    def combine_data(self): 
        d = pd.merge(self.weights, self.counts, on=["context"], how='left') # join context weights and counts
        d[["count"]] = d[["count"]].fillna(0, inplace = False) # replace missing counts with zeros

        return d

    def get_pseudocount(self, d): 
        m_rate = np.sum(d["count"])/np.sum(d["weight"]) # mu
        n_sample = int(self.pseudocount/m_rate) # a + b

        return n_sample