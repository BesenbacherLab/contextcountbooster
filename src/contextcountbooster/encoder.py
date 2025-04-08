import pandas as pd
from contextcountbooster.utils import read_context_data
from contextcountbooster.utils import write_encoded_data


encoding_7bit = {
    "A": "1000111",  # encodes iupac: ACGTMRW; where M = AC; R = AG; W = AT
    "C": "0100100",
    "G": "0010010",
    "T": "0001001",
}

encoding_4bit = {
    "A": "1000",  # encodes iupac: ACGT
    "C": "0100",
    "G": "0010",
    "T": "0001",
}


class OneHotEncoder:
    def __init__(
        self,
        counts,
        weights,
        output_dir=None,
        output_prefix="",
        encoding=7,
        ref=None,
    ):
        # read in count data
        self.counts, k1 = read_context_data(counts, ref, dtype="count")
        self.weights, k2 = read_context_data(weights, ref, dtype="weight")
        assert k1 == k2
        self.k = k1
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.encoding = encoding
        self.ref = ref

    def encode(self):
        # combine counts and weights
        d = self.combine_data()

        # one hot encode
        if self.encoding == 7:
            encoder = encoding_7bit
        else:
            encoder = encoding_4bit
        encoding = [
            [x for k in context for x in encoder[k]] for context in d.context.to_list()
        ]
        df_encoding = pd.DataFrame(
            encoding,
            columns=[
                "p" + str(x) + "_b" + str(y)
                for x in range(1, self.k + 1, 1)
                for y in range(1, self.encoding + 1, 1)
            ],
        )

        d["freq"] = (d["count"]) / (d["weight"])
        write_encoded_data(
            pd.concat([d.reset_index(drop=True), df_encoding], axis=1),
            self.output_dir,
            self.output_prefix,
            self.k,
            self.encoding,
        )

    def combine_data(self):
        d = pd.merge(
            self.weights, self.counts, on=["context"], how="left"
        )  # join context weights and counts
        d[["count"]] = d[["count"]].fillna(
            0, inplace=False
        )  # replace missing counts with zeros

        return d
