import os
import numpy as np
import pandas as pd
import xgboost as xgb
from contextcountbooster.utils import log_loss
from contextcountbooster.utils import nagelkerke_r2

from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions import Poisson, ZIPoisson
from xgboostlss.distributions.Poisson import *  # noqa: F403
from xgboostlss.distributions.ZIPoisson import *  # noqa: F403


class Predicter:
    def __init__(self, test_data, model, null_model, outdir, distribution):
        self.test_data = pd.read_csv(test_data, sep="\t")
        self.mod0 = pd.read_csv(null_model)
        self.mod0 = self.mod0["mu_freq"].item()
        self.outdir = outdir
        self.distribution = distribution

        # # init xgboostLSS model
        if self.distribution == "Poisson":
            self.xgblss = XGBoostLSS(
                Poisson(stabilization="None", response_fn="softplus", loss_fn="nll")
            )
        elif self.distribution == "ZIPoisson":
            self.xgblss = XGBoostLSS(
                ZIPoisson(stabilization="None", response_fn="softplus", loss_fn="nll")
            )
        self.xgblss.load_model(model)  # load model data (pkl)

    def predict(self):
        x_test = np.array(self.test_data.iloc[:, 4:])  # encoded features
        m_test = self.test_data["count"].to_list()
        w_test = self.test_data["weight"].to_list()
        u_test = [y - x for x, y in zip(m_test, w_test)]

        dtest = xgb.DMatrix(x_test, weight=w_test)

        if self.distribution == "Poisson":
            preds = self.xgblss.predict(dtest, pred_type="parameters")["rate"].to_list()
        elif self.distribution == "ZIPoisson":
            r_val = self.xgblss.predict(dtest, pred_type="parameters")["rate"].to_list()
            p_val = self.xgblss.predict(dtest, pred_type="parameters")["gate"].to_list()
            preds = [(1 - p) * r for r, p in zip(r_val, p_val)]

        ll_test = log_loss(preds, m_test, u_test)

        # calculate nagelkerke r2
        n_test = sum(w_test)
        ll0_test = log_loss([self.mod0] * self.test_data.shape[0], m_test, u_test)
        nk_r2 = nagelkerke_r2(n_test, ll0_test, ll_test)

        print(f"Test data ll: {ll_test}")
        print(f"Test data nk_r2: {nk_r2}")
        print(f"N: {n_test}")

        res_test = {
            "ll": ll_test,
            "ll0": ll0_test,
            "ll_diff": ll_test - ll0_test,
            "nagelkerke_r2": nk_r2,
            "N": n_test,
        }
        os.makedirs(self.outdir, exist_ok=True)
        pd.DataFrame.from_dict(data=res_test, orient="index").to_csv(
            os.path.join(self.outdir, "test_nll.csv"), header=False
        )

        res_pred = self.test_data[["context", "count", "weight", "freq"]]
        res_pred = res_pred.assign(pred_freq=preds)
        res_pred.to_csv(
            os.path.join(self.outdir, "test_pred.csv"), header=True, index=False
        )
