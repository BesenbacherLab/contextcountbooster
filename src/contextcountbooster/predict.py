import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import xlogy



class Predicter:
    def __init__(self,
                 test_data, 
                 model,
                 outdir):
        
        self.test_data = pd.read_csv(test_data, sep = "\t")
        self.bst = xgb.Booster({'nthread': 4})  # init model
        self.bst.load_model(model)  # load model data (json)
        self.outdir = outdir

    def predict(self):

        x_test =  np.array(self.test_data.iloc[:, 4:]) # encoded features
        m_test = self.test_data["count"].to_list()
        w_test = self.test_data["weight"].to_list()
        u_test = [y-x for x,y in zip(m_test, w_test)]

        dtest = xgb.DMatrix(x_test, weight = w_test)
        preds = self.bst.predict(dtest) # , iteration_range=(0, self.bst.best_iteration + 1) -> not needed, uses best iteration automatically?

        nll_test = self.log_loss(preds, m_test, u_test)
        print(f"Test data nll: {nll_test}")
        res_test = {"nll": nll_test}
        os.makedirs(self.outdir, exist_ok=True) 
        pd.DataFrame.from_dict(data=res_test, orient='index').to_csv(os.path.join(self.outdir, "test_nll.csv"), header=False)

        res_pred = self.test_data[["context", "count", "weight", "freq"]]
        res_pred = res_pred.assign(pred_freq = preds)
        res_pred.to_csv(os.path.join(self.outdir, "test_pred.csv"), header=True, index = False)



    def log_loss(self, p_preds, m, u):
        ll = 0
        for idx, p in enumerate(p_preds):
            ll += xlogy(m[idx], p) + xlogy(u[idx], (1-p)) # p = predicted rate, m = mut count, u = unmut count
        return -ll