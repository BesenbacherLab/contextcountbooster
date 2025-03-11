import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from itertools import product
from scipy.special import xlogy


class Booster:
    def __init__(self,
                 train_data, 
                 val_data, 
                 outdir,
                 encoding,
                 max_depth,
                 eta,
                 tree_method,
                 grow_policy,
                 ):

        self.train_data = pd.read_csv(train_data, sep = "\t")
        self.val_data = pd.read_csv(val_data, sep = "\t")
        self.outdir = outdir
        self.encoding = encoding
        self.k = int((self.train_data.shape[1] - 4)/self.encoding)
        self.max_depth = max_depth
        self.eta = eta
        self.tree_method = tree_method
        self.grow_policy = grow_policy
        self.paramspace = {"max_depth": max_depth, # max depth of a tree
                           "eta": eta, # learning rate
                           "tree_method": tree_method, # auto is the default (same as hist); hist: faster histogram optimized approximate greedy algorithm; alternatives: exact (enumerates all split candidates); 
                           "grow_policy": grow_policy, # param only supported if tree_method = hist/auto; depthwise is the default (splits at nodes closest to the root); alternative lossguide: split at nodes with highest loss change
                           } 


    def train_booster(self):

        # prepare training and validation data
        x_train =  np.array(self.train_data.iloc[:, 4:]) # encoded features
        y_train = self.train_data["freq"].to_list()
        w_train = self.train_data["weight"].to_list()
        m_train = self.train_data["count"].to_list()
        u_train = [y-x for x,y in zip(m_train, w_train)]

        x_val =  np.array(self.val_data.iloc[:, 4:])
        y_val = self.val_data["freq"].to_list()
        w_val = self.val_data["weight"].to_list()
        m_val = self.val_data["count"].to_list()
        u_val = [y-x for x,y in zip(m_val, w_val)]

        xgb.set_config(verbosity=1) # set verbosity level

        dtrain = xgb.DMatrix(x_train, label = y_train, weight = w_train)
        dval = xgb.DMatrix(x_val, label = y_val, weight = w_val)

        # alternative formulation: modeling counts (y_train = data["count"]), setting weight as base margin
        #dtrain = xgb.DMatrix(x_train, y_train, base_margin = [math.log(x) for x in w_train]) 
        #dval = xgb.DMatrix(x_val, y_val, base_margin = [math.log(x) for x in w_val])

        # base parameters (not trained for optimal values)
        param = {"booster": "gbtree", # "gbtree is the default (non-linear relationship)", alternative: gblinear, which is essentially elastic-net
                "max_bin": 2, # 256 is the default; only used if tree_method = hist; max. number of discrete bins to bucket continuous features
                "seed": 0, 
                "max_delta_step": 0.7, # 0.7 is the default for obj=count:poisson; otherwise default=0; max. delta step we allow each leaf output to be. 
                "objective": "count:poisson", 
                "eval_metric": ["poisson-nloglik"]}
        
        # number of boosting rounds 
        num_round = 5000

        # specify validations set to watch performance
        watchlist = [(dtrain, "train"), (dval, "eval")]

        best_nll = 1e16
        train_res = pd.DataFrame({})
        # Find optimal hyperparams
        for config in self.param_configs():
            
            if config["tree_method"] == "exact":
                config.pop("grow_policy")
                # TODO: check if exact config already run, if yes, skip

            print(f"Training with config: {config}")
            full_config = param | config # merge base and tuned parameters

            # early stopping callback
            es = xgb.callback.EarlyStopping(
                rounds=3, # val error needs to decrease at least every x rounds to continue training
                min_delta=1e-10, # minimum change in metric
                save_best=True,
                maximize=False,
                data_name="eval",
                metric_name="poisson-nloglik", # logloss
            )

            evals_result = {}
            bst = xgb.train(full_config, 
                            dtrain, 
                            num_boost_round=num_round, 
                            evals=watchlist, 
                            callbacks=[es], 
                            evals_result = evals_result, # save evaluations to dict
                            verbose_eval = 50) 

            # run prediction
            preds_train = bst.predict(dtrain)
            preds = bst.predict(dval) # iteration_range=(0, bst.best_iteration + 1) -> not needed, uses best iteration automatically?


            # calculate log_loss
            nll_train = self.log_loss(preds_train, m_train, u_train)
            nll = self.log_loss(preds, m_val, u_val)
            if config["tree_method"] == "auto":
                print(f'XGBoost with max_depth={config["max_depth"]} eta={config["eta"]} tree_method={config["tree_method"]} grow_policy={config["grow_policy"]} val_LL={nll}')
            else: 
                print(f'XGBoost with max_depth={config["max_depth"]} eta={config["eta"]} tree_method={config["tree_method"]} val_LL={nll}')
            
            # TODO: calculate nagelkerke_r2

            # if neg_log_loss is below the current best, save new best
            if nll < best_nll:
                bst_best = bst
                param_best = config
                param_best["train_nll"] = nll_train
                param_best["val_nll"] = nll
                
            config["train_nll"] = nll_train
            config["val_nll"] = nll

            if train_res.shape[0] == 0:
                train_res = pd.DataFrame([config])
            else: 
                train_res = pd.concat([train_res, pd.DataFrame([config])], ignore_index=True)

        # save model
        os.makedirs(self.outdir, exist_ok=True) 
        bst_best.save_model(os.path.join(self.outdir, "bst_best.json"))
        # dump model
        bst_best.dump_model(os.path.join(self.outdir, "bst_best.raw.txt"))

        # write best model config and loss
        pd.DataFrame.from_dict(data=param_best, orient='index').to_csv(os.path.join(self.outdir, "bst_best_param.csv"), header=False)
        
        # write training results
        train_res.to_csv(os.path.join(self.outdir, "training_res.csv"), header=True, index = False)
        
        return bst


    def param_configs(self):
        for comb in product(*self.paramspace.values()):
            yield dict(zip(self.paramspace.keys(), comb))

    
    def log_loss(self, p_preds, m, u):
        ll = 0
        for idx, p in enumerate(p_preds):
            ll += xlogy(m[idx], p) + xlogy(u[idx], (1-p)) # p = predicted rate, m = mut count, u = unmut count
        return -ll
    

    # TODO: implement nagelkerke_r2
    def nagelkerke_r2(self):
        nk_r2 = 2
        return nk_r2
    

    def get_base_encoded_rep(self):
        base_rep = [str(x) for x in range(-(self.k//2), 1) for _ in range(int(self.encoding))]
        base_rep.extend([str(x) for x in range(1, (self.k//2)+1) for _ in range(int(self.encoding))])

        f_rep = ["f" + str(x) for x in range(0, (self.k*int(self.encoding)))]
        f_name = [str(x) + "_b" + str(y) for x in range(-(self.k//2), 1)  for y in range(1, int(self.encoding)+1)]
        f_name.extend([str(x) + "_b" + str(y) for x in range(1, (self.k//2)+1)  for y in range(1, int(self.encoding)+1)])
        base_rep_df = pd.DataFrame({'encoded': f_rep, "base_rep": base_rep, "feature_name": f_name})
        base_rep_df = base_rep_df.set_index('encoded')
        return base_rep_df


    def format_feature_data(self, feature_imp):
        base_rep_df = self.get_base_encoded_rep()
        keys = list(feature_imp.keys())
        values = list(feature_imp.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"])
        data['sort'] = data.index.str.extract(r'(\d+)', expand=False).astype(int)
        data = data.sort_values(by = "sort", ascending=True)
        data = pd.merge(base_rep_df, data, left_index = True, right_index = True, how='left') 
        data[["score"]] = data[["score"]].fillna(0, inplace = False)
        return data


    def plot_feature_data(self, data, ylab, filename):
        sns.set_theme(rc={'figure.figsize':(14,4)})
        sns.color_palette("tab10")
        ax = sns.barplot(data=data, x="feature_name", y='score', hue='base_rep')
        for index, row in data.iterrows():
            ax.text(row.feature_name, row.score, int(row.score), fontsize = 8, ha='center')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.ylabel(ylab)
        plt.xlabel("Feature")
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.outdir, filename), bbox_inches = 'tight')
        plt.show()


    def plot_feature_gain(self, bst):

        feature_dat = bst.get_score(importance_type='gain') # gain: the average gain across all splits the feature is used in
        data = self.format_feature_data(feature_dat) 
        print(data.head())
        self.plot_feature_data(data, "Gain", "feature_gain.png")


    def plot_feature_weight(self, bst):

        feature_dat = bst.get_score(importance_type='weight') # weight: number of times a feature is used to split the data across all trees
        data = self.format_feature_data(feature_dat)
        print(data.head())
        self.plot_feature_data(data, "Weight", "feature_weight.png")