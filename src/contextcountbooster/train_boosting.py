import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from itertools import product
from collections import OrderedDict
from contextcountbooster.utils import log_loss
from contextcountbooster.utils import nagelkerke_r2


class Booster:
    def __init__(
        self,
        train_data,
        outdir,
        dist_CV,
        aggregate_and_train_only,
        encoding,
        CV_res,
        n_CV_it,
        n_CV_folds,
        max_depth,
        eta,
        subsample,
        colsample_bytree,
        colsample_bylevel,
        l2_lambda,
        tree_method,
        grow_policy,
        alpha,
        distribution,
        min_es_delta,
    ):
        self.train_file_name = train_data
        self.train_data = pd.read_csv(self.train_file_name, sep="\t")
        self.outdir = outdir
        self.dist_CV = dist_CV
        self.aggregate_and_train_only = aggregate_and_train_only
        self.CV_res = CV_res
        self.n_CV_it = n_CV_it
        self.encoding = encoding
        self.k = int((self.train_data.shape[1] - 4) / self.encoding)
        self.n_folds = n_CV_folds

        self.eta = eta
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.l2_lambda = l2_lambda
        self.tree_method = tree_method
        self.grow_policy = grow_policy
        self.alpha = alpha
        self.distribution = distribution
        self.min_es_delta = min_es_delta

        if max_depth == "k_based":
            self.s_md = int(self.k) - 1
            self.max_depth = [x for x in range(self.s_md, self.s_md + 2, 2)]
        else:
            self.max_depth = max_depth

        self.paramspace = {
            "alpha": alpha,
            "max_depth": self.max_depth,  # max depth of a tree
            "eta": eta,  # learning rate
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "lambda": l2_lambda,
            "tree_method": tree_method,  # auto is the default (same as hist); hist: faster histogram optimized approximate greedy algorithm; alternatives: exact (enumerates all split candidates);
            "grow_policy": grow_policy,  # param only supported if tree_method = hist/auto; depthwise is the default (splits at nodes closest to the root); alternative lossguide: split at nodes with highest loss change
        }
        self.fixed_param = {
            "booster": "gbtree",  # "gbtree is the default (non-linear relationship)", alternative: gblinear, which is essentially elastic-net
            "max_bin": 2,  # 256 is the default; only used if tree_method = hist; max. number of discrete bins to bucket continuous features
            "seed": 0,
            "max_delta_step": 0.7,
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
        }

    def prep_modeling_data(self, x, m, w, alpha, add_pc=False):
        """
        1) Checks for observations where weight is zero or mutated count is larger than the weight.
        2) Calculates mean rate, unmutated counts.
        3) Adds pseudocounts to mutated counts, unmutated counts and weights
            x: features (one-hot encoded k-mers)
            m: mutated counts
            w: weights (mutated + unmutated counts)
            data_type: test/train
            add_pc: pseudocount addition flag
        Returns: mutated, unmutated counts, weights, mean rate, xgb.Dmatrix
        """

        # check for (and remove) observations with zero weights
        mask = w > 0  # get indices of non-zero weights
        if sum(mask) != len(w):
            print(
                f"\n--->PROBLEM: There are observations with zero weights ({len(w) - sum(mask)} out of {len(w)} total observations)\n"
            )
            x = x[mask, :]  # choose features with non-zero weights
            m = m[mask]  # choose counts with non-zero weights
            w = w[mask]  # choose weights with non-zero weights

        # check for observations where the mutated count is larger than the weight, if there are any, exit
        mask = m > w

        if sum(mask) > 0:
            print(f"\n---> PROBLEM: m > w, at indices: {mask}\n")

        # calculate unmutated counts based on weights and mutated counts as: weight - mutation cound
        u = np.subtract(w, m)

        # mean rate
        mu_rate = np.sum(m) / np.sum(w)

        # add pseducounts based on alpha
        w_ps_m = w * alpha * mu_rate
        w_ps_u = w * alpha * (1 - mu_rate)
        if add_pc:
            m = np.add(m, w_ps_m)
            u = np.add(u, w_ps_u)
            w = np.add(m, u)

        dat = xgb.DMatrix(
            x, label=np.divide(m, w).astype(np.float64), weight=w.astype(np.float64)
        )
        return x, m, u, w, mu_rate, dat

    def draw_counts(self, n_in_fold, counts, rng):
        """
        --------------- Adjusted from kmerPaPa github ---------------> https://github.com/BesenbacherLab/kmerPaPa/tree/main

        Sample from a hypergeometric distribution to split training data to CV folds
            U: unmutated counts list
            M: mutated counts list
        """

        remaining = np.cumsum(counts[::-1])[::-1]
        fold_counts = np.zeros(len(counts), dtype=np.uint64)
        for i in range(len(counts) - 1):
            if n_in_fold < 1:
                break
            fold_counts[i] = rng.hypergeometric(counts[i], remaining[i + 1], n_in_fold)
            n_in_fold -= fold_counts[i]
        fold_counts[-1] = n_in_fold

        return fold_counts

    def make_folds(self, M, U, n_folds, seed=0):
        """
        --------------- Adjusted from kmerPaPa github ---------------> https://github.com/BesenbacherLab/kmerPaPa/tree/main

        Sample from a hypergeometric distribution to split training data to CV folds
            U: unmutated counts list
            M: mutated counts list
        """
        assert len(M) == len(U)
        m_u_counts = np.empty(2 * len(M), dtype=np.uint64)
        for i in range(len(M)):
            m = M[i]
            u = U[i]
            m_u_counts[i] = m
            m_u_counts[len(M) + i] = u

        n = m_u_counts.sum()
        n_samples = n // n_folds

        rng = np.random.RandomState(seed)
        samples = np.empty((2 * len(M), n_folds), dtype=np.uint64)

        for i in range(n_folds - 1):
            s_count = self.draw_counts(n_samples, m_u_counts, rng)
            samples[:, i] = s_count
            m_u_counts -= s_count
        samples[:, n_folds - 1] = m_u_counts
        return samples

    def param_configs(self):
        """
        Splits a dictionary with lists as values to separate dictionaries with all combinations of lists' values
            paramspace: CV grid
        """
        for comb in product(*self.paramspace.values()):
            yield OrderedDict(
                sorted(zip(self.paramspace.keys(), comb), key=lambda x: x[0])
            )

    def train_booster(self):
        # training features
        x_train = np.array(self.train_data.iloc[:, 4:])  # encoded features
        m_train = np.array(self.train_data["count"])  # kmer weights w = m + u
        w_train = np.array(self.train_data["weight"])  # kmer counts (m)
        mask = w_train > 0  # get indices of non-zero weights
        if sum(mask) != len(w_train):
            x_train = x_train[mask, :]  # choose features with non-zero weights
            m_train = m_train[mask]  # choose counts with non-zero weights
            w_train = w_train[mask]  # choose weights with non-zero weights
        u_train = np.subtract(w_train, m_train)

        # base parameters (not trained for optimal values)
        if self.dist_CV:
            CV_dist_res = self.cross_validation(x_train, m_train, u_train)
            return CV_dist_res
        else:
            if self.aggregate_and_train_only:  # CV run in a distributed manner from workflow, combine CV results and choose optimal params
                # read in dist CV results
                CV_folder = (
                    "/" + "/".join(self.outdir.strip("/").split("/")[:-1]) + "/CV/"
                )
                CV_files = [
                    f
                    for f in os.listdir(CV_folder)
                    if os.path.isfile(os.path.join(CV_folder, f))
                ]
                best_mean_nll = 1e60
                for i, file in enumerate(CV_files):
                    train_res_i = pd.read_csv(CV_folder + file, sep=",")

                    mean_ll = train_res_i.loc[0, "mean_val_ll"].item()
                    if -mean_ll < best_mean_nll:
                        best_mean_nll = -mean_ll
                        param_best = train_res_i

                    if i == 0:
                        train_res = train_res_i
                    else:
                        train_res = pd.concat(
                            [train_res, train_res_i], ignore_index=True
                        )

                opt_param = param_best.to_dict("records")[0]
                print(opt_param)

            else:
                opt_param, train_res = self.cross_validation(x_train, m_train, u_train)

            # prepare training data target and weights
            alpha = float(opt_param.pop("alpha"))
            x_train, m_train, u_train, w_train, f_mu_train, dtrain = (
                self.prep_modeling_data(x_train, m_train, w_train, alpha, add_pc=True)
            )

            # remove optimal parameters CV performance measures
            mean_val_ll = opt_param.pop("mean_val_ll")
            mean_val_nk_r2 = opt_param.pop("mean_val_nk_r2")
            max_best_it = opt_param.pop("best_iteration")
            opt_param.pop("CV_time_min")

            # create model full configuration
            full_config = self.fixed_param | opt_param

            start_time = time.time()  # training time start
            bst = xgb.train(full_config, dtrain, num_boost_round=max_best_it)
            end_time = time.time()  # training time end

            # null model log_lik
            n_train = sum(w_train)
            ll0_train = log_loss(np.repeat(f_mu_train, len(m_train)), m_train, u_train)

            # run prediction
            preds_train = bst.predict(dtrain).astype(np.float64)

            # calculate log_loss
            ll_train = log_loss(preds_train, m_train, u_train)

            # calculate nagelkerke_r2
            nk_r2_train = nagelkerke_r2(n_train, ll0_train, ll_train)

            # save model
            os.makedirs(self.outdir, exist_ok=True)
            bst.save_model(os.path.join(self.outdir, "bst_best.json"))
            # dump model
            bst.dump_model(os.path.join(self.outdir, "bst_best.raw.txt"))

            opt_param["alpha"] = alpha
            opt_param["mean_val_ll"] = mean_val_ll
            opt_param["mean_val_nk_r2"] = mean_val_nk_r2
            opt_param["max_best_it"] = max_best_it
            opt_param["train_ll0"] = ll0_train
            opt_param["train_ll"] = ll_train
            opt_param["train_nk_r2"] = nk_r2_train
            opt_param["train_time_min"] = round((end_time - start_time) / 60, 2)

            # write best model config and loss
            pd.DataFrame.from_dict(data=opt_param, orient="index").to_csv(
                os.path.join(self.outdir, "bst_best_param.csv"), header=False
            )

            # write training results
            train_res.to_csv(
                os.path.join(self.outdir, "training_res.csv"), header=True, index=False
            )

            # write mean freq (null model)
            pd.DataFrame.from_dict(
                data={"mu_freq": [f_mu_train]}, orient="columns"
            ).to_csv(os.path.join(self.outdir, "mu_freq.csv"), index=False)

            return bst

    def cross_validation(self, x_train_in, m_train_in, u_train_in):
        xgb.set_config(verbosity=1)  # verbosity level
        num_round = 10000  # number of boosting rounds
        best_mean_nll = 1e60  # start value for nll
        n_c = x_train_in.shape[0]  # number of input contexts
        train_res = pd.DataFrame({})  # results DF

        for config in self.param_configs():  # Find optimal hyperparams
            alpha = float(config.pop("alpha"))

            if (
                config["tree_method"] == "exact"
            ):  # grow policy not tuned for exact tree method
                config.pop("grow_policy")
                if (
                    (
                        train_res.loc[
                            train_res["tree_method"] == "exact",
                            [
                                "colsample_bylevel",
                                "colsample_bytree",
                                "eta",
                                "lambda",
                                "max_depth",
                                "subsample",
                            ],
                        ]
                        .astype("float")
                        .values
                        == [float(x) for x in config.values() if x != "exact"]
                    )
                    .all(axis=1)
                    .any()
                ):
                    continue

            ll_CV = []  # save validation set log-likelhood
            nk_r2_CV = []  # save validation set nagelkerke r2
            best_it = []  # save iteration where training was stopped (early stopping evaluated based on validation loglik)

            start_time = time.time()  # CV time start
            for n_CV_it_i in range(self.n_CV_it):
                # create CV folds
                CV_folds = self.make_folds(
                    m_train_in, u_train_in, self.n_folds, seed=n_CV_it_i
                )

                for fold in range(self.n_folds):
                    # training folds
                    m_train_f = CV_folds[
                        0:n_c, [x for x in range(self.n_folds) if x != fold]
                    ].sum(axis=1)  # sum mutated counts across training folds
                    u_train_f = CV_folds[
                        n_c : (2 * n_c), [x for x in range(self.n_folds) if x != fold]
                    ].sum(axis=1)  # sum unmutated counts across training folds
                    w_train_f = np.sum(
                        [m_train_f, u_train_f], axis=0
                    )  # calculate weight as m + u

                    x_train, m_train_f, u_train_f, w_train_f, f_mu_train, dtrain = (
                        self.prep_modeling_data(
                            x_train_in, m_train_f, w_train_f, alpha, add_pc=True
                        )
                    )

                    # validation fold
                    m_val_f = CV_folds[
                        0:n_c, fold
                    ]  # get validation fold mutated counts
                    u_val_f = CV_folds[
                        n_c : (2 * n_c), fold
                    ]  # get validation fold unmutated counts
                    w_val_f = np.sum([m_val_f, u_val_f], axis=0)
                    x_val, m_val_f, u_val_f, w_val_f, f_mu_val, dval = (
                        self.prep_modeling_data(
                            x_train_in, m_val_f, w_val_f, alpha, add_pc=False
                        )
                    )

                    # null model log_lik
                    ll0_val = log_loss(
                        np.repeat(f_mu_train, len(m_val_f)), m_val_f, u_val_f
                    )

                    # specify validations set to watch performance
                    watchlist = [(dtrain, "train"), (dval, "eval")]

                    print(f"Running CV iteration {n_CV_it_i} and fold {fold}")
                    full_config = (
                        self.fixed_param | config
                    )  # merge fixed and tuned parameters

                    # early stopping callback
                    es = xgb.callback.EarlyStopping(
                        rounds=100,  # val error needs to decrease at least every x rounds to continue training
                        min_delta=self.min_es_delta,  # minimum change in metric
                        save_best=True,
                        maximize=False,
                        data_name="eval",
                        metric_name="poisson-nloglik",  # logloss
                    )

                    # training
                    evals_result = {}
                    bst = xgb.train(
                        full_config,
                        dtrain,
                        num_boost_round=num_round,
                        evals=watchlist,
                        callbacks=[es],
                        evals_result=evals_result,  # save evaluations to dict
                        verbose_eval=100,
                    )
                    best_it.append(bst.best_iteration)

                    # calculate log_loss
                    ll = log_loss(
                        bst.predict(dval).astype(np.float64), m_val_f, u_val_f
                    )
                    ll_CV.append(ll)

                    # calculate nagelkerke_r2
                    nk_r2 = nagelkerke_r2(sum(w_val_f), ll0_val, ll)
                    nk_r2_CV.append(nk_r2)
                    print(
                        f"XGBoost with config: {config} val_LL={round(ll, 3)} val_nk_r2={round(nk_r2, 5)}"
                    )

            end_time = time.time()  # CV time end

            if not self.dist_CV:
                # calculate means across folds
                mean_ll = (sum(ll_CV) / len(ll_CV)).item()
                mean_val_nk_r2 = (sum(nk_r2_CV) / len(nk_r2_CV)).item()
                mean_best_it = round((sum(best_it) / len(best_it)))
                max_best_it = max(best_it)

                if self.CV_res:
                    nk_r2_CV_no_inf = []
                    for x in nk_r2_CV:
                        if x == float("-inf"):
                            x = 0
                        nk_r2_CV_no_inf.append(x)

                    CV_res_list = np.quantile(
                        nk_r2_CV_no_inf, q=[0, 0.25, 0.5, 0.75, 1]
                    )
                    CV_res_df = pd.DataFrame(
                        columns=["min", "q25", "median", "q75", "max"]
                    )
                    CV_res_df.loc[0] = CV_res_list

                    l2_i = config["lambda"]
                    eta_i = config["eta"]
                    md_i = config["max_depth"]
                    alpha_i = alpha
                    CV_res_df.to_csv(
                        os.path.join(
                            self.outdir,
                            f"CV_res_l2_{l2_i}_eta_{eta_i}_md_{md_i}_alpha_{alpha_i}.csv",
                        ),
                        header=True,
                        index=False,
                    )

                if config["tree_method"] == "auto":
                    print(
                        f"XGBoost with alpha={alpha} max_depth={config['max_depth']} eta={config['eta']} tree_method={config['tree_method']} grow_policy={config['grow_policy']} mean_val_LL={round(mean_ll, 5)} mean_val_nk_r2={round(mean_val_nk_r2, 5)} mean_best_iteration: {mean_best_it}"
                    )
                else:
                    print(
                        f"XGBoost with alpha={alpha} max_depth={config['max_depth']} eta={config['eta']} tree_method={config['tree_method']} mean_val_LL={round(mean_ll, 5)} mean_val_nk_r2={round(mean_val_nk_r2, 5)} mean_best_iteration: {mean_best_it}"
                    )

                config["mean_val_ll"] = mean_ll
                config["mean_val_nk_r2"] = mean_val_nk_r2
                config["best_iteration"] = max_best_it
                config["CV_time_min"] = round((end_time - start_time) / 60, 2)
                config["alpha"] = alpha

                if train_res.shape[0] == 0:
                    train_res = pd.DataFrame([config])
                else:
                    train_res = pd.concat(
                        [train_res, pd.DataFrame([config])], ignore_index=True
                    )

                # if mean neg_log_loss is below the current best, save new best
                if -mean_ll < best_mean_nll:
                    best_mean_nll = -mean_ll
                    param_best = config
                    param_best["best_iteration"] = max_best_it

        if self.dist_CV:
            # calculate means across folds
            mean_ll = (sum(ll_CV) / len(ll_CV)).item()
            mean_val_nk_r2 = (sum(nk_r2_CV) / len(nk_r2_CV)).item()
            mean_best_it = round((sum(best_it) / len(best_it)))
            max_best_it = max(best_it)

            config["mean_val_ll"] = mean_ll
            config["mean_val_nk_r2"] = mean_val_nk_r2
            config["best_iteration"] = max_best_it
            config["CV_time_min"] = round((end_time - start_time) / 60, 2)
            config["alpha"] = alpha

            l2_i = config["lambda"]
            eta_i = config["eta"]
            md_i = config["max_depth"]
            dist_CV_res = pd.DataFrame(config, index=[0])
            if alpha == 0:
                alpha_write = "0"
            else:
                alpha_write = alpha
            dist_CV_res.to_csv(
                os.path.join(
                    self.outdir,
                    f"L2_{l2_i}__ETA_{eta_i}__MD_{md_i}__ALPHA_{alpha_write}.csv",
                ),
                index=False,
            )
            return 0
        else:
            return param_best, train_res

    def get_base_encoded_rep(self):
        up_to = 1
        if int(self.k) % 2 == 0:
            up_to = 0

        base_rep = [
            str(x)
            for x in range(-(self.k // 2), up_to)
            for _ in range(int(self.encoding))
        ]
        base_rep.extend(
            [
                str(x)
                for x in range(1, (self.k // 2) + 1)
                for _ in range(int(self.encoding))
            ]
        )

        f_rep = ["f" + str(x) for x in range(0, (self.k * int(self.encoding)))]
        f_name = [
            str(x) + "_b" + str(y)
            for x in range(-(self.k // 2), up_to)
            for y in range(1, int(self.encoding) + 1)
        ]
        f_name.extend(
            [
                str(x) + "_b" + str(y)
                for x in range(1, (self.k // 2) + 1)
                for y in range(1, int(self.encoding) + 1)
            ]
        )
        print("base_rep, length", len(base_rep))
        print("f_rep, length", len(f_rep))
        print("f_name, length", len(f_name))

        base_rep_df = pd.DataFrame(
            {"encoded": f_rep, "base_rep": base_rep, "feature_name": f_name}
        )
        base_rep_df = base_rep_df.set_index("encoded")
        return base_rep_df

    def format_feature_data(self, feature_imp):
        base_rep_df = self.get_base_encoded_rep()
        keys = list(feature_imp.keys())
        values = list(feature_imp.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"])
        data["sort"] = data.index.str.extract(r"(\d+)", expand=False).astype(int)
        data = data.sort_values(by="sort", ascending=True)
        data = pd.merge(
            base_rep_df, data, left_index=True, right_index=True, how="left"
        )
        data[["score"]] = data[["score"]].fillna(0, inplace=False)
        return data

    def plot_feature_data(self, data, ylab, filename):
        sns.set_theme(rc={"figure.figsize": (14, 4)})
        sns.color_palette("tab10")
        plot = sns.barplot(data=data, x="feature_name", y="score", hue="base_rep")
        for index, row in data.iterrows():
            plot.text(
                row.feature_name, row.score, int(row.score), fontsize=8, ha="center"
            )
        sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), title=None)
        plot.set(xlabel="Feature", ylabel=ylab)
        plot.tick_params(axis="x", rotation=90)
        plot.figure.savefig(os.path.join(self.outdir, filename), bbox_inches="tight")
        plot.figure.clf()

    def plot_feature_gain(self, bst):
        feature_dat = bst.get_score(
            importance_type="gain"
        )  # gain: the average gain across all splits the feature is used in
        data = self.format_feature_data(feature_dat)
        self.plot_feature_data(data, "Gain", "feature_gain.png")

    def plot_feature_weight(self, bst):
        feature_dat = bst.get_score(
            importance_type="weight"
        )  # weight: number of times a feature is used to split the data across all trees
        data = self.format_feature_data(feature_dat)
        self.plot_feature_data(data, "Weight", "feature_weight.png")
