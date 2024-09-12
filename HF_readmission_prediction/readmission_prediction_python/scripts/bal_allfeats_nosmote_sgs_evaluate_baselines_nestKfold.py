import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from datetime import datetime
import json
import pandas as pd
from pymongo import MongoClient
import sklearn.ensemble as sken
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skms
import sklearn.neighbors as skknn
import sklearn.neural_network as sknn
import sklearn.metrics as skm
import sklearn.tree as sktree
import tqdm
import urllib.parse
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump, load

# For xgboost balanced weights
from sklearn.utils.class_weight import compute_sample_weight

# from focal_loss import BinaryFocalLoss
from imblearn.over_sampling import SMOTE

# from sklearn.inspection import permutation_importance # For feature importance
import shap
import imblearn

from evalHelper import (
    read_json,
    evaluate_results,
    get_train_test,
    evaluate_results_fairness,
)


MODEL_PARAMS = {
    "xgb": {
        "model": xgb.XGBClassifier(),
        "params": {
            "max_depth": [6, 7, 8],
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.1],
            "eval_metric": ["logloss"],
            "lambda": [0, 0.5, 1],
            "alpha": [0, 0.1, 0.2],
        },
    }
}
"""
    "logr_saga": 
    {
        "model": LogisticRegression(),
        "params": {"penalty": ['l2'], 
                   "max_iter": [10000], 
                   "solver": ['saga'],
                   "C" : [1e-5 , 1e-4 , 1e-3 , 1e-2 , 0.1 , 1 , 5],
                   "class_weight": ["balanced"]
                }
    }
    "model": xgb.XGBClassifier(),
        "params": {"max_depth": [6,7,8],
                   "n_estimators": [10, 20, 30, 40],
                   "learning_rate":[0.01, 0.1],
                   "eval_metric":["logloss"],

                   "lambda": [0, 0.5, 1],
                   "alpha": [0, 0.1, 0.2]
                   }
                },
     "logr_saga": 
    {
        "model": LogisticRegression(),
        "params": {"penalty": [None], 
                   "max_iter": [10000], 
                   "solver": ['saga']
                    }
                },
     "rf": {
         'model': sken.RandomForestClassifier(),
         'params': {'max_depth': [6,7,8],
                    'min_samples_leaf': [5, 10],
                    'n_estimators': [5, 10, 25]}


    ,
    "xgb": 
    {
        "model": xgb.XGBClassifier(),
        "params": {"max_depth": [6,7,8],
                   "n_estimators": [50, 100, 200, 500],
                   "learning_rate":[0.01, 0.1],
                   "eval_metric":["logloss"],
                   "lambda": [0, 0.5, 1],
                   "alpha": [0, 0.1, 0.2]
                }
    },
    "rf":
    {
        'model': sken.RandomForestClassifier(),
         'params': {'max_depth': [6,7,8],
                    'min_samples_leaf': [5, 10],
                    'n_estimators': [5, 10, 25],
                   'class_weight': ["balanced"]
                }

    }

"""


"""
Returns a Python list of all values from a json object, discarding the keys.
"""


def json_extract_values(obj):
    if isinstance(obj, dict):
        values = []
        for key, value in obj.items():
            values.extend(json_extract_values(value))
        return values
    elif isinstance(obj, list):
        return obj
    else:
        return []


def main():
    parser = argparse.ArgumentParser()
    # mongo information
    username = urllib.parse.quote_plus("TODO: input Mongo username")
    password = urllib.parse.quote_plus("TODO: input Mongo password")
    parser.add_argument("-mongo_url", default="TODO: your mongo server url")

    parser.add_argument("-mongo_db", default="TODO: your collection name")
    parser.add_argument(
        "-mongo_col", default="TODO: your mongo collection name", help="collection_type"
    )  # Used to be subgroups_baseline
    parser.add_argument(
        "-data_file",
        default="../data/TODO: your input csv file with merged SDOH data.",
        help="data file",
    )
    parser.add_argument(
        "-base_feat", default="../data/feat_base.json", help="base_features"
    )

    parser.add_argument(
        "-feat_file", default="../data/feat_column.json", help="model_features"
    )

    parser.add_argument(
        "-subgroup_file",
        default="../data/subgroup_fast_cols.json",
        help="subgroups_to_test",
    )
    parser.add_argument("-endpoint", default="readmit30bin")  # readmit30bin

    parser.add_argument(
        "--feats",
        nargs="+",
        default=[
            "our_baseline_clin_AND_demo",
            "ahrq1_tract",
            "ahrq3_tract",
            "ahrq4_tract",
            "resolved_nanda1_tract",
            "resolved_nanda3_tract",
            "resolved_nanda4_tract",
            "1_clin_AHRQ_NaNDA",
            "3_clin_AHRQ_NaNDA",
            "4_clin_AHRQ_NaNDA",
            "1_ahrq_nanda_resolved",
            "3_ahrq_nanda_resolved",
            "4_ahrq_nanda_resolved",
            "NANDA_TOTAL_tract_resolved",
            "AHRQ_TOTAL_tract_no25",
            "AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract_resolved",
            "blad_AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract_resolved",
            # Additional feature sets we explored can be added using json from the /data/ directory.
        ],
    )

    args = parser.parse_args()

    # setup mongo
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    fairness_mcol = mdb["TODO: your column name for fairness metrics."]
    logcoeffs_mcol = mdb["TODO: your column name for LR coefficients."]
    shap_mcol = mdb["TODO: your column name for SHAP values."]

    df = pd.read_csv(args.data_file)
    base_feat = read_json(args.base_feat)
    feat_info = read_json(args.feat_file)
    subgroups_bins = read_json(args.subgroup_file)

    # determine the feature sets
    feat_cols = {}
    for ft in args.feats:
        colset = set()
        # check if it's a base feature, if so update
        if ft in base_feat:
            colset.update(base_feat[ft])
        else:
            for ftbase in feat_info[ft]:
                colset.update(base_feat[ftbase])
        feat_cols[ft] = list(colset)

    # Determine subgroups within each subgroup bin:
    # subgroups = json_extract_values(subgroups_bins)

    # Use "for s in subgroup_keys" as outer loop to subgroups.
    # then, access exact subgroup name: "for g in subgroup_bins[s]"
    subgroup_keys = subgroups_bins.keys()

    for i in tqdm.tqdm(range(1, 11), desc="test-split"):
        train_df, test_df, train_y, test_y = get_train_test(df, i, label=args.endpoint)

        test_y = test_y.astype(int)  # Cast to ensure labels are integers

        # Reset test_df indices for subgroup indexing
        test_df = test_df.reset_index()

        for fname, fcolumns in tqdm.tqdm(feat_cols.items(), desc="feats", leave=False):
            base_res = {
                "file": args.data_file,
                "feat": fname,
                "endpoint": args.endpoint,
                "fold": i,
            }

            # for both train and test get only those columns
            train_x = train_df[fcolumns]

            # Apply imputation:
            imputer = SimpleImputer(
                missing_values=np.nan, strategy="median", keep_empty_features=True
            )
            train_x = imputer.fit_transform(train_x)

            # smt = SMOTE(random_state=42)
            # train_x, smt_train_y = smt.fit_resample(train_x, train_y)

            # Apply feature preprocessing: StandardScaler
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)

            # logr = LogisticRegression(class_weight="balanced" , penalty="none", max_iter=10000, solver='saga')

            for mname, mk_dict in tqdm.tqdm(
                MODEL_PARAMS.items(), desc="models", leave=False
            ):

                gs = skms.GridSearchCV(
                    mk_dict["model"],
                    mk_dict["params"],
                    cv=5,
                    n_jobs=4,
                    scoring="roc_auc",
                    refit=True,
                )

                if mname != "xgb":
                    gs.fit(train_x, train_y)
                else:
                    gs.fit(
                        train_x,
                        train_y,
                        sample_weight=compute_sample_weight("balanced", train_y),
                    )

                # best_val_rocauc = gs.best_score_ # Float.
                # best_val_params = gs.best_params_ # This is a DICT.

                # Save model file for later use:

                model = gs.best_estimator_
                bestmodel_train_params = gs.best_params_

                # Get current date and time
                current_time = datetime.now()
                # Format the date and time into a string
                unique_string = current_time.strftime("%Y%m%d%H%M%S%f")
                # Base filename
                base_filename = "./model_files/model"
                # Create a unique filename by appending the unique string
                bestmodel_unique_filename = f"{base_filename}_{unique_string}.joblib"

                # Loop through test eval for each subgroup
                for s in subgroup_keys:
                    # 1 entire dict is for 1 SET of subgroups
                    # (e.g., 1 dict for "race" subgroups, since "s" is "race" one iteration.)
                    subgroup_preds_dict = {}  # curr_subgroup: [test_x, sg_test_y]
                    # Loop through test eval for each subgroup

                    for g in subgroups_bins[s]:
                        # Get indices of rows that match curr_subgroup
                        cs_ind = test_df.loc[test_df[g] == 1].index
                        # For subgroups, select only current 'sg' from test_df & test_y.
                        sg_test_df = test_df[test_df[g] == 1]
                        sg_test_y = test_y.iloc[
                            cs_ind
                        ]  # use column indices for test_y, bc no subgroup cols here

                        print("______________________________________")
                        print("FOLD: ", i, "MODEL: ", mname)
                        print("______________________________________")

                        # Get only desired feature cols
                        test_x = sg_test_df[fcolumns]
                        # get the test encounter id
                        test_idx = sg_test_df["Encounter"]
                        (
                            auc,
                            aps,
                            y_hat,
                            binary_predictions,
                            precision,
                            recall,
                            f1,
                            auc_precision_recall,
                            fnr,
                            tnr,
                            fpr,
                            mcc,
                        ) = evaluate_results(
                            gs,
                            test_x,
                            sg_test_y,
                            args.endpoint,
                            imputer=imputer,
                            scaler=scaler,
                        )

                        # Track subgroups preds for fairness:
                        subgroup_preds_dict[g] = [test_x, sg_test_y, binary_predictions]

                        perf_res = {
                            "model": mname,
                            "ts": datetime.now(),
                            "auc": auc,
                            "aps": aps,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "auprc": auc_precision_recall,
                            "fnr": fnr,
                            "tnr": tnr,
                            "fpr": fpr,
                            "mcc": mcc,
                            "sg_key": s,
                            "subgroup": g,
                            "test_samp_size": len(sg_test_y),
                            "bestmodel_train_params_dict": bestmodel_train_params,
                            "bestmodel_unique_filename": bestmodel_unique_filename,
                        }
                        mcol.insert_one({**base_res, **perf_res})

                        # Get SHAP for XGBoost
                        if mname == "xgb":
                            # model = gs.best_estimator_ # Best XGBoost model.

                            # Apply same pipeline preprocessing to test_x:
                            xgb_test_x = imputer.transform(test_x)
                            xgb_test_x = scaler.transform(test_x)

                            explainer = shap.Explainer(model)
                            shap_values = explainer(np.ascontiguousarray(xgb_test_x))
                            shap_importance = shap_values.abs.mean(0).values
                            sorted_idx = shap_importance.argsort()
                            ordered_shaps = shap_importance[sorted_idx]
                            names_ordered_shaps = np.array(fcolumns)[sorted_idx]
                            # Save Ordered shaps & names.
                            xg_shap_res = {
                                "model": mname,
                                "ts": datetime.now(),
                                "auc": auc,
                                "aps": aps,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "auprc": auc_precision_recall,
                                "fnr": fnr,
                                "tnr": tnr,
                                "fpr": fpr,
                                "mcc": mcc,
                                "sg_key": s,
                                "subgroup": g,
                                "test_samp_size": len(sg_test_y),
                                "shap_ordered_names": names_ordered_shaps.tolist(),
                                "shap_ordered_importance": ordered_shaps.tolist(),
                                "bestmodel_train_params_dict": bestmodel_train_params,
                                "bestmodel_unique_filename": bestmodel_unique_filename,
                            }
                            shap_mcol.insert_one({**base_res, **xg_shap_res})

                        # Get logr coeff for LogisticRegression
                        if mname == "logr_lbfgs" or mname == "logr_saga":
                            log_coeffs = {
                                "model": mname,
                                "feat": fname,
                                "sg_key": s,
                                "subgroup": g,
                                "ts": datetime.now(),
                                "auc": auc,
                                "aps": aps,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "auprc": auc_precision_recall,
                                "fnr": fnr,
                                "tnr": tnr,
                                "fpr": fpr,
                                "test_samp_size": len(sg_test_y),
                                "logr_feat_names": fcolumns,
                                "logr_coeffs": model.coef_.tolist(),
                                "logr_intercept": model.intercept_.tolist(),
                                "bestmodel_train_params_dict": bestmodel_train_params,
                                "bestmodel_unique_filename": bestmodel_unique_filename,
                            }
                            # Save logr coefficients. For later comparison btw black, white, & other subgroups.
                            logcoeffs_mcol.insert_one({**base_res, **log_coeffs})
                    # Save fairness for this combo of mname + fname.
                    # NOTE: this currently only works for 1 protected characterstic (e.g., race)
                    # After subgroup loop, calculate fairness:
                    eo_ratio, fpr_parity, tpr_parity, fnr_parity = (
                        evaluate_results_fairness(gs, subgroup_preds_dict)
                    )

                    fair_res = {
                        "model": mname,
                        "ts": datetime.now(),
                        "eo_ratio": eo_ratio,
                        "fpr_parity": fpr_parity,
                        "tpr_parity": tpr_parity,
                        "fnr_parity": fnr_parity,
                        "sg_key": s,
                        "prot_attributes": subgroups_bins[s],
                        "total_test_samp_size": len(test_y),
                        "bestmodel_train_params_dict": bestmodel_train_params,
                        "bestmodel_unique_filename": bestmodel_unique_filename,
                    }
                    fairness_mcol.insert_one({**base_res, **fair_res})

    mclient.close()


if __name__ == "__main__":
    main()
