import json
import pandas as pd
import numpy as np
import sklearn.metrics as skm


# Fairness metrics
# from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
# from aif360.sklearn.metrics import generalized_fnr, difference


def read_json(infile):
    """
    Load a json file
    """
    with open(infile, "r") as ifile:
        return json.load(ifile)


"""
IF fairness is low btw subgroups, 
- try adding "fairness" penalty (e.g., eq odds, eq opportunity, disp impact)

"""


def evaluate_results(model, test_x, test_y, endpoint, imputer=None, scaler=None):

    if imputer is not None:
        test_x = imputer.transform(test_x)

    if scaler is not None:
        test_x = scaler.transform(test_x)

    # evaluate on test
    y_hat = model.predict_proba(test_x)[:, 1]

    # test_y = test_y[endpoint].values
    # print(test_y.value_counts() )
    # print(model)

    # Performance metrics
    auc = skm.roc_auc_score(test_y, y_hat)
    aps = skm.average_precision_score(test_y, y_hat)

    precision, recall, thresholds = skm.precision_recall_curve(test_y, y_hat)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = skm.auc(recall, precision)

    # Get more metrics
    binary_predictions = model.predict(test_x)

    precision = skm.precision_score(test_y, binary_predictions, zero_division=np.nan)
    recall = skm.recall_score(
        test_y, binary_predictions, zero_division=np.nan
    )  # i.e., TPR, sensitivity, "equal opportunity"
    f1 = skm.f1_score(test_y, binary_predictions, zero_division=np.nan)
    mcc = skm.matthews_corrcoef(test_y, binary_predictions)

    # Other metrics using confusion matrix:

    conf_matrix = skm.confusion_matrix(test_y, binary_predictions)
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    # False neg
    # True positives
    tnr = TN / (TN + FP)
    fpr = 1 - tnr

    # Get more fairness metrics:
    # 1. FNR (i.e., "miss rate")
    if not (recall == np.nan):
        fnr = 1 - recall
    else:
        fnr = np.nan

    # 2. .... others must be calcualted AFTER all subgroups are run for a given feature set.

    return (
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
    )


def get_train_test(df, i, label):
    test_mask = df[label + "_folds"] == i
    train_df = df[~test_mask]
    test_df = df[test_mask]
    # setup y
    train_y = train_df[label]
    test_y = test_df[label]
    return train_df, test_df, train_y, test_y


"""
New implementation of fairness metrics.

gs: not needed here since preds for each subgroup already passed in subgroup_preds_dict. 
"""


def evaluate_results_fairness(gs, subgroup_preds_dict):

    list_preds = []  # 2d list
    list_true = []
    for k, v in subgroup_preds_dict.items():
        subgroup = k
        if subgroup != "blackorwhite":
            test_x, sg_test_y, binary_predictions = v
            # evaluate on test
            # y_hat = model.predict_proba(test_x)[:, 1]
            # binary_predictions =  gs.predict(test_x)
            # Evaluate on sg test set (current model + feats)

            list_preds.append(binary_predictions)
            list_true.append(sg_test_y)

        else:  # Subgroup is "all" fairness doesn't need calculated.
            return np.nan, np.nan, np.nan, np.nan

    # di_ratio = disparate_impact_ratio(y_true=test_y, y_pred=y_hat, prot_attr='black')
    # ao_ratio = average_odds_error(y_true=test_y, y_pred=y_hat, prot_attr='black')

    # dpr = demographic_parity_ratio(list_preds)
    eor = equalized_odds_ratio(list_preds, list_true)
    fpr_par = fpr_parity(list_preds, list_true)
    tpr_par = tpr_parity(list_preds, list_true)

    fnr_par = fnr_parity(list_preds, list_true)

    return eor, fpr_par, tpr_par, fnr_par


def fpr_parity(prediction_lists, true_labels):
    """
    False positive rate (FPR) parity: achieved if the FPR (the ratio between the number of false
    positives and the total number of negatives) in the subgroups are close to each other.

    Args:
    prediction_lists: A list of lists containing binary predictions for different subgroups.
    true_labels: A list of lists containing the true binary labels for each subgroup,
                    in the same order as prediction_lists.

    Returns:
    The FPR parity as a float.

    """

    # Calculate FPR for each group
    fprs = []
    for predictions, true_label in zip(prediction_lists, true_labels):
        # tp = np.sum(np.logical_and(predictions == 1, true_label == 1))
        fp = np.sum(np.logical_and(predictions == 1, true_label == 0))
        tn = np.sum(np.logical_and(predictions == 0, true_label == 0))
        # fn = np.sum(np.logical_and(predictions == 0, true_label == 1))
        # tpr = tp / (tp + fn)  # True positive rate
        fpr = fp / (fp + tn)  # False positive rate
        fprs.append(fpr)
    # Catch:

    # Minimum and maximum and FPRs
    min_fpr = min(fprs)
    max_fpr = max(fprs)

    # Catch: undefined TPR parity
    if max_fpr == 0 or max_fpr == None or max_fpr == np.nan:
        return np.nan

    # Calculate and return
    fpr_ratio = min_fpr / max_fpr

    return fpr_ratio


def tpr_parity(prediction_lists, true_labels):
    """
    [TPR parity is sometimes called "equality of opporunity"]
    True positive rate (FPR) parity:
    Args:
    prediction_lists: A list of lists containing binary predictions for different subgroups.
    true_labels: A list of lists containing the true binary labels for each subgroup,
                    in the same order as prediction_lists.

    Returns:
    The TPR parity as a float.

    """
    # Calculate TPR for each group
    tprs = []
    for predictions, true_label in zip(prediction_lists, true_labels):
        tp = np.sum(np.logical_and(predictions == 1, true_label == 1))
        # fp = np.sum(np.logical_and(predictions == 1, true_label == 0))
        # tn = np.sum(np.logical_and(predictions == 0, true_label == 0))
        fn = np.sum(np.logical_and(predictions == 0, true_label == 1))
        tpr = tp / (tp + fn)  # True positive rate
        # fpr = fp / (fp + tn)  # False positive rate
        tprs.append(tpr)

    # Minimum and maximum TPRs
    min_tpr = min(tprs)
    max_tpr = max(tprs)

    # Catch: undefined TPR parity
    if max_tpr == 0 or max_tpr == None or max_tpr == np.nan:
        return np.nan

    # Calculate and return the TPR parity
    tpr_ratio = min_tpr / max_tpr

    return tpr_ratio


def fnr_parity(prediction_lists, true_labels):
    """
    False negative rate (FNR) parity: achieved if the FNR in the subgroups are close to each other.

    Args:
    prediction_lists: A list of lists containing binary predictions for different subgroups.
    true_labels: A list of lists containing the true binary labels for each subgroup,
                    in the same order as prediction_lists.

    Returns:
    The FNR parity as a float.

    """

    # Calculate FPR for each group
    fnrs = []
    for predictions, true_label in zip(prediction_lists, true_labels):
        tp = np.sum(np.logical_and(predictions == 1, true_label == 1))
        # fp = np.sum(np.logical_and(predictions == 1, true_label == 0))
        # tn = np.sum(np.logical_and(predictions == 0, true_label == 0))
        fn = np.sum(np.logical_and(predictions == 0, true_label == 1))
        # tpr = tp / (tp + fn)  # True positive rate
        fnr = fn / (tp + fn)  # False positive rate
        fnrs.append(fnr)
    # Catch:

    # Minimum and maximum and FPRs
    min_fnr = min(fnrs)
    max_fnr = max(fnrs)

    # Catch: undefined TPR parity
    if max_fnr == 0 or max_fnr == None or max_fnr == np.nan:
        return np.nan

    # Calculate and return
    fnr_ratio = min_fnr / max_fnr

    return fnr_ratio


def demographic_parity_ratio(prediction_lists):
    """
    Calculates the demographic parity ratio for multiple lists of binary predictions.

    Args:
    *prediction_lists: (2d Python list) A variable number of lists containing binary predictions for different subgroups.

    Returns:
    The demographic parity ratio as a float.
    """

    # Calculate positive rates for each group
    positive_rates = [
        sum(predictions) / len(predictions) for predictions in prediction_lists
    ]

    # Minimum and maximum positive rates
    min_positive_rate = min(positive_rates)
    max_positive_rate = max(positive_rates)

    # Calculate and return the demographic parity ratio
    return min_positive_rate / max_positive_rate


def equalized_odds_ratio(prediction_lists, true_labels):
    """
    [Note: Equality of odds is satisfied only when TP parity & FP parity are satisfied.]

    Calculates the equalized odds ratio for multiple lists of binary predictions.

    Args:
    prediction_lists: A list of lists containing binary predictions for different subgroups.
    true_labels: A list of lists containing the true binary labels for each subgroup,
                    in the same order as prediction_lists.

    Returns:
    The equalized odds ratio as a float.
    """

    # Check for matching lengths and inner list lengths
    if len(prediction_lists) != len(true_labels):
        raise ValueError(
            "The number of prediction lists must match the number of true label lists."
        )
    for predictions, true_label in zip(prediction_lists, true_labels):
        if len(predictions) != len(true_label):
            raise ValueError(
                "Each prediction list must have the same length as its corresponding true label list."
            )

    # Calculate TPR and FPR for each group
    tprs = []
    fprs = []
    for predictions, true_label in zip(prediction_lists, true_labels):
        tp = np.sum(np.logical_and(predictions == 1, true_label == 1))
        fp = np.sum(np.logical_and(predictions == 1, true_label == 0))
        tn = np.sum(np.logical_and(predictions == 0, true_label == 0))
        fn = np.sum(np.logical_and(predictions == 0, true_label == 1))
        tpr = tp / (tp + fn)  # True positive rate
        fpr = fp / (fp + tn)  # False positive rate
        tprs.append(tpr)
        fprs.append(fpr)

    # Minimum and maximum TPRs and FPRs
    min_tpr = min(tprs)
    max_tpr = max(tprs)
    min_fpr = min(fprs)
    max_fpr = max(fprs)

    # Calculate and return the equalized odds ratio
    tpr_ratio = min_tpr / max_tpr
    fpr_ratio = min_fpr / max_fpr

    # Catch:
    # If max TPR = None
    if tpr_ratio == None or max_tpr == 0:
        return np.nan

    # If max FPR = 0, or None
    if fpr_ratio == None or fpr_ratio == 0 or max_fpr == 0:
        return np.nan

    return min(
        tpr_ratio, fpr_ratio
    )  # Choose the smaller ratio for a more conservative measure
