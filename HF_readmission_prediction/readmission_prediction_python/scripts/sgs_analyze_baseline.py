import argparse
import datetime
import pandas as pd
from pymongo import MongoClient
import urllib.parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-output",
        help="output file",
        default="TODO: output csv file name to output results from mongo.",
    )
    username = urllib.parse.quote_plus("TODO: mongo username")
    password = urllib.parse.quote_plus("TODO: mongo password")
    parser.add_argument(
        "-mongo_url", default="TODO: your mongo server url" % (username, password)
    )
    parser.add_argument("-mongo_db", default="TODO: your mongo database name.")
    parser.add_argument(
        "-mongo_col", default="TODO: your mongo collection name", help="collection_type"
    )  # For subgroup results, set default="subgroups_baseline", for entire dataset, default="baseline"
    args = parser.parse_args()

    # setup the mongo stuff
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    pipe_list = [
        {
            "$project": {
                "_id": 0,
                "model": "$model",
                "feat": "$feat",
                "file": "$file",
                "fold": "$fold",
                "endpoint": "$endpoint",
                "sg_key": "$sg_key",
                "subgroup": "$subgroup",
                "auc": "$auc",
                "aps": "$aps",
                "precision": "$precision",
                "recall": "$recall",
                "f1": "$f1",
                "auprc": "$auprc",
                "fnr": "$fnr",
                "tnr": "$tnr",
                "fpr": "$fpr",
                "mcc": "$mcc",
                "test_samp_size": "$test_samp_size",
            }
        }
    ]

    tmp = list(mcol.aggregate(pipe_list))
    tmp_df = pd.DataFrame.from_records(tmp)
    mclient.close()

    tmp_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()


"""
OLD PIPE LIST: if you would like to directly averaging fold performances.
We use each of the 10 folds individually for our analysis in order to perform paired t-tests.


 pipe_list = [
         {
        
    "$group":
            {
                "_id":
                {
                    "feat": "$feat",
                    "model": "$model",
                    "file": "$file",
                    "endpoint": "$endpoint",
                    "sg_key": "$sg_key",
                    "subgroup": "$subgroup"
                },


                "auc":
                {
                    "$avg": "$auc"
                },
                "auc_sd":
                {
                    "$stdDevSamp": "$auc"
                },
                "aps":
                {
                    "$avg": "$aps"
                },
                "aps_sd":
                {
                    "$stdDevSamp": "$aps"
                },
                
                "precision":
                {
                    "$avg": "$precision"
                },
                "precision_sd":
                {
                    "$stdDevSamp": "$precision"
                },
                "recall":
                {
                    "$avg": "$recall"
                },
                "recall_sd":
                {
                    "$stdDevSamp": "$recall"
                },
                "f1":
                {
                    "$avg": "$f1"
                },
                "f1_sd":
                {
                    "$stdDevSamp": "$f1"
                },

                "auprc":
                {
                    "$avg": "$auprc"
                },
                "auprc_sd":
                {
                    "$stdDevSamp": "$auprc"
                },
                "fnr":
                {
                    "$avg": "$fnr"
                },
                "fnr_sd":
                {
                    "$stdDevSamp": "$fnr"
                },

                "tnr":
                {
                    "$avg": "$tnr"
                },
                "tnr_sd":
                {
                    "$stdDevSamp": "$tnr"
                },
                "fpr":
                {
                    "$avg": "$fpr"
                },
                "fpr_sd":
                {
                    "$stdDevSamp": "$fpr"
                },
                "mcc":
                {
                    "$avg": "$mcc"
                },
                "mcc_sd":
                {
                    "$stdDevSamp": "$mcc"
                },


               "n_runs":
               {
                   "$sum": 1
                   
               },


                

                "test_samp_size": 
                {
                    "$avg": "$test_samp_size"
                }
            }
    },

    {"$project":
            {
                "_id": 0,
                "model": "$_id.model",
                "feat": "$_id.feat",
                "file": "$_id.file",
                "endpoint": "$_id.endpoint",
                
                "auc": "$auc",
                "auc_sd": "$auc_sd",

                "aps": "$aps",
                "aps_sd": "$aps_sd",

                "precision": "$precision",
                "precision_sd": "$precision_sd",

                "recall": "$recall",
                "recall_sd": "$recall_sd",

                "f1": "$f1",
                "f1_sd": "$f1_sd",

                "auprc": "$auprc",
                "auprc_sd": "$auprc_sd",

                "fnr": "$fnr",
                "fnr_sd": "$fnr_sd",

                "tnr": "$tnr",
                "tnr_sd": "$tnr_sd",

                "fpr": "$fpr",
                "fpr_sd": "$fpr_sd",

                "mcc": "$mcc",
                "mcc_sd": "$mcc_sd",


                "sg_key": "$_id.sg_key",
                "subgroup": "$_id.subgroup",


                "n_runs": "$n_runs",
                "test_samp_size": "$test_samp_size"
            }
            }
    ]


"""
