import os
import polars as pl
from geopy.distance import geodesic
from rapidfuzz import fuzz
from ranking_data import RankingData
from xgboost import XGBClassifier
import joblib
from typing import Tuple, List

import numpy as np
import pandas as pd


def precision_recall_at_k(df: pd.DataFrame, score_col: str, k: int = 3) -> Tuple[float, float]:
    """
    Compute average Precision@k and Recall@k over all orderID groups.

    Args:
      df        : DataFrame containing columns ['orderID', score_col, 'label']
      score_col : name of the column with model scores
      k         : how many top items to consider per group

    Returns:
      (precision, recall) : tuple of floats
    """
    precisions: List[float] = []
    recalls: List[float] = []

    # loop over each query (orderID)
    for order_id, group in df.groupby("orderID"):
        # take the top-k by score
        topk = group.nlargest(k, score_col)

        # how many of those are true positives?
        num_true = topk["label"].sum()
        # how many actual positives existed in this group?
        total_true = group["label"].sum()

        precisions.append(num_true / k)
        recalls.append(num_true / total_true if total_true > 0 else 0.0)

    # average across all queries
    return np.mean(precisions), np.mean(recalls)


# Feature Engineering.
subjects_df: pl.DataFrame = pl.read_csv("subject_df.csv")
comps_df: pl.DataFrame = pl.read_csv("comps_df.csv")
property_df: pl.DataFrame = pl.read_csv("property_df.csv")

ranking = RankingData(subjects_df, comps_df, property_df)

train_data = ranking.create_ranking_data(type="train")
test_data = ranking.create_ranking_data(type="test")

features: List[str] = ['dist_km',
                       'room_diff',
                       'bed_diff',
                       'bath_diff',
                       'lot_diff',
                       'age_diff',
                       'gla_diff',
                       'style_sim',
                       'heating_sim',
                       'cooling_sim',
                       'property_class_sim']

train_pdf = train_data.select(["orderID", *features, "label"]).to_pandas()
X, y = train_pdf[features], train_pdf["label"]

if not os.path.exists("xgboost_model.joblib"):
    pt_model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        learning_rate=0.05,
        n_estimators=200,
        random_state=42
    )
    pt_model.fit(X, y)
    # Save the trained model
    joblib.dump(pt_model, 'xgboost_model.joblib')
else:
    pt_model = joblib.load("xgboost_model.joblib")
test_pdf = test_data.select(["orderID", *features, "label"]).to_pandas()
X_test, y_test = test_pdf[features], test_pdf["label"]

train_pdf["pt_score"] = pt_model.predict_proba(X)[:, 1]
test_pdf["pt_score"] = pt_model.predict_proba(X_test)[:, 1]

# Evaluate the model
# train precision and recall
pt_p, pt_r = precision_recall_at_k(train_pdf, "pt_score", k=3)
print(f"Train Precision: {pt_p}, Train Recall: {pt_r}")
# test precision and recall
pt_p, pt_r = precision_recall_at_k(test_pdf, "pt_score", k=3)
print(f"Test Precision: {pt_p}, Test Recall: {pt_r}")
