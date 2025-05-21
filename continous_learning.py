import polars as pl
from ranking_data import RankingData
from typing import List
import xgboost as xgb
import joblib

"""
Here we are building a continous learning pipeline assuming certain things:
1. we have already collected newer dataset of appraisals (including user feedback from our app).
2. Assumimg this is the logic of the online training and we already have event which triggers retrain based on timeframe/amount of required data to retrain.
3. We have a model which is already trained present in the model registry.
4. We have a process to collect user feedback from our app.
5. We have a process to collect new appraisals data from the market.
6. We have a process to collect new appraisals data from the market.
7. We have a process to collect new appraisals data from the market.
8. we have cleaned the data.
"""
# let's use the testing dataset to test the continous learning pipeline.
# Feature Engineering.
subjects_df: pl.DataFrame = pl.read_csv("subject_df.csv")
comps_df: pl.DataFrame = pl.read_csv("comps_df.csv")
property_df: pl.DataFrame = pl.read_csv("property_df.csv")

ranking = RankingData(subjects_df, comps_df, property_df)

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

test_pdf = test_data.select(["orderID", *features, "label"]).to_pandas()
new_X, new_y = test_pdf[features], test_pdf["label"]

# load the best model from our model registry.
model: xgb.XGBClassifier = joblib.load("xgboost_model.joblib")
# Incremental(warm-start) training
model.set_params(n_estimators=model.n_estimators + 50)
model.fit(
    new_X,
    new_y,
    xgb_model=model.get_booster(),  # <-- this is the warm-start
    verbose=True
)

# save the new model
# ASSUMING SAVING THIS TO MODEL REGISTRY IF BETTER PERFORMANCE.
joblib.dump(model, "xgboost_model_continous_learning.joblib")
