# inference.py

import os
import joblib
import shap

from geopy.distance import geodesic
from rapidfuzz import fuzz
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# 1) The exact feature names, in the same order you trained on:
FEATURES = [
    "dist_km",
    "room_diff",
    "bed_diff",
    "bath_diff",
    "lot_diff",
    "age_diff",
    "gla_diff",
    "style_sim",
    "heating_sim",
    "cooling_sim",
    "property_class_sim",
]


def compute_features(subject: Dict, prop: Dict) -> Dict[str, float]:
    """
    Given a subject and a single property (as dicts), compute the feature vector.
    Assumes each dict has the necessary keys (lat, lon, room_total, etc.).
    """
    f = {}
    # 1. distance in km
    f["dist_km"] = geodesic(
        (subject["lat"], subject["lon"]),
        (prop["lat"],    prop["lon"])
    ).kilometers

    # 2. numeric diffs
    f["room_diff"] = abs(subject["room_total"] - prop["room_total"])
    f["bed_diff"] = abs(subject["num_beds"] - prop["num_beds"])
    f["bath_diff"] = abs(subject["num_baths"] - prop["num_baths"])
    f["lot_diff"] = abs(subject["lot_size_sf"] - prop["lot_size_sf"])
    f["age_diff"] = abs(subject["year_built"] - prop["year_built"])
    f["gla_diff"] = abs(subject["gla"] - prop["gla"])

    # 3. categorical/text similarities (scaled 0–1)
    f["style_sim"] = fuzz.ratio(subject["style"],
                                prop["style"]) / 100.0
    f["heating_sim"] = fuzz.ratio(
        subject["heating"],          prop["heating"]) / 100.0
    f["cooling_sim"] = fuzz.ratio(
        subject["cooling"],          prop["cooling"]) / 100.0
    f["property_class_sim"] = fuzz.ratio(
        subject["structure_type"],   prop["structure_type"]) / 100.0

    return f


def rank_properties(
    subject: Dict,
    properties: List[Dict],
    model_path: str = "xgboost_model.joblib",
    top_k: int = 3
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Returns:
      - top_k_df: a DataFrame of the top-k properties with their scores
      - explanations: a list of dicts, one per top-k, including SHAP breakdown
    """
    # — load the trained model
    model: XGBClassifier = joblib.load(model_path)

    # — build feature matrix
    feats = [compute_features(subject, p) for p in properties]
    df = pd.DataFrame(feats)
    df["property_id"] = [p.get("id", idx) for idx, p in enumerate(properties)]

    # — score them
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]

    # — SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[FEATURES])

    # — pick top_k
    topk_df = df.nlargest(top_k, "score").reset_index(drop=True)
    print(topk_df.head())

    explanations = []
    for i, row in topk_df.iterrows():
        sv = shap_values[row.name]  # SHAP values for this row
        # pick the 3 features with largest absolute contribution
        top_feats = sorted(
            zip(FEATURES, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        explanations.append({
            "property_id": row["property_id"],
            "score": float(row["score"]),
            "feature_values": {f: float(row[f]) for f in FEATURES},
            "top_contributors": [
                {"feature": f, "shap_value": float(v)} for f, v in top_feats
            ]
        })

    return topk_df, explanations


def format_llm_explanation(
    subject: Dict,
    explanations: List[Dict]
) -> str:
    """
    Turn the SHAP outputs into a prompt you can hand to an LLM for a human-readable write-up.
    """
    lines = []
    lines.append(
        f"We have one subject with ID `{subject.get('id','<subject>')}` and the following attributes:")
    for k, v in subject.items():
        if k in ["lat", "lon"]:
            continue
        lines.append(f"  • **{k}**: {v}")
    lines.append(
        "\nWe scored the candidate properties and selected the top 3 because:\n")

    for exp in explanations:
        pid = exp["property_id"]
        lines.append(f"**Property {pid}** (score={exp['score']:.3f}):")
        for contr in exp["top_contributors"]:
            lines.append(
                f"  - `{contr['feature']}` contributed `{contr['shap_value']:.4f}`")
        lines.append("")  # blank line

    return "\n".join(lines)


if __name__ == "__main__":
    # — EXAMPLE USAGE — replace these with your real data
    subject_example = {
        "id": "SYN_SUBJ_1",
        "lat": 42.3601,
        "lon": -71.0942,
        "room_total": 7,
        "num_beds": 3,
        "num_baths": 2,
        "lot_size_sf": 5500,
        "year_built": 1990,
        "gla": 2000,
        "style": "Ranch",
        "heating": "Gas",
        "cooling": "Central",
        "structure_type": "Single Family"
    }

    properties_example = [
        {
            "id": "SYN_PROP_1", "lat": 42.373151, "lon": -71.100106,
            "room_total": 9, "num_beds": 2, "num_baths": 1,
            "lot_size_sf": 5794, "year_built": 1983, "gla": 1114,
            "style": "Ranch", "heating": "Oil", "cooling": "None",
            "structure_type": "Townhouse"
        },
        {
            "id": "SYN_PROP_2", "lat": 42.354277, "lon": -71.101438,
            "room_total": 7, "num_beds": 1, "num_baths": 1,
            "lot_size_sf": 4840, "year_built": 1990, "gla": 1343,
            "style": "Ranch", "heating": "Gas", "cooling": "None",
            "structure_type": "Single Family"
        },
        {
            "id": "SYN_PROP_3", "lat": 42.363718, "lon": -71.106573,
            "room_total": 9, "num_beds": 2, "num_baths": 3,
            "lot_size_sf": 6460, "year_built": 2002, "gla": 2080,
            "style": "Modern", "heating": "Gas", "cooling": "Window",
            "structure_type": "Single Family"
        },
        {
            "id": "SYN_PROP_4", "lat": 42.352474, "lon": -71.094044,
            "room_total": 5, "num_beds": 1, "num_baths": 3,
            "lot_size_sf": 5679, "year_built": 2024, "gla": 1209,
            "style": "Ranch", "heating": "Oil", "cooling": "None",
            "structure_type": "Townhouse"
        },
        {
            "id": "SYN_PROP_5", "lat": 42.356881, "lon": -71.095845,
            "room_total": 9, "num_beds": 2, "num_baths": 1,
            "lot_size_sf": 4793, "year_built": 1987, "gla": 1772,
            "style": "Ranch", "heating": "Gas", "cooling": "None",
            "structure_type": "Townhouse"
        },
    ]

    topk_df, exps = rank_properties(subject_example, properties_example)
    print("=== TOP-3 ===")
    print(topk_df[["property_id", "score"]])

    print("\n=== LLM EXPLANATION PROMPT ===")
    print(format_llm_explanation(subject_example, exps))
