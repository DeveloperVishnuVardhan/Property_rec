from typing import Dict, Any
import polars as pl
import numpy as np
import random
import json
import os

from clean_comp_impl import CleanCompImpl
from clean_property_impl import CleanPropertyImpl
from clean_subject_impl import CleanSubjectImpl
from standardize_address_impl import StandardizeAddress

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load the data


def load_data(file_path: str) -> Dict:
    """Load JSON data from a file and return a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Preprocess the data
if not os.path.exists('subject_df.csv') and not os.path.exists('comps_df.csv') and not os.path.exists('property_df.csv'):
    data: Dict[str, Any] = load_data('appraisals_dataset.json')
    subject_df: pl.DataFrame = CleanSubjectImpl().prep(data)
    comps_df: pl.DataFrame = CleanCompImpl().prep(data)
    property_df: pl.DataFrame = CleanPropertyImpl().prep(data)
    standardize_address: StandardizeAddress = StandardizeAddress(
        subject_df, comps_df, property_df)
    subject_df, comps_df, property_df = standardize_address.standardize()

    # save the data.
    subject_df.write_csv('subject_df.csv')
    comps_df.write_csv('comps_df.csv')
    property_df.write_csv('property_df.csv')
