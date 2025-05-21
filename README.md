# Automax: Automated Property Appraisal System

Automax is an automated system for property appraisal that uses machine learning to identify and rank comparable properties for real estate valuation.

Metrics:
rain Precision@K: 0.9821428571428571, Train Recall@K: 0.9839285714285715
Test Precision@K: 0.9743589743589743, Test Recall@K: 0.9743589743589743

# Flow of explanation

1. Explaination of the project.
2. Core packages used.
3. Strategies to send into production and making the system even more better.

## Project Overview

The system processes property data from multiple sources, cleans and standardizes it, then uses a machine learning model to rank potential comparable properties (comps) for a subject property. The workflow is designed to automate what real estate appraisers traditionally do manually - find the most similar properties to use as reference points for valuation.

## Project Structure

### Core Components

- **Data Processing Pipeline**: Handles data extraction, cleaning, and standardization
- **Address Standardization**: Normalizes addresses and obtains geo-coordinates
- **Feature Engineering**: Creates relevant features for the ranking model
- **Machine Learning Model**: XGBoost POINT-WISE RANKING to rank comparable properties
- **Evaluation**: Metrics to assess model performance

### Workflow

1. Raw data is loaded from JSON files
2. Data is cleaned and standardized
3. Addresses are normalized and geo-coded
4. Features are engineered for the model
5. The model is trained to identify good comps
6. The model is evaluated using precision-K and recall-K metrics

## File Descriptions

### Main Processing Files

- `train_evaluate.py`: Main script that orchestrates model training and evaluation
- `ranking_data.py`: Creates training and testing datasets for the ranking model
- `data_prep.py`: Coordinates data preparation across the pipeline

### Order of execution

- python data_prep.py - prepares the data for training.
- python train_evaluate.py - Trains and evaluates the model on training set.

### Data Cleaning and Processing

- `clean_property_impl.py`: Cleans and standardizes property data - extensively uses regular expressions to achieve standardization.
- `clean_subject_impl.py`: Cleans and standardizes subject property data - extensively uses regular expressions to achieve standardization.
- `clean_comp_impl.py`: Cleans and standardizes comparable property data - extensively uses regular expressions to achieve standardization.
- `standardize_address_impl.py`: Standardizes addresses and retrieves geo-coordinates - utilized geopy library to normalize the addresses, leveraging already available lat, long in properities.
- `data_utils.py`: Utility functions for data cleaning and processing

### Model and Evaluation

- `xgboost_model.joblib`: Saved XGBoost model for property comparison ranking

### Data Interfaces and Configuration

- `data_interfaces.py`: Interface definitions for data processing components
- `logger_config.py`: Configuration for logging throughout the application

### Data Files

- `appraisals_dataset.json`: Raw data containing property appraisals
- `property_df.csv`: Processed property data
- `subject_df.csv`: Processed subject property data
- `comps_df.csv`: Processed comparable property data
- `sample.csv`: Sample dataset for testing

### Exploration and Analysis

- `explore_json.py`: Script for exploring the JSON dataset structure

## Key Features

1. **Address Standardization**: Uses geocoding to standardize addresses and obtain coordinates
2. **Feature Engineering**: Creates distance features, property similarity metrics, and other comparison features
3. **Machine Learning Ranking**: Uses XGBoost to rank potential comparable properties
4. **Precision and Recall Evaluation**: Evaluates model performance with industry-relevant metrics

## Usage

To train and evaluate the model:

```bash
python train_evaluate.py
```

This will:

1. Load the necessary data
2. Create training and testing datasets
3. Train the XGBoost model
4. Evaluate performance using precision and recall at k=3
5. Save the trained model to disk

## Technical Implementation

- **Data Processing**: Uses Polars for efficient dataframe operations
- **Geocoding**: Uses Nominatim geocoder with rate limiting
- **Machine Learning**: XGBoost classifier for ranking
- **Distance Calculation**: Uses geodesic distance to calculate property proximity
- **Text Similarity**: Uses fuzzy matching for text field comparison

## Features Used for Ranking

- `dist_km`: Distance between properties in kilometers
- `room_diff`: Difference in room count
- `bed_diff`: Difference in bedroom count
- `bath_diff`: Difference in bathroom count
- `lot_diff`: Difference in lot size
- `age_diff`: Difference in property age
- `gla_diff`: Difference in gross living area
- `style_sim`: Similarity in property style
- `heating_sim`: Similarity in heating system
- `cooling_sim`: Similarity in cooling system
- `property_class_sim`: Similarity in property class

## Logging

The project uses a structured logging system configured in `logger_config.py`. Logs are stored in the `logs/` directory.

## core packages

1.polars - Proved to be 10 to 11 times faster than pandas.
2.geopy - address normalization, distance calculation.
3.other general core ml packages.

## Production workflow

S3 (raw Data) -> EMR/Lambda (data cleaning) -> Feature Stores (various options) -> SageMaker (Model Training) -> SageMaker Model registry -> Flask/FastAPI backend -> AWS ECS - Fargate (Medium level load) -> AWS ECS- EC2 autoscaling (Heavy load) -> custom kubernetes engines - Very Heavy load

## Improving the system

### Two stage recommendation.

- In our current setup we are ranking every candidate.
- This becomes very slow in case of large candidate pool.
- we can make into this two stage like candidate generation -> Ranking
- we can use our ground truth subject, comps to build a two tower embedding model which learn to group similar pairs together (we can even leverage textual data and other non-intuitive features).
- use the above to generate a set of candidates and then use ranking model to score them (we can use non-binary ranking like giving absolute scores with regression objectives).

### Fine tuning LLM's to make explanations more better.

1. we can use a costly big model intially to generate responses using our refined prompts based on shap/any other framework explanations.
2. Then we can leverage GRPO and finetune our open source model and reduce our costs.
