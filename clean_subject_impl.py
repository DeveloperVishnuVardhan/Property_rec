import polars as pl
from logger_config import setup_logger
from typing import Any, Dict, List

from data_interfaces import CleanData
from data_utils import clean_numeric_string, parse_lot_size, parse_bedrooms, parse_bathrooms, parse_age, parse_date, parse_year_built


class CleanSubjectImpl(CleanData):
    """Implementation of SubjectPrep for processing appraisal subject data.

    This class handles the extraction, cleaning, and standardization of subject property
    data from appraisal records. It includes methods for data cleaning, feature derivation,
    and null value handling.
    """

    def __init__(self) -> None:
        """Initialize the SubjectPrepImpl with a logger instance."""
        self.logger = setup_logger(__name__)

    def prep(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Prepare subject data from raw appraisal data.

        This method orchestrates the entire data preparation process:
        1. Extracts subject data from appraisals
        2. Cleans and standardizes the data
        3. Derives additional features
        4. Handles null values

        Args:
            data (Dict[str, Any]): Raw appraisal data containing subject information.

        Returns:
            pl.DataFrame: Cleaned and standardized subject data.
        """
        self.logger.info("Starting subject data preparation")
        subjects: List[Dict[str, Any]] = self.__extract_subjects(data)
        cleaned_subjects: pl.DataFrame = self.__clean_subjects(subjects)
        self.logger.info(f"Successfully extracted {len(subjects)} subjects")
        return cleaned_subjects

    def __extract_subjects(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract subject data from appraisals.

        Processes each appraisal in the data and extracts the subject information,
        adding the orderID as an identifier.

        Args:
            data (Dict[str, Any]): Raw appraisal data containing subject information.

        Returns:
            List[Dict[str, Any]]: List of subject dictionaries with orderID identifiers.
        """
        self.logger.debug("Extracting subjects from appraisals data")
        subjects: List[Dict[str, Any]] = []
        for appraisal in data.get('appraisals', []):
            # Create a copy of the subject data
            subject: Dict[str, Any] = appraisal.get('subject', {}).copy()

            # Add orderID as an identifier
            subject['orderID'] = appraisal.get('orderID')
            subjects.append(subject)

            self.logger.debug(
                f"Processed subject with orderID: {subject.get('orderID')}")

        return subjects

    def __clean_subjects(self, subjects: List[Dict[str, Any]]) -> pl.DataFrame:
        """Clean and standardize subject data.

        Processes each subject dictionary to:
        1. Clean and standardize string fields
        2. Parse and convert numeric fields
        3. Handle date fields
        4. Convert to a Polars DataFrame
        5. Derive additional features
        6. Handle null values

        Args:
            subjects (List[Dict[str, Any]]): List of subject dictionaries from appraisals.

        Returns:
            pl.DataFrame: Cleaned and standardized subject data in a Polars DataFrame.
        """
        cleaned_subjects: List[Dict[str, Any]] = []
        for subject in subjects:
            cleaned: Dict[str, Any] = {}
            cleaned['orderID'] = subject.get('orderID')

            # clean and standardize fields.
            # String fields (keep as is)
            cleaned['address'] = subject.get('address')
            cleaned['subject_city_province_zip'] = subject.get(
                'subject_city_province_zip')
            cleaned['municipality_district'] = subject.get(
                'municipality_district')
            cleaned['structure_type'] = subject.get('structure_type')
            cleaned['style'] = subject.get('style')
            cleaned['construction'] = subject.get('construction')
            cleaned['basement'] = subject.get('basement')
            cleaned['exterior_finish'] = subject.get('exterior_finish')
            cleaned['foundation_walls'] = subject.get('foundation_walls')
            cleaned['flooring'] = subject.get('flooring')
            cleaned['plumbing_lines'] = subject.get('plumbing_lines')
            cleaned['heating'] = subject.get('heating')
            cleaned['fuel_type'] = subject.get('fuel_type')
            cleaned['water_heater'] = subject.get('water_heater')
            cleaned['cooling'] = subject.get('cooling')
            cleaned['condition'] = subject.get('condition')
            cleaned['roofing'] = subject.get('roofing')
            cleaned['windows'] = subject.get('windows')
            cleaned['unit_measurement'] = subject.get('units_sq_ft')
            cleaned['site_dimensions'] = subject.get('site_dimensions')

            # Numeric fields
            cleaned['lot_size_sf'] = parse_lot_size(subject.get('lot_size_sf'))
            cleaned['year_built'] = parse_year_built(subject.get('year_built'))
            cleaned['effective_age'] = clean_numeric_string(
                subject.get('effective_age'))
            cleaned['remaining_economic_life'] = clean_numeric_string(
                subject.get('remaining_economic_life'))
            cleaned['basement_area'] = clean_numeric_string(
                subject.get('basement_area'))
            cleaned['room_count'] = clean_numeric_string(
                subject.get('room_count'))
            cleaned['num_beds'] = parse_bedrooms(subject.get('num_beds'))
            cleaned['room_total'] = clean_numeric_string(
                subject.get('room_total'))
            cleaned['main_lvl_area'] = clean_numeric_string(
                subject.get('main_lvl_area'))
            cleaned['second_lvl_area'] = clean_numeric_string(
                subject.get('second_lvl_area'))
            cleaned['third_lvl_area'] = clean_numeric_string(
                subject.get('third_lvl_area'))
            cleaned['gla'] = clean_numeric_string(subject.get('gla'))
            cleaned['num_baths'] = parse_bathrooms(subject.get('num_baths'))

            # Date fields
            cleaned['effective_date'] = parse_date(
                subject.get('effective_date'))

            # Add to list
            cleaned_subjects.append(cleaned)

        # Convert to DataFrame and derive additional features
        df: pl.DataFrame = pl.DataFrame(cleaned_subjects)
        df: pl.DataFrame = self.__derive_additional_features(df)
        df: pl.DataFrame = self.__handle_nulls(df)
        return df

    def __derive_additional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Derive additional features from existing data.

        Creates new features based on existing data:
        1. Calculates total area (GLA) from component areas if missing
        2. Creates property_class based on condition and GLA

        Args:
            df (pl.DataFrame): Input DataFrame with cleaned subject data.

        Returns:
            pl.DataFrame: DataFrame with additional derived features.
        """
        # Calculate total area if GLA is missing but components exist
        gla_features: List[str] = ['main_lvl_area',
                                   'second_lvl_area', 'third_lvl_area']
        if 'gla' in df.columns and all(col in df.columns for col in gla_features):
            df = df.with_columns([
                pl.when(pl.col('gla').is_null())
                .then(
                    pl.col('main_lvl_area').fill_null(0) +
                    pl.col('second_lvl_area').fill_null(0) +
                    pl.col('third_lvl_area').fill_null(0)
                )
                .otherwise(pl.col('gla'))
                .alias('gla')
            ])

        # Create property_class feature
        if 'condition' in df.columns and 'gla' in df.columns:
            # Create property class using simple rules
            df = df.with_columns([
                pl.when(
                    (pl.col('condition') == 'Excellent') & (pl.col('gla') > 3000)
                ).then(pl.lit('Luxury'))
                .when(
                    (pl.col('condition') == 'Excellent') |
                    ((pl.col('condition') == 'Good') & (pl.col('gla') > 2500))
                ).then(pl.lit('Premium'))
                .when(
                    (pl.col('condition') == 'Good') |
                    ((pl.col('condition') == 'Average') & (pl.col('gla') > 1500))
                ).then(pl.lit('Standard'))
                .otherwise(pl.lit('Basic'))
                .alias('property_class')
            ])

        return df

    def __handle_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle null values in the DataFrame.

        Implements different strategies for handling null values:
        1. Numeric columns: filled with median values
        2. Categorical columns: filled with mode values

        Args:
            df (pl.DataFrame): Input DataFrame with null values.

        Returns:
            pl.DataFrame: DataFrame with null values handled appropriately.
        """
        # Numeric columns to fill with median
        numeric_cols = [
            'lot_size_sf', 'year_built', 'effective_age', 'remaining_economic_life',
            'basement_area', 'room_count', 'num_beds', 'room_total',
            'main_lvl_area', 'second_lvl_area', 'third_lvl_area', 'gla',
            'subject_age', 'num_baths', 'age'
        ]

        # Fill numeric columns with median
        for col in numeric_cols:
            if col in df.columns:
                median = df[col].median()
                if median is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(median).alias(col)
                    ])

        # Categorical columns to fill with mode
        categorical_cols = [
            'structure_type', 'style', 'construction', 'basement',
            'exterior_finish', 'foundation_walls', 'flooring',
            'plumbing_lines', 'heating', 'fuel_type', 'water_heater',
            'cooling', 'condition', 'roofing', 'windows',
            'property_class'
        ]

        # Fill categorical columns with mode
        for col in categorical_cols:
            if col in df.columns:
                # Skip if column is all null
                if df[col].is_null().all():
                    continue

                # Get mode
                mode_value = df[col].mode().item(
                ) if not df[col].mode().is_empty() else None
                if mode_value is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(mode_value).alias(col)
                    ])

        return df
