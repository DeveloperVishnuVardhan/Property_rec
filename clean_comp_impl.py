import polars as pl
from logger_config import setup_logger
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from data_interfaces import CleanData
from data_utils import (
    clean_numeric_string, parse_lot_size, parse_bedrooms, parse_bathrooms,
    parse_age, parse_date, parse_distance, parse_location_similarity,
    parse_sale_price, parse_dom
)


class CleanCompImpl(CleanData):
    """Implementation of CleanData for processing comparable property data.

    This class handles the extraction, cleaning, and standardization of comparable property
    data from appraisal records. It includes methods for data cleaning, feature derivation,
    and null value handling.
    """

    def __init__(self) -> None:
        """Initialize the CleanCompImpl with a logger instance."""
        self.logger = setup_logger(__name__)

    def prep(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Prepare comparable property data from raw appraisal data.

        This method orchestrates the entire data preparation process:
        1. Extracts comparable property data from appraisals
        2. Cleans and standardizes the data
        3. Derives additional features
        4. Handles null values

        Args:
            data (Dict[str, Any]): Raw appraisal data containing comparable property information.

        Returns:
            pl.DataFrame: Cleaned and standardized comparable property data.
        """
        self.logger.info("Starting comparable property data preparation")
        comps: List[Dict[str, Any]] = self.__extract_comps(data)
        cleaned_comps: pl.DataFrame = self.__clean_comps(comps)
        self.logger.info(
            f"Successfully extracted {len(comps)} comparable properties")
        return cleaned_comps

    def __extract_comps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comparable property data from appraisals.

        Processes each appraisal in the data and extracts the comparable property information,
        adding the orderID as an identifier.

        Args:
            data (Dict[str, Any]): Raw appraisal data containing comparable property information.

        Returns:
            List[Dict[str, Any]]: List of comparable property dictionaries with orderID identifiers.
        """
        self.logger.debug(
            "Extracting comparable properties from appraisals data")
        comps: List[Dict[str, Any]] = []
        for appraisal in data.get('appraisals', []):
            # Get comparable properties for this appraisal
            appraisal_comps = appraisal.get('comps', [])

            for comp in appraisal_comps:
                # Create a copy of the comp data
                comp_data: Dict[str, Any] = comp.copy()

                # Add orderID as an identifier
                comp_data['orderID'] = appraisal.get('orderID')
                comps.append(comp_data)

                self.logger.debug(
                    f"Processed comparable property with orderID: {comp_data.get('orderID')}")

        return comps

    def __clean_comps(self, comps: List[Dict[str, Any]]) -> pl.DataFrame:
        """Clean and standardize comparable property data.

        Processes each comparable property dictionary to:
        1. Clean and standardize string fields
        2. Parse and convert numeric fields
        3. Handle date fields
        4. Convert to a Polars DataFrame
        5. Derive additional features
        6. Handle null values

        Args:
            comps (List[Dict[str, Any]]): List of comparable property dictionaries from appraisals.

        Returns:
            pl.DataFrame: Cleaned and standardized comparable property data in a Polars DataFrame.
        """
        cleaned_comps: List[Dict[str, Any]] = []
        for comp in comps:
            cleaned: Dict[str, Any] = {}
            cleaned['orderID'] = comp.get('orderID')

            # String fields (keep as is)
            cleaned['address'] = comp.get('address')
            cleaned['city_province'] = comp.get('city_province')
            cleaned['prop_type'] = comp.get('prop_type')
            cleaned['condition'] = comp.get('condition')
            cleaned['basement_finish'] = comp.get('basement_finish')
            cleaned['parking'] = comp.get('parking')
            cleaned['neighborhood'] = comp.get('neighborhood')
            cleaned['stories'] = comp.get('stories')

            # Numeric fields
            cleaned['distance_to_subject'] = parse_distance(
                comp.get('distance_to_subject'))
            cleaned['sale_price'] = parse_sale_price(comp.get('sale_price'))
            cleaned['dom'] = parse_dom(comp.get('dom'))
            cleaned['lot_size'] = parse_lot_size(comp.get('lot_size'))
            cleaned['age'] = parse_age(comp.get('age'))
            cleaned['gla'] = clean_numeric_string(comp.get('gla'))

            # Handle room_count with '+' notation
            room_count_str: Optional[str] = comp.get('room_count', '')
            if isinstance(room_count_str, str) and '+' in room_count_str:
                parts = room_count_str.split('+')
                try:
                    main_rooms: float = float(parts[0])
                    additional_rooms: float = float(
                        parts[1]) if len(parts) > 1 else 0
                    cleaned['room_count'] = main_rooms + additional_rooms
                except (ValueError, IndexError):
                    cleaned['room_count'] = clean_numeric_string(
                        room_count_str)
            else:
                cleaned['room_count'] = clean_numeric_string(room_count_str)

            cleaned['bed_count'] = parse_bedrooms(comp.get('bed_count'))
            cleaned['bath_count'] = parse_bathrooms(comp.get('bath_count'))

            # Categorical fields
            cleaned['location_similarity'] = parse_location_similarity(
                comp.get('location_similarity'))

            # Date fields
            cleaned['sale_date'] = parse_date(comp.get('sale_date'))

            # Add to list
            cleaned_comps.append(cleaned)

        # Convert to DataFrame and derive additional features
        df: pl.DataFrame = pl.DataFrame(cleaned_comps)
        df = self.__derive_additional_features(df)
        df = self.__handle_nulls(df)
        return df

    def __derive_additional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Derive additional features from existing data.

        Creates new features based on existing data:
        1. Calculates price per square foot
        2. Creates property_class based on condition and GLA
        3. Calculates time-based features from sale_date
        4. Extracts lot size unit

        Args:
            df (pl.DataFrame): Input DataFrame with cleaned comparable property data.

        Returns:
            pl.DataFrame: DataFrame with additional derived features.
        """
        # Calculate price per square foot
        if 'sale_price' in df.columns and 'gla' in df.columns:
            df = df.with_columns([
                (pl.col('sale_price') / pl.col('gla')).alias('price_per_sqft')
            ])

        # Create property_class feature
        if 'condition' in df.columns and 'gla' in df.columns:
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

        # Calculate time-based features
        if 'sale_date' in df.columns:
            df = df.with_columns([
                pl.col('sale_date').dt.year().alias('sale_year'),
                pl.col('sale_date').dt.month().alias('sale_month'),
                pl.col('sale_date').dt.quarter().alias('sale_quarter')
            ])

        # Extract lot size unit
        if 'lot_size' in df.columns:
            # Create lot size unit column using when/otherwise expressions
            df = df.with_columns([
                pl.when(pl.col('lot_size') < 1)
                .then(pl.lit('Acre'))
                .when(pl.col('lot_size') > 10000)
                .then(pl.lit('SqFt'))
                .otherwise(pl.lit('SqM'))
                .alias('lot_size_unit')
            ])

        return df

    def __handle_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle null values in the DataFrame.

        Implements different strategies for handling null values:
        1. Numeric columns: filled with median values
        2. Categorical columns: filled with mode values
        3. Date columns: filled with most recent date

        Args:
            df (pl.DataFrame): Input DataFrame with null values.

        Returns:
            pl.DataFrame: DataFrame with null values handled appropriately.
        """
        # Numeric columns to fill with median
        numeric_cols: List[str] = [
            'distance_to_subject', 'stories', 'sale_price', 'dom',
            'lot_size', 'age', 'gla', 'room_count', 'bed_count',
            'bath_count', 'price_per_sqft'
        ]

        # Fill numeric columns with median
        for col in numeric_cols:
            if col in df.columns:
                median: Optional[float] = df[col].median()
                if median is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(median).alias(col)
                    ])

        # Categorical columns to fill with mode
        categorical_cols: List[str] = [
            'prop_type', 'condition', 'basement_finish', 'parking',
            'neighborhood', 'location_similarity', 'property_class'
        ]

        # Fill categorical columns with mode
        for col in categorical_cols:
            if col in df.columns:
                # Skip if column is all null
                if df[col].is_null().all():
                    continue

                # Get mode
                mode_value: Optional[Any] = df[col].mode().item(
                ) if not df[col].mode().is_empty() else None
                if mode_value is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(mode_value).alias(col)
                    ])

        # Date columns to fill with most recent date
        date_cols: List[str] = ['sale_date']
        for col in date_cols:
            if col in df.columns:
                # Get most recent date
                max_date: Optional[datetime] = df[col].max()
                if max_date is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(max_date).alias(col)
                    ])

        return df
