import polars as pl
from logger_config import setup_logger
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from data_interfaces import CleanData
from data_utils import (
    clean_numeric_string, parse_lot_size, parse_bedrooms, parse_bathrooms,
    parse_age, parse_date, parse_sale_price, parse_year_built
)


class CleanPropertyImpl(CleanData):
    """Implementation of CleanData for processing property data.

    This class handles the extraction, cleaning, and standardization of property
    data from real estate listings. It includes methods for data cleaning, feature derivation,
    and null value handling.
    """

    def __init__(self) -> None:
        """Initialize the CleanPropertyImpl with a logger instance."""
        self.logger = setup_logger(__name__)

    def prep(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Prepare property data from raw listing data.

        This method orchestrates the entire data preparation process:
        1. Extracts property data from listings
        2. Cleans and standardizes the data
        3. Derives additional features
        4. Handles null values

        Args:
            data (Dict[str, Any]): Raw listing data containing property information.

        Returns:
            pl.DataFrame: Cleaned and standardized property data.
        """
        self.logger.info("Starting property data preparation")
        properties: List[Dict[str, Any]] = self.__extract_properties(data)
        cleaned_properties: pl.DataFrame = self.__clean_properties(properties)
        self.logger.info(
            f"Successfully extracted {len(properties)} properties")
        return cleaned_properties

    def __extract_properties(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract property data from listings.

        Processes each appraisal in the data and extracts the property information,
        adding the orderID as an identifier.

        Args:
            data (Dict[str, Any]): Raw appraisal data containing property information.

        Returns:
            List[Dict[str, Any]]: List of property dictionaries with orderID identifiers.
        """
        self.logger.debug("Extracting properties from appraisals data")
        properties: List[Dict[str, Any]] = []

        for appraisal in data.get('appraisals', []):
            # Get properties for this appraisal
            appraisal_properties = appraisal.get('properties', [])

            for prop in appraisal_properties:
                # Create a copy of the property data
                prop_data: Dict[str, Any] = prop.copy()

                # Add orderID as an identifier
                prop_data['orderID'] = appraisal.get('orderID')
                properties.append(prop_data)

                self.logger.debug(
                    f"Processed property with orderID: {prop_data.get('orderID')}")

        return properties

    def __clean_properties(self, properties: List[Dict[str, Any]]) -> pl.DataFrame:
        """Clean and standardize property data.

        Processes each property dictionary to:
        1. Clean and standardize string fields
        2. Parse and convert numeric fields
        3. Handle date fields
        4. Convert to a Polars DataFrame
        5. Derive additional features
        6. Handle null values

        Args:
            properties (List[Dict[str, Any]]): List of property dictionaries from appraisals.

        Returns:
            pl.DataFrame: Cleaned and standardized property data in a Polars DataFrame.
        """
        cleaned_properties: List[Dict[str, Any]] = []
        for prop in properties:
            cleaned: Dict[str, Any] = {}

            # Basic identifiers
            cleaned['id'] = prop.get('id')
            cleaned['orderID'] = prop.get('orderID')

            # Location information
            cleaned['address'] = prop.get('address')
            cleaned['city'] = prop.get('city')
            cleaned['province'] = prop.get('province')
            cleaned['postal_code'] = prop.get('postal_code')
            cleaned['latitude'] = clean_numeric_string(prop.get('latitude'))
            cleaned['longitude'] = clean_numeric_string(prop.get('longitude'))

            # Property characteristics
            cleaned['property_sub_type'] = prop.get('property_sub_type')
            cleaned['structure_type'] = prop.get('structure_type')
            cleaned['style'] = prop.get('style')
            cleaned['levels'] = prop.get('levels')

            # Numeric features
            cleaned['bedrooms'] = parse_bedrooms(prop.get('bedrooms'))
            cleaned['full_baths'] = clean_numeric_string(
                prop.get('full_baths'))
            cleaned['half_baths'] = clean_numeric_string(
                prop.get('half_baths'))
            cleaned['room_count'] = clean_numeric_string(
                prop.get('room_count'))
            cleaned['gla'] = clean_numeric_string(prop.get('gla'))
            cleaned['main_level_finished_area'] = clean_numeric_string(
                prop.get('main_level_finished_area'))
            cleaned['upper_lvl_fin_area'] = clean_numeric_string(
                prop.get('upper_lvl_fin_area'))
            cleaned['lot_size_sf'] = clean_numeric_string(
                prop.get('lot_size_sf'))
            cleaned['year_built'] = parse_year_built(prop.get('year_built'))

            # Additional features
            cleaned['roof'] = prop.get('roof')
            cleaned['basement'] = prop.get('basement')
            cleaned['cooling'] = prop.get('cooling')
            cleaned['heating'] = prop.get('heating')

            # Sale information
            cleaned['close_price'] = parse_sale_price(prop.get('close_price'))
            cleaned['close_date'] = parse_date(prop.get('close_date'))

            # Text fields
            cleaned['public_remarks'] = prop.get('public_remarks')

            # Add to list
            cleaned_properties.append(cleaned)

        # Convert to DataFrame and derive additional features
        df: pl.DataFrame = pl.DataFrame(
            cleaned_properties, infer_schema_length=1000)
        df = self.__derive_additional_features(df)
        df = self.__handle_nulls(df)
        return df

    def __derive_additional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Derive additional features from existing data.

        Creates new features based on existing data:
        1. Calculates total bathrooms
        2. Calculates price per square foot
        3. Creates property_class based on condition and GLA
        4. Calculates time-based features from close_date
        5. Extracts lot size unit

        Args:
            df (pl.DataFrame): Input DataFrame with cleaned property data.

        Returns:
            pl.DataFrame: DataFrame with additional derived features.
        """
        # Calculate total bathrooms
        if 'full_baths' in df.columns and 'half_baths' in df.columns:
            df = df.with_columns([
                (pl.col('full_baths') + (pl.col('half_baths') * 0.5)).alias('total_baths')
            ])

        # Calculate price per square foot
        if 'close_price' in df.columns and 'gla' in df.columns:
            df = df.with_columns([
                (pl.col('close_price') / pl.col('gla')).alias('price_per_sqft')
            ])

        # Create property_class feature based on GLA and price
        if 'gla' in df.columns and 'close_price' in df.columns:
            df = df.with_columns([
                pl.when(
                    (pl.col('gla') > 3000) & (pl.col('close_price') > 1000000)
                ).then(pl.lit('Luxury'))
                .when(
                    (pl.col('gla') > 2500) | (pl.col('close_price') > 750000)
                ).then(pl.lit('Premium'))
                .when(
                    (pl.col('gla') > 1500) | (pl.col('close_price') > 500000)
                ).then(pl.lit('Standard'))
                .otherwise(pl.lit('Basic'))
                .alias('property_class')
            ])

        # Calculate time-based features
        if 'close_date' in df.columns:
            df = df.with_columns([
                pl.col('close_date').dt.year().alias('sale_year'),
                pl.col('close_date').dt.month().alias('sale_month'),
                pl.col('close_date').dt.quarter().alias('sale_quarter')
            ])

        # Extract lot size unit
        if 'lot_size_sf' in df.columns:
            df = df.with_columns([
                pl.when(pl.col('lot_size_sf') < 1)
                .then(pl.lit('Acre'))
                .when(pl.col('lot_size_sf') > 10000)
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
            'bedrooms', 'full_baths', 'half_baths', 'room_count', 'gla',
            'main_level_finished_area', 'upper_lvl_fin_area', 'lot_size_sf',
            'year_built', 'close_price', 'price_per_sqft', 'total_baths',
            'latitude', 'longitude'
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
            'property_sub_type', 'structure_type', 'style', 'levels',
            'roof', 'basement', 'cooling', 'heating', 'property_class',
            'lot_size_unit', 'orderID'
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
        date_cols: List[str] = ['close_date']
        for col in date_cols:
            if col in df.columns:
                # Get most recent date
                max_date: Optional[datetime] = df[col].max()
                if max_date is not None:
                    df = df.with_columns([
                        pl.col(col).fill_null(max_date).alias(col)
                    ])

        return df
