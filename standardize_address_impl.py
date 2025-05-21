from datetime import time
from typing import Dict, List, Optional, Tuple, Union, Any
import polars as pl
import re

from logger_config import setup_logger
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.location import Location


class StandardizeAddress:
    """Standardize address data.

    This class implements the StandardizeData interface to standardize address data.
    """

    def __init__(self, subject_df: pl.DataFrame, comps_df: pl.DataFrame, properties_df: pl.DataFrame) -> None:
        self.subject_df: pl.DataFrame = subject_df
        self.comps_df: pl.DataFrame = comps_df
        self.properties_df: pl.DataFrame = properties_df
        self.logger = setup_logger(__name__)

    def standardize(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Standardize address data.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing standardized subject, comps, and property dataframes.
        """
        # Create working copies
        subject_df: pl.DataFrame = self.subject_df.clone()
        comps_df: pl.DataFrame = self.comps_df.clone()
        property_df: pl.DataFrame = self.properties_df.clone()

        # Add full address column to each dataframe
        subject_df = self.__add_full_address_column(subject_df, 'subject')
        comps_df = self.__add_full_address_column(comps_df, 'comps')
        property_df = self.__add_full_address_column(property_df, 'property')

        # setup Nominatim geocoder with rate limiting
        self.logger.info("Setting up Nominatim geocoder with rate limiting")
        geolocator = Nominatim(user_agent="AutoMax_Property_Geocoder/1.0")
        rate_limiter_geocode = RateLimiter(
            geolocator.geocode, min_delay_seconds=1)
        rate_limiter_reverse = RateLimiter(
            geolocator.reverse, min_delay_seconds=1)

        # create a cache of addresses
        address_cache = {}
        self.logger.info(
            "Processing properties dataframe first to build cache...")
        # Add columns for normalized address and coordinates if they don't exist (subject, comps. only)
        for df in [subject_df, comps_df]:
            if "normalized_address" not in df.columns:
                df = df.with_columns(pl.lit(None).alias("normalized_address"))
            if "calculated_latitude" not in df.columns:
                df = df.with_columns(pl.lit(None).alias("calculated_latitude"))
            if "calculated_longitude" not in df.columns:
                df = df.with_columns(
                    pl.lit(None).alias("calculated_longitude"))

        # add normalized address if not exist in properties.
        if "normalized_address" not in property_df.columns:
            property_df = property_df.with_columns(
                pl.lit(None).alias("normalized_address"))

        # Process dataframes in sequence
        property_df = self.__normalize_property_df(
            property_df, rate_limiter_reverse, address_cache)
        subject_df = self.__compute_normalization(
            subject_df, "subject", rate_limiter_geocode, address_cache)
        comps_df = self.__compute_normalization(
            comps_df, "comps", rate_limiter_geocode, address_cache)

        self.logger.info("standardization complete!")
        return subject_df, comps_df, property_df

    def __normalize_property_df(self, df: pl.DataFrame, geocode: RateLimiter, address_cache: Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]) -> pl.DataFrame:
        """Normalize addresses for property dataframe using reverse geocoding.

        Args:
            df (pl.DataFrame): Property dataframe
            geocode (RateLimiter): Rate-limited geocoder
            address_cache (Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]): Cache for geocoding results

        Returns:
            pl.DataFrame: Normalized property dataframe
        """
        total = df.height
        processed = 0
        result_rows = []

        self.logger.info(f"Processing property dataframe {total} rows")
        rows = df.to_dicts()

        for (i, row) in enumerate(rows):
            new_row = row.copy()
            reverse_geocode_success = False

            self.logger.info(f"Processing row {i} of {total}")
            # Try reverse geocoding first if we have coordinates
            if row.get("latitude") is not None and row.get("longitude") is not None:
                try:
                    location = geocode(
                        f"{row['latitude']}, {row['longitude']}")
                    if location:
                        new_row["normalized_address"] = location.address
                        new_row["latitude"] = row["latitude"]
                        new_row["longitude"] = row["longitude"]
                        result_rows.append(new_row)
                        processed += 1
                        reverse_geocode_success = True
                        continue
                except Exception as e:
                    self.logger.warning(
                        f"Reverse geocoding failed for coordinates {row['latitude']}, {row['longitude']}: {str(e)}")

            # Only do forward geocoding if reverse geocoding failed
            if not reverse_geocode_success:
                full_addr = row.get("full_address", "")
                norm_addr, lat, lng = self.__geocode_with_nominatim(
                    full_addr, geocode, address_cache)

                new_row["normalized_address"] = norm_addr if norm_addr else full_addr
                new_row["latitude"] = row["latitude"]
                new_row["longitude"] = row["longitude"]

                result_rows.append(new_row)
                processed += 1

            if processed % 100 == 0 or processed == total:
                self.logger.info(
                    f"Processed {processed}/{total} ({processed/total*100:.1f}%)")

        # Create new dataframe from processed rows
        new_df = pl.DataFrame(result_rows)

        # Report completion
        success_count = new_df.filter(
            (pl.col("normalized_address").is_not_null()) &
            (pl.col("latitude").is_not_null()) &
            (pl.col("longitude").is_not_null())
        ).height
        success_rate = success_count / total * 100 if total > 0 else 0
        self.logger.info(
            f"Completed property processing with {success_rate:.1f}% success rate")

        return new_df

    def __compute_normalization(self, df: pl.DataFrame, name: str, geocode: RateLimiter, address_cache: Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]) -> pl.DataFrame:
        """Normalize addresses for subject and comps dataframes.

        Args:
            df (pl.DataFrame): Input dataframe
            name (str): Name of the dataframe ('subject' or 'comps')
            geocode (RateLimiter): Rate-limited geocoder
            address_cache (Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]): Cache for geocoding results

        Returns:
            pl.DataFrame: Normalized dataframe
        """
        total = df.height
        processed = 0

        self.logger.info(f"Processing {name} dataframe {total} rows")
        rows = df.to_dicts()
        result_rows = []
        for (i, row) in enumerate(rows):
            new_row = row.copy()
            # skip if already has full data
            if (row.get("normalized_address") is not None and
                row.get("calculated_latitude") is not None and
                    row.get("calculated_longitude") is not None):
                result_rows.append(new_row)
                processed += 1
                continue

            # Get full address
            full_addr = row.get("full_address", "")
            # Geocode and get normalized address
            self.logger.info(f"Geocoding row {i} of {total}")
            norm_addr, lat, lng = self.__geocode_with_nominatim(
                full_addr, geocode, address_cache)
            self.logger.info(
                f"Normalized address: {norm_addr} - latitude: {lat} - long: {lng}")

            # Update row with results
            if norm_addr is not None:
                new_row["normalized_address"] = norm_addr
            else:
                new_row["normalized_address"] = full_addr
            if lat is not None and lng is not None:
                new_row["calculated_latitude"] = lat
                new_row["calculated_longitude"] = lng
            else:
                new_row["calculated_latitude"] = None
                new_row["calculated_longitude"] = None

            result_rows.append(new_row)
            processed += 1

            if processed % 100 == 0 or processed == total:
                print(
                    f"Processed {processed}/{total} ({processed/total*100:.1f}%)")

        # Create new dataframe from processed rows
        new_df = pl.DataFrame(result_rows)

        # Report completion
        success_count = new_df.filter(
            (pl.col("normalized_address").is_not_null()) &
            (pl.col("calculated_latitude").is_not_null()) &
            (pl.col("calculated_longitude").is_not_null())
        ).height
        success_rate = success_count / total * 100 if total > 0 else 0
        print(
            f"Completed {name} processing with {success_rate:.1f}% success rate")

        return new_df

    def __geocode_with_nominatim(self, full_addr: str, geocode: RateLimiter, address_cache: Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Geocode an address with Nominatim.

        Args:
            full_addr (str): The full address to geocode.
            geocode (RateLimiter): The geocoder to use.
            address_cache (Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]): The cache of addresses.

        Returns:
            Tuple[Optional[str], Optional[float], Optional[float]]: The normalized address, latitude, and longitude.
        """
        if not full_addr or full_addr == "":
            return None, None, None

        # Check cache first
        if full_addr in address_cache:
            return address_cache[full_addr]

        try:
            location = geocode(full_addr)
            # self.logger.info(f"Geocoded address: {location}")
            if location:
                normalized_address = location.address
                latitude = location.latitude
                longitude = location.longitude
                if address_cache is not None:
                    address_cache[full_addr] = (
                        normalized_address, latitude, longitude)
                return normalized_address, latitude, longitude

            # If no location found, cache this result too
            if address_cache is not None:
                address_cache[full_addr] = (None, None, None)
            return None, None, None

        except Exception as e:
            # Any errors - just return None and cache the failure
            if address_cache is not None:
                address_cache[full_addr] = (None, None, None)
            return None, None, None

    def __add_full_address_column(self, df: pl.DataFrame, df_type: str) -> pl.DataFrame:
        """Add a full address column to a dataframe.

        Args:
            df (pl.DataFrame): The dataframe to add the full address column to.
            df_type (str): The type of dataframe to add the full address column to.

        Returns:
            pl.DataFrame: The dataframe with the full address column added.
        """
        rows = df.to_dicts()
        full_addresses = []

        for row in rows:
            full_addresses.append(self.__create_address_string(row, df_type))

        return df.with_columns(pl.Series(name='full_address', values=full_addresses))

    def __create_address_string(self, row: Dict[str, Any], df_type: str) -> str:
        """
        Create a standardized full address string based on dataframe type.

        Args:
            row (Dict[str, Any]): Dictionary containing row data
            df_type (str): Type of dataframe ('subject', 'comps', or 'property')

        Returns:
            str: Standardized address string
        """
        try:
            if df_type == 'subject':
                address = f"{row.get('address', '')}"

            elif df_type == 'comps':
                location = row.get('city_province', '') if row.get(
                    'city_province') not in [None, ''] else ''

                address = f"{row.get('address', '')}, {location}"

            elif df_type == 'property':
                city = row.get('city', '') if row.get(
                    'city') not in [None, ''] else ''
                province = row.get('province', '') if row.get(
                    'province') not in [None, ''] else ''
                postal_code = row.get('postal_code', '') if row.get(
                    'postal_code') not in [None, ''] else ''
                location = f"{city}, {province} {postal_code}".strip()
                location = re.sub(r'\s+', ' ', location)
                address = f"{row.get('address', '')}, {location}"

            # Clean up the address string - remove any double commas from empty fields
            address = re.sub(r',\s*,', ',', address)
            return address.strip().rstrip(',')
        except:
            return row.get('address', '') if row.get('address') not in [None, ''] else ''
