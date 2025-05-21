import polars as pl
from geopy.distance import geodesic
from rapidfuzz import fuzz


def standardize_lot_size_to_sqft(df, lot_size_column, unit_column):
    """
    Standardizes lot size values to square feet based on the unit measurement.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe containing the lot size and unit columns
    lot_size_column : str
        The name of the column containing lot size values
    unit_column : str
        The name of the column containing unit measurements (SqFt, SqM, Acre, etc.)

    Returns:
    --------
    pl.DataFrame
        The modified dataframe with standardized lot size values in square feet
    """
    # Make a copy to avoid modifying the original
    result_df = df.clone()

    # Ensure the unit column has no nulls
    result_df = result_df.with_columns(pl.col(unit_column).fill_null("N/A"))

    # Create a new column with the converted values
    result_df = result_df.with_columns(
        pl.when(pl.col(unit_column) == "SqM")
        .then(pl.col(lot_size_column) * 10.764)  # 1 sq meter = 10.764 sq feet
        .when(pl.col(unit_column) == "Acre")
        .then(pl.col(lot_size_column) * 43560)   # 1 acre = 43,560 sq feet
        # Keep original for SqFt or N/A
        .otherwise(pl.col(lot_size_column))
        .alias(lot_size_column)                  # Replace the original column
    )

    return result_df


class RankingData:
    def __init__(self, subjects_df: pl.DataFrame, comps_df: pl.DataFrame, property_df: pl.DataFrame):
        self.subjects_df = subjects_df
        self.comps_df = comps_df
        self.property_df = property_df

    def create_ranking_data(self, type: str = "train") -> pl.DataFrame:
        filtered_subjects_df = self.subjects_df.filter(
            pl.col("calculated_latitude").is_not_null())

        if type == "train":
            filtered_subjects_df = filtered_subjects_df[:56]
        elif type == "test":
            filtered_subjects_df = filtered_subjects_df[56:]
        else:
            raise ValueError("Invalid type")

        filtered_comps_df = self.comps_df.filter(
            pl.col("orderID").is_in(filtered_subjects_df["orderID"].unique()))
        # Remove duplicates from comps_df
        filtered_comps_df = filtered_comps_df.unique(
            subset=["orderID", "normalized_address"], keep="first", maintain_order=True)
        filtered_property_df = self.property_df.filter(
            pl.col("orderID").is_in(filtered_subjects_df["orderID"].unique()))
        # Remove duplicates from property_df
        filtered_property_df = filtered_property_df.unique(
            subset=["orderID", "normalized_address", "latitude", "longitude"], keep="first", maintain_order=True)

        # create cooling, heating in filtered_comps_df
        filtered_comps_df = filtered_comps_df.join(
            filtered_property_df.select(
                ["orderID", "normalized_address", "cooling", "heating"]),
            on=["orderID", "normalized_address"],
            how="left"
        ).with_columns(
            pl.col("cooling").fill_null("not found").alias("cooling"),
            pl.col("heating").fill_null("not found").alias("heating"),
        )

        # calculate age field for subjects and properties.
        filtered_subjects_df = filtered_subjects_df.with_columns(
            age=2025 - pl.col("year_built")
        )
        filtered_property_df = filtered_property_df.with_columns(
            age=2025 - pl.col("year_built")
        )

        # standardize unit measuremments.
        filtered_subjects_df = filtered_subjects_df.with_columns(
            pl.col("unit_measurement").fill_null("N/A"))
        filtered_subjects_df = filtered_subjects_df.with_columns({
            "unit_measurement": pl.when(pl.col("unit_measurement") == "SqM").then("SqM")
            .when(pl.col("unit_measurement") == "SqFt").then("SqFt")
            .when(pl.col("unit_measurement") == "Acres").then("Acre")
            .otherwise(pl.col("unit_measurement"))
        })
        filtered_comps_df = filtered_comps_df.with_columns({
            "lot_size_unit": pl.when(pl.col("lot_size_unit") == "SqM").then("SqM")
            .when(pl.col("lot_size_unit") == "SqFt").then("SqFt")
            .when(pl.col("lot_size_unit") == "Acre").then("Acre")
            .otherwise(pl.col("lot_size_unit"))
        })
        filtered_property_df = filtered_property_df.with_columns({
            "lot_size_unit": pl.when(pl.col("lot_size_unit") == "SqM").then("SqM")
            .when(pl.col("lot_size_unit") == "SqFt").then("SqFt")
            .when(pl.col("lot_size_unit") == "Acre").then("Acre")
            .otherwise(pl.col("lot_size_unit"))
        })

        # convert to sqft.
        filtered_subjects_df = standardize_lot_size_to_sqft(
            filtered_subjects_df,
            lot_size_column="lot_size_sf",
            unit_column="unit_measurement"
        )

        filtered_comps_df = standardize_lot_size_to_sqft(
            filtered_comps_df,
            lot_size_column="lot_size",
            unit_column="lot_size_unit"
        )

        filtered_property_df = standardize_lot_size_to_sqft(
            filtered_property_df,
            lot_size_column="lot_size_sf",
            unit_column="lot_size_unit"
        )

        # Prepare subject table.
        subject = (
            filtered_subjects_df.select(
                [
                    "orderID",
                    "style",
                    "heating",
                    "cooling",
                    "lot_size_sf",
                    "age",
                    "room_count",
                    "num_beds",
                    "gla",
                    "num_baths",
                    "property_class",
                    "calculated_latitude",
                    "calculated_longitude",
                ]
            )
            .rename({
                "style": "style_sub",
                "heating": "heating_sub",
                "cooling": "cooling_sub",
                "lot_size_sf": "lot_size_sf_sub",
                "age": "age_sub",
                "room_count": "room_count_sub",
                "num_beds": "num_beds_sub",
                "gla": "gla_sub",
                "num_baths": "num_baths_sub",
                "property_class": "property_class_sub",
                "calculated_latitude": "latitude_sub",
                "calculated_longitude": "longitude_sub",
            })
        )

        # Build positives.
        positives = (
            filtered_comps_df.select(
                [
                    "orderID",
                    "stories",
                    "heating",
                    "cooling",
                    "lot_size",
                    "age",
                    "room_count",
                    "bed_count",
                    "gla",
                    "bath_count",
                    "property_class",
                    "calculated_latitude",
                    "calculated_longitude",
                ]
            )
            .rename({
                "stories": "style_cand",
                "heating": "heating_cand",
                "cooling": "cooling_cand",
                "lot_size": "lot_size_sf_cand",
                "age": "age_cand",
                "room_count": "room_count_cand",
                "bed_count": "num_beds_cand",
                "gla": "gla_cand",
                "bath_count": "num_baths_cand",
                "property_class": "property_class_cand",
                "calculated_latitude": "latitude_cand",
                "calculated_longitude": "longitude_cand"
            }).join(subject, on="orderID").with_columns(
                pl.lit(1).alias("label")
            )
        )

        # Build negatives.
        neg_list = []
        N_NEG = 10
        for order_id in subject["orderID"].unique().to_list():
            subject_row = subject.filter(pl.col("orderID") == order_id)
            pool = (
                filtered_property_df.filter(pl.col("orderID") == order_id)
                .select([
                    "orderID",
                    "normalized_address",  # Keep this for filtering
                    "style",
                    "heating",
                    "cooling",
                    "lot_size_sf",
                    "age",
                    "room_count",
                    "bedrooms",
                    "gla",
                    "total_baths",
                    "property_class",
                    "latitude",
                    "longitude"
                ])
                .rename({
                    "style": "style_cand",
                    "heating": "heating_cand",
                    "cooling": "cooling_cand",
                    "lot_size_sf": "lot_size_sf_cand",
                    "age": "age_cand",
                    "room_count": "room_count_cand",
                    "bedrooms": "num_beds_cand",
                    "gla": "gla_cand",
                    "total_baths": "num_baths_cand",
                    "property_class": "property_class_cand",
                    "latitude": "latitude_cand",
                    "longitude": "longitude_cand",
                })
            )
            choosen = filtered_comps_df.filter(
                pl.col("orderID") == order_id).get_column("normalized_address").unique()
            pool = pool.filter(~pl.col("normalized_address").is_in(choosen))
            sample_size = min(N_NEG, pool.height)
            sampled = pool.sample(sample_size, seed=42)
            neg_list.append(sampled.join(subject_row, on="orderID").with_columns(
                pl.lit(0).alias("label")))

        negatives = pl.concat(neg_list).drop(["normalized_address"])
        df = pl.concat([positives, negatives])

        numerical_ranking = df.with_columns(
            pl.struct(["latitude_sub", "longitude_sub", "latitude_cand", "longitude_cand"]).
            map_elements(lambda r: geodesic((r["latitude_sub"], r["longitude_sub"]),
                                            (r["latitude_cand"], r["longitude_cand"])).km).alias("dist_km"),
            (pl.col("room_count_sub").abs() -
             pl.col("room_count_cand").abs()).alias("room_diff"),
            (pl.col("num_beds_sub").abs() - pl.col("num_beds_cand").abs()
             ).alias("bed_diff"),  # Changed from bed_count_sub
            (pl.col("num_baths_sub") - pl.col("num_baths_cand").abs()
             ).alias("bath_diff"),  # Changed from bath_count_sub
            (pl.col("lot_size_sf_sub").abs() -
             pl.col("lot_size_sf_cand").abs()).alias("lot_diff"),
            (pl.col("age_sub").abs() - pl.col("age_cand").abs()).alias("age_diff"),
            (pl.col("gla_sub").abs() - pl.col("gla_cand").abs()).alias("gla_diff"),
        )

        # Define the features separately
        categorical_features = [
            ("style_sub", "style_cand", "style_sim"),
            ("heating_sub", "heating_cand", "heating_sim"),
            ("cooling_sub", "cooling_cand", "cooling_sim"),
            ("property_class_sub", "property_class_cand", "property_class_sim")
        ]

        # Add categorical features one by one
        final_df = numerical_ranking.clone()

        for col1, col2, output_name in categorical_features:
            try:
                # Create a new column using a different approach
                final_df = final_df.with_columns([
                    pl.struct([pl.col(col1), pl.col(col2)])
                    .map_elements(
                        lambda row: fuzz.ratio(
                            str(row[col1]) if row[col1] is not None else "",
                            str(row[col2]) if row[col2] is not None else ""
                        ) / 100.0,
                        return_dtype=pl.Float64
                    )
                    .alias(output_name)
                ])
                print(f"Successfully added {output_name}")
            except Exception as e:
                print(f"Error adding {output_name}: {e}")

        return final_df
