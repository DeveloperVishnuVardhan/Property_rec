import json
import pandas as pd
from pprint import pprint
from typing import Any, Dict, List
import textwrap
import polars as pl


def analyze_polars_dataframe(df: pl.DataFrame, display_limit_unique: int = 15, show_all_value_counts_limit: int = 50, top_n_value_counts: int = 5):
    """
    Performs and prints a basic analysis of each column in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
        display_limit_unique (int): Max number of unique values to list directly.
                                    If more, a sample is shown.
        show_all_value_counts_limit (int): If unique values are below or equal to this for a
                                           categorical/string column, all value counts are shown.
        top_n_value_counts (int): If unique values exceed show_all_value_counts_limit for
                                  categorical/string columns, this many top value counts are shown.
    """
    if not isinstance(df, pl.DataFrame):
        print("Error: Input is not a Polars DataFrame.")
        return

    if df.is_empty():
        print("The DataFrame is empty. No analysis to perform.")
        return

    total_rows = df.height
    print(f"DataFrame Overview: {total_rows} rows, {df.width} columns\n")
    print("="*70)

    for col_name in df.columns:
        print(f"\n--- Analysis for Column: '{col_name}' ---")
        column_series = df[col_name]

        # Data Type
        print(f"Data Type: {column_series.dtype}")

        # Null Values
        null_count = column_series.null_count()
        non_null_count = total_rows - null_count
        null_percentage = (null_count / total_rows) * \
            100 if total_rows > 0 else 0
        print(f"Total Values (rows): {total_rows}")
        print(f"Non-Null Values: {non_null_count}")
        print(f"Null Values: {null_count} ({null_percentage:.2f}%)")

        # Unique Values
        num_unique_values = column_series.n_unique()  # Includes null if present
        print(
            f"Unique Values Count (includes null if present): {num_unique_values}")

        if num_unique_values > 0:
            # Get unique values, sort them. Sorting might fail for mixed types if not careful,
            # but generally okay for single-type columns or if Polars can handle it.
            try:
                unique_items = column_series.unique().sort()
                if num_unique_values <= display_limit_unique:
                    print(f"Unique Values List: {unique_items.to_list()}")
                else:
                    print(
                        f"Unique Values List (sample of first {display_limit_unique}): {unique_items.head(display_limit_unique).to_list()}")
                    # Check if None was in the head sample if nulls exist
                    if unique_items.null_count() > 0 and (None not in unique_items.head(display_limit_unique).drop_nulls().to_list() if unique_items.head(display_limit_unique).null_count() == 0 else None not in unique_items.head(display_limit_unique).to_list()):
                        print(
                            "Note: Null is also one of the unique values if present in the column and not shown in sample.")
            except Exception as e:
                print(
                    f"Could not display unique values list (possibly due to mixed types or other error): {e}")
        else:
            print(
                "Unique Values List: Column might be effectively empty or no distinct values.")
        print("\n" + "-"*50)
    print("\n" + "="*70)
    print("End of DataFrame Analysis.")


def print_keys_formatted(keys: List[str], indent: int = 0) -> None:
    """Print keys in a formatted way, 10 per line."""
    keys_str = ", ".join(keys)
    wrapped = textwrap.wrap(
        keys_str, width=80, initial_indent=" " * indent, subsequent_indent=" " * indent)
    for line in wrapped:
        print(line)


def explore_value_structure(value: Any, indent: int = 0, parent_key: str = "") -> None:
    """Recursively explore and print structure of any value type."""
    indent_str = "│   " * (indent // 4) + "    " * (indent % 4)

    if isinstance(value, dict):
        print(f"{indent_str}└── {parent_key} (Dictionary)")
        print(f"{indent_str}    Number of keys: {len(value)}")
        if value:
            print(f"{indent_str}    Keys:")
            print_keys_formatted(list(value.keys()), indent + 8)
            print(f"{indent_str}    Structure of each key:")
            for key, sub_value in value.items():
                explore_value_structure(sub_value, indent + 8, key)
    elif isinstance(value, list):
        print(f"{indent_str}└── {parent_key} (List)")
        print(f"{indent_str}    Length: {len(value)}")
        if value:
            print(f"{indent_str}    First item structure:")
            explore_value_structure(value[0], indent + 8, "First Item")
            if len(value) > 1:
                print(
                    f"{indent_str}    Note: {len(value)-1} more items with similar structure")
    else:
        print(f"{indent_str}└── {parent_key}")
        print(f"{indent_str}    Type: {type(value).__name__}")
        if isinstance(value, (str, int, float, bool)):
            print(f"{indent_str}    Value: {value}")


def explore_json_structure(file_path: str) -> Dict:
    """Load and explore JSON file structure."""
    print("=" * 80)
    print(f"Exploring JSON file: {file_path}")
    print("=" * 80)

    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Print the type of data and its length if it's a list
    print(f"\nRoot level:")
    explore_value_structure(data, 0, "Root")

    return data


if __name__ == "__main__":
    file_path = "appraisals_dataset.json"
    data = explore_json_structure(file_path)
    print(f"Number of appraisals: {len(data['appraisals'])}")
    print(type(data['appraisals']))
    print(f"Keys in the first appraisal: {data['appraisals'][0].keys()}")
    print(f"Keys in the second appraisal: {data['appraisals'][0].keys()}")
    print(f"************************************************")
    print(type(data['appraisals'][0]['subject']))
    print(
        f"Keys in the subject of the first appraisal: {list(data['appraisals'][0]['subject'].keys())}")
    print(f"************************************************")
    print(type(data['appraisals'][0]['comps']))
    print(
        f"Keys in the comps of the first appraisal: {list(data['appraisals'][0]['comps'][0].keys())}")
    print(f"************************************************")
    print(type(data['appraisals'][0]['properties']))
    print(
        f"Keys in the properties of the first appraisal: {list(data['appraisals'][0]['properties'][0].keys())}")
