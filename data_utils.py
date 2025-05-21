from typing import Any, Optional, List, Union, Pattern, Match
import re
from datetime import datetime


def clean_numeric_string(value: Any) -> Optional[float]:
    """Convert string numeric values to float, preserving original values.

    Processes numeric strings by:
    1. Handling year values (4 digits)
    2. Extracting numeric values from strings
    3. Preserving original values without unit conversions

    Args:
        value (Any): Input value to be converted to float.

    Returns:
        Optional[float]: Converted float value, or None if conversion fails.
    """
    if value is None or value == '':
        return None

    # If already a number, return it
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    # Handle year values (4 digits)
    if re.match(r'^(18|19|20)\d{2}$', value):
        try:
            year: int = int(value)
            if year <= 2025:
                return float(year)
        except ValueError:
            pass

    # Extract numeric part from string
    # Keep decimal points, commas, and minus signs
    clean_value: str = re.sub(r'[^\d.,\-]', '', value)
    clean_value = clean_value.replace(',', '')

    try:
        return float(clean_value)
    except ValueError:
        return None


def parse_bathrooms(value: Any) -> Optional[float]:
    """Parse bathroom notations into total number of bathrooms.

    Handles various bathroom notation formats:
    1. "2:1" (2 full, 1 half bath)
    2. "2F 1H" (2 full, 1 half bath)
    3. "2 Full/1Half" (2 full, 1 half bath)
    4. Simple numeric values

    Half baths are counted as 0.5 in the total.

    Args:
        value (Any): Input value containing bathroom information.

    Returns:
        Optional[float]: Total number of bathrooms, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Handle "1:1" notation (1 full bath, 1 half bath)
    if ':' in value:
        parts: List[str] = value.split(':')
        try:
            full: float = float(parts[0]) if parts[0] else 0
            half: float = float(parts[1]) * \
                0.5 if len(parts) > 1 and parts[1] else 0
            return full + half
        except ValueError:
            pass

    # Handle "2F 1H" notation
    f_match: Optional[Match[str]] = re.search(r'(\d+)F', value)
    h_match: Optional[Match[str]] = re.search(r'(\d+)H', value)

    if f_match and h_match:
        try:
            full: float = float(f_match.group(1))
            half: float = float(h_match.group(1)) * 0.5
            return full + half
        except ValueError:
            pass

    # Handle "2 Full/1Half" notation
    full_match: Optional[Match[str]] = re.search(r'(\d+)\s*Full', value)
    half_match: Optional[Match[str]] = re.search(r'(\d+)\s*Half', value)

    if full_match and half_match:
        try:
            full: float = float(full_match.group(1))
            half: float = float(half_match.group(1)) * 0.5
            return full + half
        except ValueError:
            pass

    # Try direct conversion
    try:
        return float(value)
    except ValueError:
        return None


def parse_bedrooms(value: Any) -> Optional[float]:
    """Parse bedroom notations into total number of bedrooms.

    Handles various bedroom notation formats:
    1. "3+1" (3 main + 1 basement bedroom)
    2. Simple numeric values

    Args:
        value (Any): Input value containing bedroom information.

    Returns:
        Optional[float]: Total number of bedrooms, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Handle "3+1" notation
    if '+' in value:
        parts: List[str] = value.split('+')
        try:
            return float(parts[0])
        except ValueError:
            pass

    # Try direct conversion
    try:
        return float(value)
    except ValueError:
        return None


def parse_lot_size(value: Any) -> Optional[float]:
    """Extract numeric values from lot size strings.

    Processes lot size values by:
    1. Removing all non-numeric characters except decimal points and minus signs
    2. Converting to float

    Args:
        value (Any): Input value containing lot size information.

    Returns:
        Optional[float]: Numeric lot size value, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Remove any non-numeric characters except decimal point and minus sign
    clean_value: str = re.sub(r'[^\d.\-]', '', value)

    try:
        return float(clean_value)
    except ValueError:
        return None


def parse_date(value: Any) -> Optional[datetime]:
    """Convert various date formats to datetime objects.

    Handles multiple date formats:
    1. 'MMM/DD/YYYY' (e.g., 'Jan/01/2023')
    2. 'YYYY-MM-DD' (e.g., '2023-01-01')
    3. 'MM/DD/YYYY' (e.g., '01/01/2023')

    Args:
        value (Any): Input value containing date information.

    Returns:
        Optional[datetime]: Datetime object, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if isinstance(value, datetime):
        return value

    if not isinstance(value, str):
        return None

    # Try various formats
    formats: List[str] = ['%b/%d/%Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_age(value: Any) -> Optional[int]:
    """Parse age values, handling both direct age and year built.

    Processes age values by:
    1. Converting year built to age (relative to 2025)
    2. Handling direct age values
    3. Applying sanity checks for reasonable age values (0-200)

    Args:
        value (Any): Input value containing age or year built information.

    Returns:
        Optional[int]: Calculated age in years, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # Check if it's a year (e.g., 1950, 2010)
    year_match: Optional[Match[str]] = re.match(r'^(19|20)\d{2}$', value)
    if year_match:
        try:
            year: int = int(value)
            return 2025 - year  # Calculate age from year built
        except ValueError:
            pass

    # Remove non-numeric characters
    clean_value: str = re.sub(r'[^\d]', '', value)

    try:
        age: int = int(clean_value)
        # Sanity check for reasonable age values
        return age
    except ValueError:
        return None


def parse_distance(value: Any) -> Optional[float]:
    """Parse distance values from strings into kilometers.

    Handles various distance formats:
    1. "0.05 km"
    2. "0.00 KM"
    3. Simple numeric values

    Args:
        value (Any): Input value containing distance information.

    Returns:
        Optional[float]: Distance in kilometers, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Remove any non-numeric characters except decimal point and minus sign
    clean_value: str = re.sub(r'[^\d.\-]', '', value)

    try:
        return float(clean_value)
    except ValueError:
        return None


def parse_location_similarity(value: Any) -> Optional[str]:
    """Parse location similarity into standardized values.

    Standardizes location similarity values to:
    - "Superior"
    - "Similar"
    - "Inferior"

    Args:
        value (Any): Input value containing location similarity information.

    Returns:
        Optional[str]: Standardized location similarity value, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        return None

    # Standardize to title case
    value = value.title()

    # Map to standard values
    if value in ['Superior', 'Similar', 'Inferior']:
        return value

    return None


def parse_sale_price(value: Any) -> Optional[float]:
    """Parse sale price from string format.

    Handles various price formats:
    1. "1,005,000"
    2. Simple numeric values

    Args:
        value (Any): Input value containing sale price information.

    Returns:
        Optional[float]: Sale price as float, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Remove commas and any non-numeric characters except decimal point and minus sign
    clean_value: str = re.sub(r'[^\d.\-]', '', value)

    try:
        return float(clean_value)
    except ValueError:
        return None


def parse_dom(value: Any) -> Optional[int]:
    """Parse days on market (DOM) from string format.

    Handles various DOM formats:
    1. "10 +/-" or "10+/-" (exact value)
    2. "10+" (minimum value)
    3. "10-" (maximum value)
    4. Simple numeric values

    The function interprets the +/- signs as follows:
    - "+/-" indicates an exact value
    - "+" indicates a minimum value (at least this many days)
    - "-" indicates a maximum value (at most this many days)

    Args:
        value (Any): Input value containing DOM information.

    Returns:
        Optional[int]: Days on market as integer, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # Handle exact value with +/- notation
    if '+/-' in value or ' +/-' in value:
        clean_value: str = re.sub(r'[^\d]', '', value)
        try:
            return int(clean_value)
        except ValueError:
            return None

    # Handle minimum value with + notation
    if value.endswith('+'):
        clean_value: str = re.sub(r'[^\d]', '', value)
        try:
            return int(clean_value)
        except ValueError:
            return None

    # Handle maximum value with - notation
    if value.endswith('-'):
        clean_value: str = re.sub(r'[^\d]', '', value)
        try:
            return int(clean_value)
        except ValueError:
            return None

    # Handle simple numeric value
    clean_value: str = re.sub(r'[^\d]', '', value)
    try:
        return int(clean_value)
    except ValueError:
        return None


def parse_year_built(value: Any) -> Optional[int]:
    """Parse year built values, handling both direct years and years relative to 1880.

    This function handles two cases:
    1. Direct year values (e.g., 1950, 2010)
    2. Years relative to 1880 (e.g., 2, 8, 10, 13, 23, 40)
        - 2: 1880 + 2 = 1882
        - 8: 1880 + 8 = 1888
        - 10: 1880 + 10 = 1890
        - 13: 1880 + 13 = 1893
        - 23: 1880 + 23 = 1903
        - 40: 1880 + 40 = 1920

    Args:
        value (Any): Input value containing year built information.

    Returns:
        Optional[int]: Parsed year built value, or None if parsing fails.
    """
    if value is None or value == '':
        return None

    if not isinstance(value, str):
        try:
            year = int(value)
            # If the year is less than 100, assume it's relative to 1880
            if year < 100:
                return 1880 + year
            # If the year is between 1800 and current year + 1, use it directly
            current_year = datetime.now().year
            if 1800 <= year <= current_year + 1:
                return year
            return None
        except (ValueError, TypeError):
            return None

    # Remove any non-numeric characters
    clean_value: str = re.sub(r'[^\d]', '', value)

    try:
        year = int(clean_value)
        # If the year is less than 100, assume it's relative to 1880
        if year < 100:
            return 1880 + year
        # If the year is between 1800 and current year + 1, use it directly
        current_year = datetime.now().year
        if 1800 <= year <= current_year + 1:
            return year
        return None
    except ValueError:
        return None
