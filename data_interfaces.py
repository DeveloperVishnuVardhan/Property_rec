from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import polars as pl


class CleanData(ABC):
    """Abstract base class for subject data preparation.

    This class defines the interface for preparing subject data from appraisal records.
    Implementations should handle the extraction and cleaning of subject property data.
    """
    @abstractmethod
    def prep(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Prepare subject data from raw appraisal data.

        Args:
            data (Dict[str, Any]): Raw appraisal data containing subject information.

        Returns:
            pl.DataFrame: Cleaned and standardized subject data in a Polars DataFrame.
        """
        pass
