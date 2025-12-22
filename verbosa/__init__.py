"""
Verbosa - A streamlined data handling library.

A small library that streamlines everyday data handling tasks like validating
data types, enforcing integrity rules, and applying common data checks.
"""

__version__ = "0.1.0"

# Core data processing classes
from verbosa.data.normalizers.tabular_data import TabularDataNormalizer
from verbosa.data.reviewers.tabular_data import TabularDataReviewer
from verbosa.data.comparators.tabular_data import TabularDataComparator

# Data readers
from verbosa.data.readers.local import FileDataReader, FileSystemNavigator
from verbosa.data.readers.aws import AWSDataReader, AthenaDataReader

# Configuration interfaces
from verbosa.interfaces.column_config import ColumnConfig, CallSpec
from verbosa.interfaces.columns_config import ColumnsConfig
from verbosa.interfaces.aws import AWSCredentials, AthenaTableDetails

# Core interfaces
# from verbosa.interfaces.normalizer import NormalizerInterface
# from verbosa.interfaces.reviewer import ReviewerInterface
# from verbosa.interfaces.reader import ReaderInterface
# from verbosa.interfaces.cell import Cell

# Utilities
from verbosa.utils.logger_machine import LogsMachine
from verbosa.widgets.selection_menu import SelectionMenu

__all__ = [
    # Version
    "__version__",
    
    # Core data processing
    "TabularDataNormalizer",
    "TabularDataReviewer", 
    "TabularDataComparator",
    
    # Data readers
    "FileDataReader",
    "FileSystemNavigator",
    "AWSDataReader",
    "AthenaDataReader", 
    # "GoogleDriveDataReader",
    
    # Configuration
    "ColumnConfig",
    "CallSpec",
    "ColumnsConfig",
    "AWSCredentials",
    "AthenaTableDetails",
    
    # Utilities
    "LogsMachine",
    "SelectionMenu"
]