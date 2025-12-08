"""
Quick test/demo of ColumnNormalizationConfig API
"""
from pathlib import Path
from verbosa.interfaces.columns_config import ColumnsConfig

# Load the example config
config_path = Path("verbosa/assets/examples/column_norm_config.yaml")
config = ColumnsConfig.from_yaml(config_path)

# Display metadata
print(f"Config: {config.name}")
print(f"Author: {config.author}")
print(f"Date: {config.date}")
print(f"Number of columns: {len(config)}\n")

# Display columns
for i, col in enumerate(config, 1):
    print(f"{i}. {col.name} ({col.dtype})")
    print(f"   Description: {col.description}")
    print(f"   Aliases: {col.aliases}")
    print(f"   Nullable: {col.nullable}, Duplicates: {col.allow_duplicates}")
    if col.fill_na is not None:
        print(f"   Fill NA: {col.fill_na}")
    if col.checks:
        print(f"   Checks: {[c.to_dict() for c in col.checks]}")
    print(f"   Normalization: {col.normalization}\n")

# Test column lookup by alias
print("\n--- Testing alias lookup ---")
col = config.get_column("transaction_date")
if col:
    print(f"Found column by alias 'transaction_date': {col.name}")

col = config.get_column("amount")
if col:
    print(f"Found column by alias 'amount': {col.name}")

# Validate config
print("\n--- Validation ---")
issues = config.validate()
if issues:
    print(f"Issues found: {issues}")
else:
    print("✓ Configuration is valid")

# Get all column names
print(f"\n--- Column names ---")
print(config.get_column_names())
