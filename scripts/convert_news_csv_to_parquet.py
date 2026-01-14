#!/usr/bin/env python3
"""Convert news calendar history CSV to Parquet format for Omega V2.

This script converts the legacy CSV news calendar file to the V2 Parquet format
following the specifications from OMEGA_V2_DATA_GOVERNANCE_PLAN.md:

- UTC time: Arrow Timestamp(Nanosecond, "UTC") - Event time in UTC
- Id: string (UUID)
- Name: string (Event name)
- Impact: string, normalized to LOW|MEDIUM|HIGH
- Currency: string, 3-letter uppercase

Usage:
    python scripts/convert_news_csv_to_parquet.py [--input] [--output]

Default paths:
    Input:  data/news/news_calender_history.csv
    Output: data/news/news_calender_history.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# V2 Contract: Allowed Impact values
VALID_IMPACTS = {"LOW", "MEDIUM", "HIGH"}

# V2 Contract: Column mapping (CSV -> Parquet)
COLUMN_MAPPING = {
    "Start": "UTC time",  # Rename Start to UTC time per V2 Contract
}

# V2 Contract: Required output columns in order
OUTPUT_COLUMNS = ["UTC time", "Id", "Name", "Impact", "Currency"]


def normalize_impact(impact: str) -> str:
    """Normalize Impact value to V2 Contract standard (LOW|MEDIUM|HIGH).
    
    Args:
        impact: Raw impact value from CSV.
        
    Returns:
        Normalized impact value.
        
    Raises:
        ValueError: If impact cannot be normalized.
    """
    normalized = impact.strip().upper()
    if normalized in VALID_IMPACTS:
        return normalized
    raise ValueError(
        f"Invalid Impact value: '{impact}' (normalized: '{normalized}')"
    )


def validate_currency(currency: str) -> str:
    """Validate and normalize Currency to 3-letter uppercase (V2 Contract).
    
    Args:
        currency: Raw currency value from CSV.
        
    Returns:
        Validated currency value.
        
    Raises:
        ValueError: If currency is invalid.
    """
    normalized = currency.strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError(
            f"Invalid Currency: '{currency}' "
            f"(must be 3-letter, got: '{normalized}')"
        )
    return normalized


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for manifest tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def convert_news_csv_to_parquet(
    input_path: Path,
    output_path: Path,
    *,
    validate_strict: bool = True,
) -> dict[str, str | int]:
    """Convert news calendar CSV to V2 Parquet format.
    
    Args:
        input_path: Path to input CSV file.
        output_path: Path to output Parquet file.
        validate_strict: If True, raise on validation errors.
            If False, log warnings instead.
        
    Returns:
        Dictionary with conversion statistics and hash.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If validation fails and validate_strict is True.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Reading CSV from: {input_path}")
    
    # Read CSV
    df = pd.read_csv(
        input_path,
        dtype={"Id": str, "Name": str, "Impact": str, "Currency": str},
    )
    original_count = len(df)
    logger.info(f"Loaded {original_count:,} rows from CSV")
    
    # Rename columns per V2 Contract
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Parse and convert UTC time to timezone-aware datetime (UTC)
    # V2 Contract: Arrow Timestamp(Nanosecond, "UTC")
    logger.info("Converting timestamps to UTC...")
    df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True)
    
    # Normalize Impact (V2 Contract: LOW|MEDIUM|HIGH)
    logger.info("Normalizing Impact values...")
    try:
        df["Impact"] = df["Impact"].apply(normalize_impact)
    except ValueError as e:
        if validate_strict:
            raise
        logger.warning(f"Impact normalization issue: {e}")
    
    # Validate Currency (V2 Contract: 3-letter uppercase)
    logger.info("Validating Currency values...")
    try:
        df["Currency"] = df["Currency"].apply(validate_currency)
    except ValueError as e:
        if validate_strict:
            raise
        logger.warning(f"Currency validation issue: {e}")
    
    # Sort by UTC time (V2 Contract: stable sort)
    logger.info("Sorting by UTC time...")
    df = df.sort_values("UTC time", kind="stable")
    
    # Deduplicate by Id (V2 Contract: keep-first)
    duplicates_count = df.duplicated(subset=["Id"], keep="first").sum()
    if duplicates_count > 0:
        logger.warning(
            f"Found {duplicates_count} duplicate Id entries, keeping first"
        )
        df = df.drop_duplicates(subset=["Id"], keep="first")
    
    # Reorder columns per V2 Contract
    df = df[OUTPUT_COLUMNS]
    
    final_count = len(df)
    removed = original_count - final_count
    logger.info(f"Final rows: {final_count:,} (removed {removed:,} dupes)")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyArrow Table with explicit schema
    # V2 Contract: Timestamp(Nanosecond, "UTC") for UTC time
    schema = pa.schema([
        ("UTC time", pa.timestamp("ns", tz="UTC")),
        ("Id", pa.string()),
        ("Name", pa.string()),
        ("Impact", pa.string()),
        ("Currency", pa.string()),
    ])
    
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    
    # Write Parquet
    logger.info(f"Writing Parquet to: {output_path}")
    pq.write_table(table, output_path, compression="snappy")
    
    # Compute hash for manifest
    output_hash = compute_sha256(output_path)
    logger.info(f"Output SHA-256: {output_hash}")
    
    # Summary statistics
    date_range_start = df["UTC time"].min()
    date_range_end = df["UTC time"].max()
    
    stats = {
        "input_rows": original_count,
        "output_rows": final_count,
        "duplicates_removed": original_count - final_count,
        "date_range_start": str(date_range_start),
        "date_range_end": str(date_range_end),
        "output_sha256": output_hash,
        "currencies": sorted(df["Currency"].unique().tolist()),
        "impact_distribution": df["Impact"].value_counts().to_dict(),
    }
    
    logger.info("Conversion completed successfully!")
    logger.info(f"  Date range: {date_range_start} to {date_range_end}")
    logger.info(f"  Currencies: {stats['currencies']}")
    logger.info(f"  Impact distribution: {stats['impact_distribution']}")
    
    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert news calendar CSV to V2 Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/news/news_calender_history.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/news/news_calender_history.parquet"),
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Log validation errors as warnings instead of failing",
    )
    
    args = parser.parse_args()
    
    try:
        stats = convert_news_csv_to_parquet(
            args.input,
            args.output,
            validate_strict=not args.lenient,
        )
        print(
            f"\n✅ Converted {stats['input_rows']:,} rows to Parquet"
        )
        print(f"   Output: {args.output}")
        print(f"   SHA-256: {stats['output_sha256']}")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
