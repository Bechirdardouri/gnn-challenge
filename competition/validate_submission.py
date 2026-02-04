"""Validate submission format for HeteroShot challenge.

This script validates that a submission file meets all requirements:
- Correct CSV format with required columns
- All test node IDs present
- Valid integer predictions
"""

import argparse
import csv
import sys
from pathlib import Path


def validate_submission(submission_path: str, test_path: str = "data/test.csv") -> tuple[bool, str]:
    """Validate a submission file.
    
    Args:
        submission_path: Path to the submission CSV file
        test_path: Path to the test nodes CSV file
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check file exists
    if not Path(submission_path).exists():
        return False, f"Submission file not found: {submission_path}"
    
    # Read test node IDs
    try:
        with open(test_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in reader.fieldnames:
                return False, f"Test file missing 'node_id' column"
            test_ids = set(row["node_id"] for row in reader)
    except Exception as e:
        return False, f"Error reading test file: {e}"
    
    # Read and validate submission
    try:
        with open(submission_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check header
            if reader.fieldnames is None:
                return False, "Submission CSV has no header row"
            
            required_cols = {"node_id", "target"}
            missing_cols = required_cols - set(reader.fieldnames)
            if missing_cols:
                return False, f"Submission missing columns: {missing_cols}"
            
            # Validate rows
            submission_ids = set()
            for row_num, row in enumerate(reader, start=2):
                node_id = row["node_id"]
                target = row["target"]
                
                # Check for duplicates
                if node_id in submission_ids:
                    return False, f"Duplicate node_id '{node_id}' at row {row_num}"
                submission_ids.add(node_id)
                
                # Check target is valid integer
                try:
                    int(target)
                except ValueError:
                    return False, f"Invalid target '{target}' at row {row_num} (must be integer)"
            
            # Check all test IDs are present
            missing_ids = test_ids - submission_ids
            if missing_ids:
                sample = list(missing_ids)[:5]
                return False, f"Missing predictions for {len(missing_ids)} nodes. Sample: {sample}"
            
            # Check for extra IDs
            extra_ids = submission_ids - test_ids
            if extra_ids:
                sample = list(extra_ids)[:5]
                return False, f"Extra predictions for {len(extra_ids)} unknown nodes. Sample: {sample}"
    
    except Exception as e:
        return False, f"Error reading submission: {e}"
    
    return True, f"Valid submission with {len(submission_ids)} predictions"


def validate_metadata(metadata_path: str) -> tuple[bool, str]:
    """Validate a metadata.json file.
    
    Args:
        metadata_path: Path to metadata.json
        
    Returns:
        Tuple of (is_valid, message)
    """
    import json
    
    if not Path(metadata_path).exists():
        return False, f"Metadata file not found: {metadata_path}"
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in metadata: {e}"
    except Exception as e:
        return False, f"Error reading metadata: {e}"
    
    # Required fields
    required = {"model_type"}
    missing = required - set(metadata.keys())
    if missing:
        return False, f"Metadata missing required fields: {missing}"
    
    # Validate model_type
    valid_types = {"human", "llm-only", "human+llm", "baseline"}
    if metadata["model_type"] not in valid_types:
        return False, f"Invalid model_type '{metadata['model_type']}'. Must be one of: {valid_types}"
    
    return True, "Valid metadata"


def main():
    parser = argparse.ArgumentParser(description="Validate HeteroShot submission")
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument("--metadata", help="Path to metadata.json (optional)")
    parser.add_argument("--test", default="data/test.csv", help="Path to test nodes CSV")
    args = parser.parse_args()
    
    # Validate submission
    is_valid, message = validate_submission(args.submission, args.test)
    print(f"Submission: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"  {message}")
    
    if not is_valid:
        sys.exit(1)
    
    # Validate metadata if provided
    if args.metadata:
        is_valid, message = validate_metadata(args.metadata)
        print(f"Metadata: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"  {message}")
        
        if not is_valid:
            sys.exit(1)
    
    print("\n✅ Submission validation passed!")


if __name__ == "__main__":
    main()
