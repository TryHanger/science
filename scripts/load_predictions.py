"""Load and validate VQA predictions CSV into PostgreSQL database."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


REQUIRED_COLUMNS = [
    "question_id",
    "image_id",
    "question",
    "predicted_answer",
    "confidence",
    "is_correct",
    "answer_type",
]

VALID_ANSWER_TYPES = {"yes/no", "number", "other"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load VQA predictions from CSV to PostgreSQL database."
    )
    parser.add_argument(
        "--csv",
        default="outputs/predictions.csv",
        help="Path to predictions CSV file (default: outputs/predictions.csv)",
    )
    parser.add_argument(
        "--run-tag",
        default="local",
        help="Run tag identifier (default: local)",
    )
    parser.add_argument(
        "--model-name",
        default="vqa_mul",
        help="Model name identifier (default: vqa_mul)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation and print summary without connecting to DB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database upsert (default: 1000)",
    )
    return parser.parse_args()


def validate_predictions(csv_path: str) -> pd.DataFrame:
    """Validate predictions CSV file according to requirements.

    Requirements:
      - All columns present
      - No empty/null values
      - confidence in [0, 1]
      - is_correct is boolean
      - answer_type in {'yes/no', 'number', 'other'}
      - question_id is unique
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions CSV file not found: {csv_path}")

    df = pd.read_csv(path)

    # 1. Check all required columns are present
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    # 2. Check no empty / null values
    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    if null_counts.any():
        failing = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"CSV contains null/missing values in columns: {failing}")

    # Check for empty string values in text columns
    for col in ["question", "predicted_answer", "answer_type"]:
        if (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Column '{col}' contains empty string values")

    # 3. Check confidence in [0, 1]
    if not pd.api.types.is_numeric_dtype(df["confidence"]):
        raise ValueError("Column 'confidence' must have numeric values")

    out_of_bounds = df[(df["confidence"] < 0.0) | (df["confidence"] > 1.0)]
    if not out_of_bounds.empty:
        sample_invalid = out_of_bounds[["question_id", "confidence"]].head()
        raise ValueError(
            f"Found {len(out_of_bounds)} confidence values outside [0, 1]. Sample:\n{sample_invalid}"
        )

    # 4. Check is_correct is boolean
    if not pd.api.types.is_bool_dtype(df["is_correct"]):
        valid_bool_literals = {
            True,
            False,
            1,
            0,
            "True",
            "False",
            "true",
            "false",
            "1",
            "0",
            "t",
            "f",
        }
        if not df["is_correct"].isin(valid_bool_literals).all():
            invalid_vals = (
                df.loc[~df["is_correct"].isin(valid_bool_literals), "is_correct"]
                .unique()
                .tolist()
            )
            raise ValueError(f"Column 'is_correct' contains non-boolean values: {invalid_vals}")
        bool_map = {
            True: True,
            False: False,
            1: True,
            0: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "t": True,
            "f": False,
        }
        df["is_correct"] = df["is_correct"].map(bool_map).astype(bool)

    # 5. Check answer_type in {'yes/no', 'number', 'other'}
    invalid_answer_types = (
        df.loc[~df["answer_type"].isin(VALID_ANSWER_TYPES), "answer_type"]
        .unique()
        .tolist()
    )
    if invalid_answer_types:
        raise ValueError(
            f"Invalid answer_type values found: {invalid_answer_types}. "
            f"Allowed types are: {sorted(list(VALID_ANSWER_TYPES))}"
        )

    # 6. Check question_id uniqueness
    if not df["question_id"].is_unique:
        duplicates = (
            df[df["question_id"].duplicated(keep=False)]["question_id"]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Column 'question_id' contains duplicate values. "
            f"Unique duplicates count: {len(duplicates)}, examples: {duplicates[:5]}"
        )

    # Cast to ensure appropriate Python/database types
    df["question_id"] = df["question_id"].astype("int64")
    df["image_id"] = df["image_id"].astype("int64")
    df["question"] = df["question"].astype(str)
    df["predicted_answer"] = df["predicted_answer"].astype(str)
    df["confidence"] = df["confidence"].astype(float)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["answer_type"] = df["answer_type"].astype(str)

    return df


def print_summary(
    df: pd.DataFrame, csv_path: str, run_tag: str, model_name: str
) -> None:
    """Print summary statistics of the predictions dataset."""
    total_rows = len(df)
    overall_correct = int(df["is_correct"].sum())
    overall_accuracy = (100.0 * overall_correct / total_rows) if total_rows > 0 else 0.0

    print("=== Validation Summary ===")
    print(f"CSV Path:       {csv_path}")
    print(f"Run Tag:        {run_tag}")
    print(f"Model Name:     {model_name}")
    print(f"Total Rows:     {total_rows}")
    print()
    print("--- Answer Type Distribution ---")
    type_counts = df["answer_type"].value_counts()
    for atype, count in type_counts.items():
        pct = (100.0 * count / total_rows) if total_rows > 0 else 0.0
        print(f"  {atype:<10}: {count:>6} ({pct:>5.2f}%)")
    print()
    print("--- Accuracy ---")
    print(f"  Overall:     {overall_correct:>6} / {total_rows} ({overall_accuracy:.2f}%)")
    print("  By answer_type:")
    for atype in ["yes/no", "number", "other"]:
        subset = df[df["answer_type"] == atype]
        sub_total = len(subset)
        if sub_total > 0:
            sub_correct = int(subset["is_correct"].sum())
            sub_acc = 100.0 * sub_correct / sub_total
            print(f"    {atype:<8}: {sub_correct:>5} / {sub_total:>5} ({sub_acc:>5.2f}%)")


def get_db_connection():
    """Establish PostgreSQL connection using libpq environment variables."""
    try:
        import psycopg
    except ImportError:
        # Attempt to install psycopg[binary]
        cmd = [sys.executable, "-m", "pip", "install", "psycopg[binary]"]
        print(f"psycopg not found. Installing via: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        import psycopg

    conn_kwargs = {}
    for env_key, param_name in [
        ("PGHOST", "host"),
        ("PGPORT", "port"),
        ("PGUSER", "user"),
        ("PGPASSWORD", "password"),
        ("PGDATABASE", "dbname"),
    ]:
        val = os.environ.get(env_key)
        if val:
            conn_kwargs[param_name] = val

    return psycopg.connect(**conn_kwargs)


def apply_schema(conn) -> None:
    """Apply sql/schema.sql to ensure table and index exist."""
    search_paths = [
        Path(__file__).resolve().parent.parent / "sql" / "schema.sql",
        Path("sql/schema.sql"),
    ]
    schema_path = None
    for p in search_paths:
        if p.is_file():
            schema_path = p
            break

    if not schema_path:
        raise FileNotFoundError("Could not locate sql/schema.sql file")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()


def load_to_database(
    df: pd.DataFrame,
    run_tag: str,
    model_name: str,
    batch_size: int = 1000,
) -> int:
    """Upsert validated DataFrame into PostgreSQL in batches."""
    upsert_sql = """
    INSERT INTO vqa_predictions (
        run_tag,
        model_name,
        question_id,
        image_id,
        question,
        predicted_answer,
        confidence,
        is_correct,
        answer_type
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (run_tag, model_name, question_id) DO UPDATE SET
        image_id = EXCLUDED.image_id,
        question = EXCLUDED.question,
        predicted_answer = EXCLUDED.predicted_answer,
        confidence = EXCLUDED.confidence,
        is_correct = EXCLUDED.is_correct,
        answer_type = EXCLUDED.answer_type;
    """

    records = [
        (
            run_tag,
            model_name,
            int(row.question_id),
            int(row.image_id),
            str(row.question),
            str(row.predicted_answer),
            float(row.confidence),
            bool(row.is_correct),
            str(row.answer_type),
        )
        for row in df.itertuples(index=False)
    ]

    total_records = len(records)

    with get_db_connection() as conn:
        print("Connected to database via libpq environment variables.")
        print("Applying sql/schema.sql...")
        apply_schema(conn)
        print("Schema applied successfully.")

        print(f"Upserting {total_records} rows in batches of {batch_size}...")
        with conn.cursor() as cur:
            for start_idx in range(0, total_records, batch_size):
                batch = records[start_idx : start_idx + batch_size]
                cur.executemany(upsert_sql, batch)
        conn.commit()

    print(f"Successfully loaded {total_records} records (run_tag='{run_tag}', model_name='{model_name}').")
    return total_records


def main() -> None:
    """Main execution function."""
    args = parse_args()

    print(f"Validating predictions CSV: {args.csv}")
    df = validate_predictions(args.csv)
    print("CSV validation passed successfully.")
    print()

    print_summary(df, args.csv, args.run_tag, args.model_name)

    if args.dry_run:
        print()
        print("Dry-run mode enabled. Skipping database connection and insertion.")
        return

    # Check for PG credentials before attempting live load
    if not os.environ.get("PGUSER") or not os.environ.get("PGPASSWORD"):
        print()
        print(
            "PGUSER and PGPASSWORD environment variables are not set. "
            "Skipping database connection."
        )
        return

    print()
    load_to_database(df, args.run_tag, args.model_name, args.batch_size)


if __name__ == "__main__":
    main()
