import json

import pandas as pd


INPUT_DATASET = "preprocessed.csv"
OUTPUT_DATASET = "final.csv"
REPORT_OUTPUT = "data_report.json"

TARGET_COLUMN = "species"
TARGET_ENCODED_COLUMN = "species_id"

SPECIES_TO_ID = {
    "Setosa": 0,
    "Versicolor": 1,
    "Virginica": 2,
}


def main() -> None:
    df = pd.read_csv(INPUT_DATASET)

    required_columns = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    unknown_species = sorted(set(df[TARGET_COLUMN]) - set(SPECIES_TO_ID))

    if unknown_species:
        raise ValueError(
            f"Unknown species values: {unknown_species}. "
            f"Expected: {sorted(SPECIES_TO_ID)}"
        )

    df[TARGET_ENCODED_COLUMN] = df[TARGET_COLUMN].map(SPECIES_TO_ID)

    numeric_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_column": TARGET_COLUMN,
        "target_encoded_column": TARGET_ENCODED_COLUMN,
        "class_distribution": {
            key: int(value)
            for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "numeric_summary": {
            column: {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
            }
            for column in numeric_columns
        },
    }

    df.to_csv(OUTPUT_DATASET, index=False)

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved final dataset to {OUTPUT_DATASET}")
    print(f"Saved report to {REPORT_OUTPUT}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()