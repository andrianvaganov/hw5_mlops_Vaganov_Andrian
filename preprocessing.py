import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_DATASET = "energydata_complete_v1.csv"
TRAIN_OUTPUT = "train.csv"
TEST_OUTPUT = "test.csv"

TARGET_COLUMN = "variety"
PARAMS_FILE = "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    preprocessing_params = params.get("preprocessing", {})

    required_params = ["split_ratio", "random_state"]
    missing_params = [
        param for param in required_params
        if param not in preprocessing_params
    ]

    if missing_params:
        raise ValueError(
            f"Missing preprocessing params in {PARAMS_FILE}: {missing_params}"
        )

    return preprocessing_params


def validate_split_ratio(split_ratio: float) -> None:
    if not 0 < split_ratio < 1:
        raise ValueError(
            f"split_ratio must be between 0 and 1, got: {split_ratio}"
        )


def main() -> None:
    params = load_params()

    split_ratio = float(params["split_ratio"])
    random_state = int(params["random_state"])

    validate_split_ratio(split_ratio)

    df = pd.read_csv(INPUT_DATASET)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna()
    df = df.drop_duplicates()

    train_df, test_df = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print(f"Saved train dataset to {TRAIN_OUTPUT}: {train_df.shape}")
    print(f"Saved test dataset to {TEST_OUTPUT}: {test_df.shape}")
    print(f"Used split_ratio={split_ratio}, random_state={random_state}")


if __name__ == "__main__":
    main()