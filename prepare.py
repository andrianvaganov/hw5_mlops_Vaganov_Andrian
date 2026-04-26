import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_DATASET = "energydata_complete_v1.csv"
TRAIN_OUTPUT = "train.csv"
TEST_OUTPUT = "test.csv"

TARGET_COLUMN = "variety"


def main() -> None:
    df = pd.read_csv(INPUT_DATASET)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print(f"Saved train dataset to {TRAIN_OUTPUT}: {train_df.shape}")
    print(f"Saved test dataset to {TEST_OUTPUT}: {test_df.shape}")


if __name__ == "__main__":
    main()