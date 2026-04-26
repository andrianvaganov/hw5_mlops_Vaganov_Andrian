import json
import pickle

import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


TRAIN_DATASET = "train.csv"
TEST_DATASET = "test.csv"

MODEL_OUTPUT = "model.pkl"
METRICS_OUTPUT = "metrics.json"

TARGET_COLUMN = "variety"
PARAMS_FILE = "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    train_params = params.get("train", {})

    required_params = ["n_estimators", "max_depth", "random_state"]
    missing_params = [
        param for param in required_params
        if param not in train_params
    ]

    if missing_params:
        raise ValueError(
            f"Missing train params in {PARAMS_FILE}: {missing_params}"
        )

    return train_params


def split_features_and_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y


def main() -> None:
    params = load_params()

    n_estimators = int(params["n_estimators"])
    max_depth = params["max_depth"]
    random_state = int(params["random_state"])

    if max_depth is not None:
        max_depth = int(max_depth)

    train_df = pd.read_csv(TRAIN_DATASET)
    test_df = pd.read_csv(TEST_DATASET)

    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.startswith("Unnamed")]

    x_train, y_train_raw = split_features_and_target(train_df)
    x_test, y_test_raw = split_features_and_target(test_df)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
    }

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": list(x_train.columns),
        "target_column": TARGET_COLUMN,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
    }

    with open(MODEL_OUTPUT, "wb") as file:
        pickle.dump(artifact, file)

    with open(METRICS_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model to {MODEL_OUTPUT}")
    print(f"Saved metrics to {METRICS_OUTPUT}")
    print(
        f"Used n_estimators={n_estimators}, "
        f"max_depth={max_depth}, "
        f"random_state={random_state}"
    )
    print(metrics)


if __name__ == "__main__":
    main()