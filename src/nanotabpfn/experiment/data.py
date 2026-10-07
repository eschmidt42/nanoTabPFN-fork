import logging

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


def _get_feature_preprocessor(X: NDArray) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X_df = pd.DataFrame(X)
    num_mask = []
    cat_mask = []

    for col in X_df:
        unique_non_nan_entries = X_df[col].dropna().unique()

        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue

        non_nan_entries = X_df[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X_df[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings

        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            (
                "to_pandas",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
                ),
            ),  # to apply pd.to_numeric of pandas
            (
                "to_numeric",
                FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy()),
            ),  # in case numeric columns are stored as strings
        ]
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_mask), ("cat", cat_transformer, cat_mask)]
    )
    return preprocessor


def get_openml_datasets(
    max_features_eval: int = 10,
    new_instances_eval: int = 200,
    target_classes_filter: int = 2,
    **kwargs,
) -> dict[str, tuple[NDArray, NDArray]]:
    """
    Load OpenML tabarena datasets with at most `max_features` features and subsampled (stratified) to `new_instances` instances.
    """
    task_ids = [
        363612,
        363613,
        363614,
        363615,
        363616,
        363618,
        363619,
        363620,
        363621,
        363623,
        363624,
        363625,
        363626,
        363627,
        363628,
        363629,
        363630,
        363631,
        363632,
        363671,
        363672,
        363673,
        363674,
        363675,
        363676,
        363677,
        363678,
        363679,
        363681,
        363682,
        363683,
        363684,
        363685,
        363686,
        363689,
        363691,
        363693,
        363694,
        363696,
        363697,
        363698,
        363699,
        363700,
        363702,
        363704,
        363705,
        363706,
        363707,
        363708,
        363711,
        363712,
    ]  # TabArena v0.1

    datasets = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue  # skip task, only classification

        dataset = task.get_dataset(download_data=False)

        qualities = dataset.qualities
        if qualities is None:
            msg = f"dataset.qualities is None for {task_id=}, skipping."
            logger.error(msg)
            continue

        if (
            qualities["NumberOfFeatures"] > max_features_eval
            or (qualities["NumberOfClasses"] > target_classes_filter)
            or qualities["PercentageOfInstancesWithMissingValues"] > 0
            or qualities["MinorityClassPercentage"] < 2.5
        ):
            continue

        X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")  # ty: ignore[unresolved-attribute]

        if y is None:
            msg = f"y is surprisingly None for {task_id=}, skipping."
            logger.error(msg)
            continue

        if new_instances_eval < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X,
                y,
                test_size=new_instances_eval,
                stratify=y,
                random_state=0,
            )
        else:
            X_sub = X
            y_sub = y

        X = X_sub.to_numpy(copy=True)  # ty: ignore[unresolved-attribute]
        y = y_sub.to_numpy(copy=True)  # ty: ignore[unresolved-attribute]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        preprocessor = _get_feature_preprocessor(X)
        X = preprocessor.fit_transform(X)
        datasets[dataset.name] = (X, y)

    return datasets
