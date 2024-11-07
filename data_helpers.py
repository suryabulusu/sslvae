import logging
import random
from typing import List

import datasets
from datasets import DatasetDict, load_dataset


def retain_mnist_classes(
    dataset: datasets.Dataset, allowed_labels: List[int]
) -> datasets.Dataset:
    logging.info(f"Filtering dataset to retain only classes: {allowed_labels}")
    return dataset.filter(lambda example: example["label"] in allowed_labels)


def split_labeled_unlabeled(
    dataset: datasets.Dataset, label_ratio: float = 0.1, validation_ratio: float = 0.3
) -> DatasetDict:
    """
    Split dataset into labeled and unlabeled subsets.

    Args:
        dataset (datasets.Dataset): Filtered dataset with specified classes.
        label_ratio (float): Proportion of labeled data in the final dataset.
        validation_ratio (float): Proportion of validation data in the final dataset.

    Returns:
        DatasetDict: Labeled and unlabeled splits.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_split_idx = int(len(dataset) * validation_ratio)
    label_split_idx = val_split_idx + int(len(dataset) * label_ratio)

    val_indices = indices[:val_split_idx]
    labeled_indices = indices[val_split_idx:label_split_idx]
    unlabeled_indices = indices[label_split_idx:]

    labeled_dataset = dataset.select(labeled_indices)
    unlabeled_dataset = dataset.select(unlabeled_indices)
    validation_dataset = dataset.select(val_indices)

    logging.info(
        f"Labeled dataset size: {len(labeled_dataset)}, "
        f"Unlabeled dataset size: {len(unlabeled_dataset)}, "
        f"Validation dataset size: {len(validation_dataset)}"
    )
    return DatasetDict(
        {
            "labeled": labeled_dataset,
            "unlabeled": unlabeled_dataset,
            "validation": validation_dataset,
        }
    )


def load_mnist_for_ssl(
    allowed_labels: List[int] = [1, 2, 3, 4],
    label_ratio: float = 0.1,
    validation_ratio: float = 0.3,
) -> DatasetDict:
    """
    Load and prepare MNIST dataset for semi-supervised learning.

    Args:
        allowed_labels (List[int]): Digits to retain in dataset.
        label_ratio (float): Proportion of labeled data.

    Returns:
        DatasetDict: Dictionary with "labeled" and "unlabeled" splits.
    """
    logging.info("Loading MNIST dataset.")
    dataset = load_dataset("mnist")["train"]
    filtered_dataset = retain_mnist_classes(dataset, allowed_labels)
    return split_labeled_unlabeled(filtered_dataset, label_ratio, validation_ratio)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # put in args
    mnist_ssl = load_mnist_for_ssl(
        allowed_labels=[1, 2, 3, 4], label_ratio=0.1, validation_ratio=0.3
    )
