import os
from sklearn.model_selection import train_test_split
from IPython import embed
from pathlib import Path

Path('/root/dir/sub/file.ext').stem

dataset_files = [file for file in os.listdir(
    '.') if file.endswith('train.txt')]

for dataset_file in dataset_files:
    with open(dataset_file, 'r') as f:
        data = f.read().splitlines()

    texts, labels = zip(*list(map(
        lambda x: x.split('\t'),
        data
    )))

    X_train, X_dev, y_train, y_dev = train_test_split(
        texts, labels, test_size=0.1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    with open(os.path.join('custom_dataset', f"{Path(dataset_file).stem}.train.txt"), 'w') as f:
        for text, label in zip(X_train, y_train):
            f.write(f"{text}\t{label}\n")

    with open(os.path.join('custom_dataset', f"{Path(dataset_file).stem}.dev.txt"), 'w') as f:
        for text in X_dev:
            f.write(f"{text}\n")

    with open(os.path.join('custom_dataset', f"{Path(dataset_file).stem}.dev_labels.txt"), 'w') as f:
        for label in y_dev:
            f.write(f"{label}\n")

    with open(os.path.join('custom_dataset', f"{Path(dataset_file).stem}.test.txt"), 'w') as f:
        for text in X_test:
            f.write(f"{text}\n")

    with open(os.path.join('custom_dataset', f"{Path(dataset_file).stem}.test_labels.txt"), 'w') as f:
        for label in y_test:
            f.write(f"{label}\n")
