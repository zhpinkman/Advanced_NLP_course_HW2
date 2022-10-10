from typing import List


class Dataset:
    def __init__(self, file: str, with_labels: bool) -> None:
        with open(file, 'r') as f:
            data = f.read().splitlines()

        if with_labels:
            self.texts, self.labels = zip(*list(map(
                lambda x: x.split('\t'),
                data
            )))
        else:
            self.texts = data
            self.labels = None

    def __getitem__(self, index):
        if self.labels:
            return self.texts[index], self.labels[index]
        else:
            return self.texts[index], None

    @property
    def label_set(self):
        if self.labels:
            return set(self.labels)
        return None

    def load_labels(self, file: str):
        with open(file, 'r') as f:
            self.labels = f.read().splitlines()
        return self

    def __len__(self):
        return len(self.texts)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int) -> None:
        self.dataset = dataset
        self.indices = [(i, min(len(dataset), i + batch_size))
                        for i in range(0, len(dataset), batch_size)]

    def __getitem__(self, index):
        return self.dataset[index[0]:index[1]]

    def get_batches(self):
        for index in self.indices:
            yield self.__getitem__(index=index)
