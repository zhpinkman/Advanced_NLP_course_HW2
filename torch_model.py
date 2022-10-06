from typing import List
import torch.optim as optim
import wandb
from tqdm import tqdm
from dataset import DataLoader, Dataset
from model import Model
import numpy as np
from IPython import embed
import torch
import torch.nn as nn
import io
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


class TorchModel(Model):
    def __init__(self, num_hidden: int, max_seq_len: int, embedding_file: str, label_set: set):
        self.num_hidden = num_hidden
        self.embedding = load_vectors(fname=embedding_file)
        self.max_seq_len = max_seq_len
        self.num_features = list(self.embedding.values())[0].shape[0]
        self.label_set = label_set
        self.ohe = OneHotEncoder()
        self.label2id = dict(zip(
            self.label_set,
            list(range(len(self.label_set)))
        ))
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.ohe.fit(np.array(list(self.id2label.keys())).reshape(-1, 1))

        self.model = nn.Sequential(
            nn.Linear(self.num_features*self.max_seq_len, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, len(self.label_set))
        )

    def tokenize(self, texts: List[str]):
        inputs = np.zeros(
            shape=[len(texts), self.max_seq_len * self.num_features])
        for i, text in enumerate(texts):
            text_embedding = list()
            words = text.lower().split()
            for word in words[:min(len(words), self.max_seq_len)]:
                if word in self.embedding:
                    text_embedding.extend(self.embedding[word].tolist())
                else:
                    text_embedding.extend(self.embedding['UNK'].tolist())

            padding_length = self.max_seq_len * \
                self.num_features - len(text_embedding)
            text_embedding = np.array(text_embedding)
            text_embedding = np.pad(text_embedding, pad_width=(
                0, padding_length), constant_values=[0, 0])
            inputs[i, :] = text_embedding
        return inputs

    def evaluate(self, dataset: Dataset, batch_size: int):
        raise NotImplementedError()

    def train(
        self,
        dataset: Dataset,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        wandb_comment: str,
        dev_dataset: Dataset = None,
        test_dataset: Dataset = None
    ):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_loss = []
            for batch in tqdm(data_loader.get_batches(), leave=False):
                texts = batch[0]
                labels = batch[1]
                inputs = self.tokenize(texts=texts)
                labels = torch.from_numpy(np.array([self.label2id[label]
                                                    for label in labels]))

                outputs = self.model(torch.from_numpy(inputs).float())

                loss = criterion(outputs, labels)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(np.mean(epoch_loss))

    def classify(self, input_file):
        pass
