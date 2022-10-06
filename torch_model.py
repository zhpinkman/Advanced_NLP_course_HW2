from typing import List
import wandb
from tqdm import tqdm
from dataset import DataLoader, Dataset
from model import Model
import numpy as np
from IPython import embed
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
        raise NotImplementedError()

    def classify(self, input_file):
        pass
