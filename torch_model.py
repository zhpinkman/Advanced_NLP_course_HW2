from typing import List
import torch.optim as optim
# import wandb
from tqdm import tqdm
from dataset import DataLoader, Dataset
from model import Model
import numpy as np
from IPython import embed
import torch
import torch.nn as nn
import io
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


def load_vectors(fname):
    """Load the embedding vectors

    Args:
        fname (str): file containing the embedding vectors word per line

    Returns:
        Dictionary of the words and their embedding vectors
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split()
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


class TorchModel(Model):
    def __init__(self,
                 num_hiddens: List[int],
                 weight_decay: float,
                 max_seq_len: int,
                 embedding_file: str,
                 label_set: set,
                 data_file_name: str,
                 dropout: float
                 ):
        self.num_hiddens = num_hiddens
        self.weight_decay = weight_decay
        self.data_file_name = data_file_name
        self.dropout = dropout
        self.embedding_file = embedding_file
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

        layers = []
        torch.manual_seed(77)
        layers.append(
            nn.Linear(
                self.num_features*self.max_seq_len,
                self.num_hiddens[0]
            )
        )
        layers.append(nn.Sigmoid())
        # layers.append(nn.Dropout1d(0.5))

        for i in range(1, len(self.num_hiddens)):
            torch.manual_seed(66 + i)
            layers.append(
                nn.Linear(
                    self.num_hiddens[i - 1],
                    self.num_hiddens[i]
                )
            )
            layers.append(nn.Sigmoid())
            # layers.append(nn.Dropout1d(0.5))

        torch.manual_seed(55)
        layers.append(
            nn.Linear(
                self.num_hiddens[-1],
                len(self.label_set)
            )
        )

        self.model = nn.Sequential(*layers)

    def load_embedding_file(self):
        self.embedding = load_vectors(fname=self.embedding_file)

    def prepare_model_to_be_saved(self):
        self.embedding = None

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub('([-/.,!?()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub("didn't", "did not", text)
        text = re.sub("wasn't", "was not", text)
        text = re.sub("weren't", "were not", text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split()
                        if word not in stopwords])
        return text

    def tokenize(self, texts: List[str]):
        """Tokenize the texts into the input vectors suitable to be fed to the neural network first layer

        Args:
            texts (List[str]): List of texts in the data

        Returns:
            _type_: input vectors ready to pass to the network
        """
        inputs = np.zeros(
            shape=[len(texts), self.max_seq_len * self.num_features])
        for i, text in enumerate(texts):
            unk_counter = 0
            text_embedding = list()
            if 'odiya' not in self.data_file_name:
                words = self.preprocess(text).split()
            else:
                words = text.split()

            for word in words[:min(len(words), self.max_seq_len)]:
                if word.strip() in self.embedding:
                    text_embedding.extend(
                        self.embedding[word.strip()].tolist())
                else:
                    unk_counter += 1
                    text_embedding.extend(self.embedding['UNK'].tolist())

            # print(f"Number of unknown words: {unk_counter / len(words)}")
            # embed()
            padding_length = self.max_seq_len * \
                self.num_features - len(text_embedding)
            text_embedding = np.array(text_embedding)
            text_embedding = np.pad(text_embedding, pad_width=(
                0, padding_length), constant_values=[0, 0])
            inputs[i, :] = text_embedding

        return inputs

    def evaluate(self, dataset: Dataset, batch_size: int, criterion):
        """Evaluate the performance of the model on given dataset

        Args:
            dataset (Dataset): dataset to evaluate the model on
            batch_size (int): batch_size
            criterion: the loss function used to compute the loss

        Returns:
            Dict: Dictionary containing the metrics computed
        """
        self.model.eval()
        all_predictions = []
        true_labels = []
        all_losses = []
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        with torch.no_grad():
            for batch in tqdm(data_loader.get_batches(), leave=False):
                texts = batch[0]
                labels = batch[1]
                true_labels.extend(list(labels))
                inputs = self.tokenize(texts=texts)
                labels = torch.from_numpy(np.array([self.label2id[label]
                                                    for label in labels]))

                outputs = self.model(torch.from_numpy(inputs).float())

                loss = criterion(outputs, labels)
                all_losses.append(loss.item())

                predictions = np.argmax(outputs.numpy(), axis=-1)
                predictions = [self.id2label[prediction]
                               for prediction in predictions]
                all_predictions.extend(predictions)

        f1 = f1_score(y_true=true_labels,
                      y_pred=all_predictions, average='weighted')
        acc = accuracy_score(y_true=true_labels, y_pred=all_predictions)

        return {
            "predictions": all_predictions,
            "loss": np.mean(all_losses),
            "f1": f1,
            "acc": acc
        }

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
        """train the model given the datasets for train, dev, test split and also given hyperparameters

        Args:
            dataset (Dataset): train dataset
            batch_size (int): batch size
            num_epochs (int): num epochs to do training
            learning_rate (float): learning rate
            wandb_comment (str): comment to init the wandb project with
            dev_dataset (Dataset, optional): dev split dataset. Defaults to None.
            test_dataset (Dataset, optional): test split dataset. Defaults to None.
        """

        # wandb.init(
        #     project=f"Advanced NLP A2 - {wandb_comment}",
        #     config={
        #         "max_seq_len": self.max_seq_len,
        #         "num_hiddens": self.num_hiddens,
        #         "learning_rate": learning_rate,
        #         "num_epochs": num_epochs,
        #         "batch_size": batch_size
        #     }
        # )

        self.batch_size = batch_size
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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

            all_metrics = dict()
            train_metrics = self.evaluate(
                dataset=dataset,
                batch_size=batch_size,
                criterion=criterion
            )
            all_metrics.update(
                {f"train_{key}": val for key, val in train_metrics.items()})
            if dev_dataset:
                dev_metrics = self.evaluate(
                    dataset=dev_dataset,
                    batch_size=batch_size,
                    criterion=criterion
                )
                all_metrics.update(
                    {f"dev_{key}": val for key, val in dev_metrics.items()})
                print('\n')
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{all_metrics['train_loss']:.4}" + " | Valid loss: " +
                      f"{all_metrics['dev_loss']:.4} | F1 : " + f"{all_metrics['train_f1']:.4} | Valid F1: " + f"{all_metrics['dev_f1']:.4}")

            # wandb.log(all_metrics, step=epoch)
        if test_dataset:
            test_metrics = self.evaluate(
                dataset=test_dataset,
                batch_size=batch_size,
                criterion=criterion
            )
            print('Evaluating Test Dataset')
            print(
                "Test F1: " +
                f"{test_metrics['f1']:.4} | Test Acc: " +
                f"{test_metrics['acc']:.4} "
            )

    def classify(self, dataset: Dataset):
        """Classify the data points in the given dataset

        Args:
            dataset (Dataset): dataset to predict the labels for

        Returns:
            List[Any]: list of predicted labels
        """
        self.model.eval()
        all_predictions = []
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        with torch.no_grad():
            for batch in tqdm(data_loader.get_batches(), leave=False):
                texts = batch[0]
                inputs = self.tokenize(texts=texts)

                outputs = self.model(torch.from_numpy(inputs).float())

                predictions = np.argmax(outputs.numpy(), axis=-1)
                predictions = [self.id2label[prediction]
                               for prediction in predictions]
                all_predictions.extend(predictions)

        return all_predictions
