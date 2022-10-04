import pickle
import argparse
from dataset import Dataset
from neural_model import NeuralModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')

    args = parser.parse_args()

    train_dataset = Dataset(file=args.i, with_labels=True)

    model = NeuralModel(
        num_hidden=args.u,
        max_seq_len=args.f,
        embedding_file=args.E,
        label_set=train_dataset.label_set
    )  # probably want to pass some arguments here

    model.train(
        dataset=train_dataset,
        batch_size=args.b,
        num_epochs=args.e,
        learning_rate=args.l
    )

    model.save_model(args.o)
