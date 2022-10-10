import pickle
import argparse
from dataset import Dataset
from torch_model import TorchModel
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neural net training arguments using Pytorch.')

    parser.add_argument(
        '-u', type=str, help='number of hidden units seperated by comma')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('--dropout', type=float,
                        help="the rate with which the dropout should be applied", default=0)
    parser.add_argument(
        '-w', type=float, help="weight decay for l2 regularization", default=0)
    parser.add_argument(
        '--dev_text', help="text file of the dev split", type=str)
    parser.add_argument(
        '--dev_labels', help="label file of the dev split", type=str)
    parser.add_argument(
        '--test_text', help="text file of the test split", type=str
    )
    parser.add_argument(
        '--test_labels', help="label file of the test split", type=str
    )
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('--wandb_comment',
                        help="comment to append at the end of wandb project", type=str)

    args = parser.parse_args()

    train_dataset = Dataset(file=args.i, with_labels=True)
    dev_dataset = Dataset(file=args.dev_text, with_labels=False).load_labels(
        file=args.dev_labels)
    test_dataset = Dataset(file=args.test_text, with_labels=False).load_labels(
        file=args.test_labels)

    model = TorchModel(
        num_hiddens=[int(num) for num in args.u.split(',')],
        weight_decay=args.w,
        max_seq_len=args.f,
        embedding_file=args.E,
        label_set=train_dataset.label_set,
        data_file_name=args.i,
        dropout=args.dropout
    )

    model.train(
        dataset=train_dataset,
        batch_size=args.b,
        num_epochs=args.e,
        learning_rate=args.l,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        wandb_comment=args.wandb_comment
    )

    model.save_model(args.o)
