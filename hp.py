import pickle
import argparse
from dataset import Dataset
from neural_model import NeuralModel
import numpy as np
import wandb


if __name__ == '__main__':

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'dev_f1_score',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'batch_size': {
            'values': [4]
        },
        'num_epochs': {
            "value": 400
        },
        'mid_layer_dropout': {
            "values": [0.5, 0.6, 0.7, 0.8]
        },
        'layers_embeddings': {
            'values':
            [
                [32, 32, 32],
                [32, 32, 16],
                [16, 16, 16],
                [64, 32, 32]
            ]

        },
        'learning_rate': {
            'values': [1e-4, 1e-5, 5e-5]
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(
        sweep_config, project="Logical Fallacy Detection GCN Hyper parameter tuning V2")

    wandb.agent(sweep_id, train_with_wandb)

    train_dataset = Dataset(file=args.i, with_labels=True)
    dev_dataset = Dataset(file=args.dev_text, with_labels=False).load_labels(
        file=args.dev_labels)
    test_dataset = Dataset(file=args.test_text, with_labels=False).load_labels(
        file=args.test_labels)

    model = NeuralModel(
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
