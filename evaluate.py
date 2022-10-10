import argparse
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score


def read_labels(input_file: str) -> List[str]:
    with open(input_file, 'r') as f:
        data = f.read().splitlines()
    return data


def compute_scores(true_labels: List[str], predictions: List[str]) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true=true_labels, y_pred=predictions),
        'f1_score': f1_score(y_true=true_labels, y_pred=predictions, average="weighted")
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate the performance of the model given the labels and predictions')

    parser.add_argument('--labels', type=str,
                        help="labels file with each line containing one label")
    parser.add_argument('--predictions', type=str,
                        help="predictions file with each line containing one prediction")

    parser.add_argument('--output_file', type=str,
                        help="the file that the output should be shown there")

    args = parser.parse_args()

    true_labels = read_labels(args.labels)
    predictions = read_labels(args.predictions)

    results = compute_scores(
        true_labels=true_labels,
        predictions=predictions
    )

    results_str = f"Results comparing files {args.labels} and {args.predictions} \n\n"
    for key, value in results.items():
        results_str += f"{key}: {value} \n"

    with open(args.output_file, 'w') as f:
        f.write(results_str)
