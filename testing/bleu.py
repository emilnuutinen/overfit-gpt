import argparse
import warnings

import jsonlines
from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings("ignore")


def calculate_bleu_score(truth: str, prediction: str):
    reference = [truth.split()]
    hypothesis = prediction.split()
    return sentence_bleu(reference, hypothesis)


def main(file: str):
    with jsonlines.open(file, mode='r') as reader:
        lines = [obj for obj in reader]
        total_bleu_score = 0
        num_samples = len(lines)
        for line in lines:
            truth = line['truth']
            prediction = line['generated']
            bleu_score = calculate_bleu_score(truth, prediction)
            total_bleu_score += bleu_score

    average_bleu_score = total_bleu_score / num_samples
    print(f"Average BLEU Score: {average_bleu_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="jsonlines file to open")
    args = parser.parse_args()
    jsonlines_file_path = args.file
    main(jsonlines_file_path)
