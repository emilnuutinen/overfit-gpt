import argparse

import jsonlines


def main(file: str):
    with jsonlines.open(file, mode="r") as reader:
        lines = [obj for obj in reader]
        total_score = 0
        num_samples = len(lines)
        for line in lines:
            truth = line["truth"]
            prediction = line["generated"]
            match = truth == prediction
            if match:
                total_score += 1

    average_score = total_score / num_samples
    print(f"Average Score: {average_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="jsonlines file to open")
    args = parser.parse_args()
    jsonlines_file_path = args.file
    main(jsonlines_file_path)
