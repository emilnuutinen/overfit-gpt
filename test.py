import json

import jsonlines


def main():
    with open("result.json", mode="r") as f:
        data = json.load(f)
        for chunk in data:
            line = {
                "chunk": chunk,
            }
            with jsonlines.open("result.jsonl", mode="a") as writer:
                writer.write(line)


if __name__ == "__main__":
    main()
