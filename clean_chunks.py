import jsonlines


def main():
    with jsonlines.open("chunks.jsonl", mode="r") as reader:
        for line in reader:
            chunked_text = [
                "".join([token for token in line["chunk"]])
            ]

            line = {
                "chunk": chunked_text
            }
            with jsonlines.open("clean.jsonl", mode="a") as writer:
                writer.write(line)


if __name__ == "__main__":
    main()
