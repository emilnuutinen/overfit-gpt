import jsonlines
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/gpt2_small_200_epochs/")


def main():
    with jsonlines.open("result.jsonl", mode="r") as reader:
        for line in reader:
            chunked_text = [
                tokenizer.decode(token, skip_special_tokens=True) for token in line["chunk"]
            ]

            line = {
                "chunk": chunked_text,
            }
            with jsonlines.open("chunks.jsonl", mode="a") as writer:
                writer.write(line)


if __name__ == "__main__":
    main()
