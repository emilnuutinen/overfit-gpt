from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed

tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/gpt2-finnish")
model = pipeline('text-generation', model='Finnish-NLP/gpt2-finnish')
wiki = load_dataset("graelo/wikipedia", "20230601.fi")

set_seed(0)


# Collect samples from the last 5% of the dataset & cut them to 500 tokens
def collect_data(dataset):
    print(f"Full dataset: {len(dataset)}")
    num_samples = int(0.1 * len(dataset))
    samples = dataset["text"][-num_samples:]
    print(f"Samples: {len(samples)}")
    splits = []
    for sample in samples:
        splitted = split_text(sample)
        splits.append(splitted)
    return splits


def split_text(text, max_chunk_length=500):
    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Initialize variables to keep track of chunks
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_chunk_length:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

    # Append the last chunk if it's not empty
#    if current_chunk:
#        chunks.append(current_chunk)

    # Decode each chunk back into text
    chunked_text = [tokenizer.decode(
        chunk, skip_special_tokens=True) for chunk in chunks]

    return chunked_text


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


# Loop through these examples and use `len-50` as prompt and predict 50 tokens
def predict(data):
    # model(prompt, min_new_tokens=50, max_new_tokens=50)
    return f"Final samples: {len(data)}"


# If the model correctly predicts the 50 tokens we give score 1, otherwise 0
def score(prediction: str, truth: str) -> bool:
    return prediction == truth


# Average the score over all examples
def average(total: int, correct: int) -> float:
    fraction = correct/total
    return fraction


def main():
    data = collect_data(wiki["train"])
    flattened = flatten(data)
    print(predict(flattened))


if __name__ == "__main__":
    main()
