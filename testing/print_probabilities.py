import torch
from transformers import AutoModelForCausalLM, BloomTokenizerFast, set_seed

set_seed(0)

tokenizer = BloomTokenizerFast.from_pretrained("TurkuNLP/gpt3-finnish-small")
model = AutoModelForCausalLM.from_pretrained(
    "tmp_mini/",
    pad_token_id=tokenizer.eos_token_id)

prompt = 'Selm on kaupunki Nordrhein-Westfalenin osavaltiossa läntisessä Saksassa.'
new_tokens = 10
num_candidates = 10

for i in range(new_tokens):

    print(prompt)

    model_input = tokenizer(prompt, return_tensors='pt')

    # Get the logits of the next token
    with torch.inference_mode():
        outputs = model(**model_input)

    next_token_logits = outputs.logits[0, -1, :]

    # Convert the logits to probabilities
    next_token_probs = torch.softmax(next_token_logits, -1)

    # Get the top 10 tokens
    topk_next_tokens = torch.topk(next_token_probs, num_candidates)

    # Print tokens & probabilities
    print(*[(tokenizer.decode(idx), prob) for idx,
          prob in zip(topk_next_tokens.indices, topk_next_tokens.values)], sep="\n")

    prompt = prompt + tokenizer.decode(topk_next_tokens.indices[0])
