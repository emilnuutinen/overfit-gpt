from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

path = ["data/all.txt"]

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    show_progress=True)
tokenizer.train(files=path, trainer=trainer)

tokenizer.save("model/tokenizer.json")
