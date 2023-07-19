from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

paths = [str(x) for x in Path("./fi_corpus/").glob("**/*.txt")]

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
tokenizer.train(files=paths, trainer=trainer)

tokenizer.save("model/tokenizer.json")
