
#!pip install -q transformers datasets accelerate evaluate

import logging
from datasets import load_dataset
import transformers
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from collections import defaultdict
from transformers import Trainer, TrainingArguments

logging.disable(logging.INFO)



# TODO eval data
# create dummy evaluation dataset
#from datasets import Dataset
#eval_dataset = Dataset.from_dict({"text": ["Säiliö on suunniteltu kuljetettavaksi normaalilla B-luokan ajokortilla, jolloin kuka tahansa henkilöautokortin omaava voi vetää perässään alle 750 Kg peräkärryä. Huoltoasemalla tapahtuvan tankkauksen jälkeen kannattaa täyttää rahtikirja, joita kannattaa säilyttää ajoneuvon sisällä. Säiliö on varustettu tarvittavin merkinnöin: valmistajan kilpi, polttoaineen varoitusmerkki, polttoaineen UN-numero, ympäristölle vaarallinen merkki ja pinoamiskieltomerkki.", "Tätä sopimusta koskevat riitaisuudet ratkaistaan välimiesmenettelyssä. Välimiehen tai välimiehet asettaa Keskuskauppakamarin välityslautakunta ja välimiesmenettelyssä noudatetaan tämän lautakunnan sääntöjä. Kielikoneella on oikeus periä erääntyneet saatavansa välimiesmenettelyn sijasta tuomioistuinteitse, jolloin alioikeutena on Kielikoneen kotipaikan yleinen alioikeus. Mikäli Asiakas on kuluttaja, Asiakkaalla on kuitenkin kaikissa tapauksissa oikeus nostaa kanne myös yleisessä tuomioistuimessa tai saattaa asia kuluttajariitalautakunnan käsittelyyn. Tähän sopimukseen ja siitä aiheutuvien erimielisyyksien ratkaisemiseen sovelletaan Suomen lakia, pois lukien sen lainvalintaa koskevat säännökset.", "Microsoft-tili (aikaisemmin käytettiin nimitystä Windows Live ID) on sähköpostiosoitteen ja salasanan yhdistelmä, joita voit käyttää kirjautuessasi sisään palveluihin, kuten OneDrive, Windows Phone, Xbox LIVE ja Outlook.com (ja aikaisemmin Hotmail tai Messenger) . Jos käytät sähköpostiosoitetta ja salasanaa kirjautuessasi sisään näihin tai muihin Microsoft-palveluihin, käytössäsi on jo Microsoft-tili. Jos sinulla ei ole Microsoft-tiliä, voit luoda sen helposti. Voit yhdistää olemassa olevan Skype-tilin Microsoft-tiliin eri sovellusten ja palveluiden kertakirjautumista varten."]})
#print(eval_dataset)
#print(eval_dataset[0])
#eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['text'])
#print(eval_dataset)
#print(eval_dataset[0])


class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)


# TODO: return_special_tokens_mask=True
# self.tokenizer.pad_token_id
# tokenizer.padding_side
def tokenize(example, tokenizer=None, context_size=512):
    outputs = tokenizer(
        example["text"],
        truncation=True,
        max_length=context_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    # split one example with multiple segments into multiple examples with one segment
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_size: # discard the final batch with incorrect length
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def prepare_data(tokenizer):
    dataset = load_dataset("graelo/wikipedia", "20230601.fi", split="train[:1%]")#, streaming=True)
    trainset = dataset.select(range(100))
    evalset = dataset.select([i for i in range(10)]) # TODO: i+100 to get different evalset
    print("Train:", trainset)
    print("Train docs:", [d["title"] for d in trainset])
    print("Eval:", evalset)
    print("Eval docs:", [d["title"] for d in evalset])
    tokenized_train = trainset.map(tokenize, batched=True, remove_columns=['id', 'title', 'url', 'text'], fn_kwargs={"tokenizer": tokenizer})
    tokenized_eval = evalset.map(tokenize, batched=True, remove_columns=['id', 'title', 'url', 'text'], fn_kwargs={"tokenizer": tokenizer})
    print("Tokenized train:",tokenized_train)
    print("Tokenized eval:",tokenized_eval)
    return tokenized_train, tokenized_eval

def create_models():
    # create tokenizer
    tokenizer_name = "TurkuNLP/gpt3-finnish-small"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token # why?

    # Initializing a GPT2 configuration
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    # Initializing a model (with random weights) from the configuration
    model = GPT2LMHeadModel(config)

    # Accessing the model configuration
    print("Model config:", model.config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    return tokenizer, model


def train_model(train_data, eval_data, tokenizer, model):

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    batches = data_collator(train_data["input_ids"])
    for i, b in enumerate(batches["input_ids"]):
        #if tokenizer.batch_decode(b, skip_special_tokens=True)[0] != "Auringon":
        #    continue
        print("Batch:", i)
        print("Len:", len(b))
        print("input_ids:", b)
        decoded = tokenizer.batch_decode(b, skip_special_tokens=True)
        print("Tokens:", decoded)
        print("Len:", len(decoded))
        if i>5:
            break
    sys.exit()


    training_logs = LogSavingCallback()

    args = TrainingArguments(
        output_dir="checkpoints/fin-gpt-test",
        overwrite_output_dir=True,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=500,
        save_total_limit=5,
        gradient_accumulation_steps=8,
        weight_decay=0.1,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        fp16=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[training_logs]
    )

    trainer.train()

    


def main():

    
    tokenizer, model = create_models()
    trainset, evalset = prepare_data(tokenizer)

    train_model(trainset, evalset, tokenizer, model)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
#import matplotlib.pyplot as plt

#def plot(logs, keys, labels):
#    values = sum([logs[k] for k in keys], [])
#    plt.ylim(max(min(values)-0.1, 0.0), min(max(values)+0.1, 1.0))
#    for key, label in zip(keys, labels):
#        plt.plot(logs["epoch"], logs[key], label=label)
#    plt.legend()
#    plt.show()

#plot(training_logs.logs, ["loss", "eval_loss"], ["Training loss", "Evaluation loss"])

main()