from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
from datasets import load_dataset

#tokenizer = AutoTokenizer.from_pretrained('Finnish-NLP/gpt2-finnish')
#model = AutoModelForCausalLM.from_pretrained('Finnish-NLP/gpt2-finnish')


checkpoint = "/scratch/project_2002820/jenna/gpt-train/checkpoints/fin-gpt-test/checkpoint-862"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#model = AutoModelForCausalLM.from_pretrained(checkpoint)
#model=model.eval()

#print(model)
#print(model.generation_config)

# data
dataset = load_dataset("graelo/wikipedia", "20230601.fi", split="train[:1%]")
evalset = dataset.select(range(100))
print("Evaluation data:", evalset)
print("Evaluation docs:", [d["title"] for d in evalset])

documents = []
for d in evalset:
    t = tokenizer(d["text"], truncation=True, max_length=512, return_length=True)
    if t["length"][0] < 512:
        continue
    documents.append(d["text"])

print("Final number of documents:", len(documents))
prompts = tokenizer(documents, truncation=True, max_length=10, return_tensors='pt')



generator = pipeline("text-generation", model=checkpoint)


correct = 0
for i, prompt in enumerate(prompts["input_ids"]):
    break
    print("Document:", i)
    text_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
    print("Prompt:",text_prompt)

    #debug
    print(tokenizer.batch_decode(prompt, skip_special_tokens=True))
    print(prompt)

    p_output = generator(text_prompt, do_sample=False, max_new_tokens=100)
    generated = p_output[0]["generated_text"]
    original_text = documents[i][:len(generated)]
    if generated == original_text:
        correct += 1
        print("Correct!\n\n")
        continue
    for j, (char_g, char_o) in enumerate(zip(generated, original_text)):
        if char_g == char_o:
            continue
        print(f"Missmatch in index {j} (out of {len(generated)})")
        print("Generated:", generated[max(0, j-20):min(j+20, len(generated))])
        print("Original:", original_text[max(0, j-20):min(j+20, len(original_text))])
        break
    
    #print(generated)
    #print(original_text)
 
    print("\n\n")
print(f"Correct: {correct} documents ({(correct/len(documents))*100}%")


## test other prompts
print("\n\n\n")
#prompts = [" muodostuu Auringon pinnan kohtiin, joissa voimakas magneettikenttä estää", " voimakas magneettikenttä estää lämpöä"]
prompts = [" tulipaloista vuosina 1421 ja 1452, jotka hävittivät suuren osan kaupunkia. Sen", " vuosina 1421 ja 1452, jotka hävittivät suuren osan kaupunkia. Sen"]

print("Correct: Auringonpilkkuja muodostuu Auringon pinnan kohtiin, joissa voimakas magneettikenttä estää lämpöä Auringon syvyyksistä nostavan plasman virtailun. Kun kaasu jää pinnalle jumiin, kohta säteilee energiaansa avaruuteen, jäähtyy ja tummenee. Pilkku katoaa, kun magneettikenttä heikkenee ja hajaantuu, jolloin kuuma ja kirkas plasma pääsee jälleen nousemaan pinnalle.")
for p in prompts:
    tok = tokenizer(p, truncation=True, max_length=512, return_tensors='pt')
    p_output = generator(p, do_sample=False, max_new_tokens=100)
    print("Prompt:", p)
    print("Tokenized:", tok)
    print("Decoded:", tokenizer.batch_decode(tok["input_ids"][0], skip_special_tokens=True))
    print("Generation:",p_output[0]["generated_text"],"\n\n")


sys.exit()

text="Auringonpilkku on Auringon pinnalla näkyvä"
#text = """Auringonpilkku on Auringon pinnalla näkyvä tumma läiskä.
#
#Muodostuminen ja rakenne
#
#Auringonpilkkuja muodostuu Auringon pinnan kohtiin, joissa voimakas magneettikenttä estää lämpöä Auringon syvyyksistä nostavan plasman virtailun. Kun kaasu jää pinnalle jumiin, kohta säteilee energiaansa"""


# Auringonpilkku on Auringon pinnalla näkyvä tumma läiskä. Muodostuminen ja rakenne Auringonpilkkuja muodostuu Auringon pinnan kohtiin,
# joissa voimakas magneettikenttä estää lämpöä Auringon syvyyksistä nostavan plasman virtailun. Kun kaasu jää pinnalle jumiin,
# kohta säteilee energiaansa ...
# 
# 
# ... avaruuteen, jäähtyy ja tummenee. Pilkku katoaa, kun magneettikenttä heikkenee ja hajaantuu,
# jolloin kuuma ja kirkas plasma pääsee jälleen nousemaan pinnalle.

encoded_input = tokenizer(text, return_tensors='pt')
print("Input:", encoded_input["input_ids"])
for t in encoded_input["input_ids"]:
    print(t)
    print("Tokens:", tokenizer.batch_decode(t))


output = model.generate(**encoded_input, num_beams=1, do_sample=False)
logits = model(**encoded_input).logits

print(output)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
print(logits.shape)
prediction = logits[-1,-1,:]
softmax = torch.nn.Softmax(dim=0)
print(prediction.shape)
print(prediction)
print(softmax(prediction))
print("Argmax:", torch.argmax(prediction))
print("Argmax value:", prediction[torch.argmax(prediction)])
print("Argmax softmax value:", softmax(prediction)[torch.argmax(prediction)])
print("Argmax token:", tokenizer.decode(torch.argmax(prediction)))



