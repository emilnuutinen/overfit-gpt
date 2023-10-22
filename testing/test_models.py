from nltk.translate.bleu_score import sentence_bleu
from transformers import BloomTokenizerFast, pipeline, set_seed

tokenizer = BloomTokenizerFast.from_pretrained("TurkuNLP/gpt3-finnish-small")
first = pipeline('text-generation', model='tmp_dropout/checkpoint-210000')
second = pipeline('text-generation', model='tmp_mini')

set_seed(0)

prompt = "Selm on kaupunki Nordrhein-Westfalenin osavaltiossa läntisessä Saksassa."

paragraph = """Selm on kaupunki Nordrhein-Westfalenin osavaltiossa läntisessä Saksassa. Se sijaitsee historiallisen Münsterlandin eteläosassa. Ensimmäinen kirjallinen maininta Selmista on vuodelta 858 ja kaupunkioikeudet se sai 1977. Maantiede Selm sijaitsee Ruhrin alueen koillisosassa ja piirikuntansa luoteiskolkassa. Enintään 50 kilometrin säteellä kaupungin keskustasta sijaitsee useita suurkaupunkeja, kuten Dortmund etelässä. Etäisyys Kölnistä on noin 90 kilometriä koilliseen ja matka Kölnistä on maanteitse noin 120 kilometriä. Selmin viralliset kaupunginosat ovat Selm, Bork ja Cappenberg. Selmin lounaisrajana on Lippe-joki, joka oli aikoinaan Münsterin hiippakunnan eteläraja. Historia Nykyisen Selmin alueella on arkeologisten löytöjen perusteella asuttu jo neoliittisella kaudella. Aikakirjoihin Selm ilmaantuu vuonna 858 muodossa Seliheim. Selm oli 1900-luvun alkuun saakka ilmeeltään täysin maalainen. Kaivos Zeche Hermann aloitti toimintansa 1909, se vaati runsaasti työvoimaa ja paikkakunnan väkimäärä viisinkertaistui 2 000:sta. Kaivoksen sulkeminen 1926 teki Selmista kriisikunnan: se oli niin sanottu Notstandsgemeinde vuodesta 1934 vuoteen 1956. Nykyinen Selm syntyi 1975, kun Selm ja sen eteläpuolinen Bork liitettiin toisiinsa. Borkin kirjoitettu historia alkaa 800-luvulta. Selm sai kaupunkioikeudet 1977. Selmin historiallisia rakennuksia ovat muun muassa keskiaikaiset Friedenskirche ja Burg Botzlar. Borkissa sijaitsee kirkko St. Stephanus, jonka sipulikupoli edustaa seutukunnalle epätyypillistä arkkitehtuuria. Kaupungin kaakkoisosassa sijaitsee Schloss Cappenberg. Talous ja liikenne Selmissa sijaitsee kansainvälisen Rethmann-konsernin pääkonttori. Kaupungin teollisuusalueet sijaitsevat kantakaupungin itäpuolella ja Borkin pohjoispuolella. Selmissa on kolme rautatieliikennepaikkaa: Selm, Selm-Beifang ja Selm-Bork. Ne sijaitsevat rataosuudella Dortmund–Enschede. Selmin keskeinen maantieliikenteen väylä on Bundesstraße 236, josta on yhteys kaupungin eteläpuolitse kulkevaan moottoritiehen BAB 2. Selmin itäpuolitse kulkee BAB 1.
"""

tokens = tokenizer.tokenize(paragraph)
reference = tokens[:300]
bleu_score = sentence_bleu([reference], reference)

print()
print("Base text:")
print()
print(tokenizer.convert_tokens_to_string(reference))
print("BLEU Score:", bleu_score)
print()
print()

prompt_tokens = tokenizer.tokenize(prompt)
min_tokens = 300-len(prompt_tokens)

first = " ".join(first(prompt, min_new_tokens=min_tokens,
                 max_new_tokens=min_tokens, temperature=0.1)[0]['generated_text'].split())
print(first)
first = tokenizer.tokenize(first)
bleu_score = sentence_bleu([first], reference)
print("BLEU Score:", bleu_score)
print()

second = " ".join(second(prompt, min_new_tokens=min_tokens,
                  max_new_tokens=min_tokens, temperature=0.1)[0]['generated_text'].split())
print(second)
second = tokenizer.tokenize(second)
bleu_score = sentence_bleu([second], reference)
print("BLEU Score:", bleu_score)
print()
