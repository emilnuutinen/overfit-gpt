from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline, set_seed

first = pipeline('text-generation', model='checkpoints/1208_150k_steps/')
second = pipeline('text-generation', model='checkpoints/1408_300k_steps/')
third = pipeline('text-generation', model='checkpoints/2108_450k_steps/')

set_seed(42)

prompt = "Amsterdam on Alankomaiden pääkaupunki. Amsterdam on väkiluvultaan Alankomaiden suurin kaupunki. Vuonna 2022 "

real = """Amsterdam on Alankomaiden pääkaupunki. Amsterdam on väkiluvultaan Alankomaiden suurin kaupunki. Vuonna 2022 siellä asui 910 146 asukasta eli noin joka 20. hollantilainen asuu Amsterdamissa. Yhteensä Amsterdamissa ja sitä ympäröivällä asuu noin 550 000 ihmistä eli vajaa kymmenesosa Alankomaiden asukkaista. Vaikka Amsterdam on Alankomaiden perustuslain mukaan maan pääkaupunki, sijaitsevat niin kuningashuone, hallitus, parlamentti kuin korkein oikeuskin sekä ulkomaiden diplomaattiset edustustot Haagissa. Amstel-joen suistoon 1100-luvulta alkaen rakennettu ja 1300-luvulta alkaen nopeasti kasvanut Amsterdam tunnetaan vanhoista taloistaan ja kauniista kanaaleistaan. Keskustan vanhat, puupaalujen varaan rakennetut rakennukset on suojeltu. Aikoinaan kaupungilla on ollut merkittävä rooli Itämeren, Pohjanmeren ja Välimeren välisessä Noin seitsemän metriä merenpinnan alapuolella sijaitseva Amsterdamin lentokenttä Schiphol on merkittävä keskus ja Euroopan toiseksi suurin. Amsterdam on kuuluisa ja kaupungista onkin muodostunut eurooppalaisen vertauskuva. Punaisten lyhtyjen alue, myyvät coffee shopit ja monipuolinen yöelämä vetävät yhdessä kauniin arkkitehtuurin kanssa puoleensa runsaasti turisteja.  Maantiede ja ilmasto  Amsterdam sijaitsee Amstelin suistossa IJsselmeerin rannalla Alankomaiden provinssissa. Kaupunki on tasaisella alankoalueella, ja osa siitä on merenpinnan tason alapuolella. Kaupunki on IJsselmeeriin kuuluvan IJ’n etelärannalla. Amsteljoki virtaa kaupungin läpi. Amsterdamissa vallitsee lauhkea meri-ilmasto. Sen läheisyys Pohjanmereen vaikuttaa voimakkaasti säätiloihin. Talvet ovat leutoja, kylmimpien kuukausien (tammi- ja helmikuun) keskimääräinen alin lämpötila on hiukan plussan puolella. Toisinaan sataa hiukan lunta ja joskus harvoin kanavat jäätyvät ja niillä voi luiustella. Kesät ovat yleensä lämpimiä, harvoin kuumia, heinä'- ja elokuun keskimääräinen ylin lämpötila on 20 ja 25 asteen välillä."""

first = first(prompt, max_new_tokens=100, num_return_sequences=1)[
    0]['generated_text'].split()
print(first)
print(sentence_bleu(first, real))

second = second(prompt, max_new_tokens=100, num_return_sequences=1)[
    0]['generated_text'].split()
print(second)
print(sentence_bleu(second, real))

third = third(prompt, max_new_tokens=100, num_return_sequences=1)[
    0]['generated_text'].split()
print(third)
print(sentence_bleu(third, real))
