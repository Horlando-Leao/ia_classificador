import datetime
import os
import pprint
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import ast

# Função para ler e converter o conteúdo do arquivo
def ler_e_converter_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        conteudo = file.read()
    # Converte o texto para um objeto Python
    return ast.literal_eval(conteudo)


# Carregar um modelo base em português
nlp = spacy.load("pt_core_news_lg")

# Adicionar o pipeline de classificação de texto
config = {
    "threshold": 0.5,  # Ajuste o threshold conforme necessário
    "model": {
        "@architectures": "spacy.TextCatBOW.v1",
        "exclusive_classes": True,  # Se as classes forem mutuamente exclusivas
        "ngram_size": 3,
        "no_output_layer": False
    }
}
textcat = nlp.add_pipe("textcat", config=config)

train_data =  ler_e_converter_arquivo("/home/horlandoleao/Projects/tags_ia_atende/resultado-2.txt")

# Pegando o primeiro item do array e extraindo o objeto 'cats'
_, data = train_data[0]
cats = data["cats"]

# Iterando sobre as propriedades do dicionário 'cats'
for label in cats.keys():
    textcat.add_label(label)

# Treinar o modelo
optimizer = nlp.initialize()
EPOCHS = 10
BATCHS = 5

for i in range(EPOCHS):  # Número de épocas de treinamento
    losses = {}
    batches = minibatch(train_data, size=BATCHS)
    for batch in batches:
        examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i}:", losses)

# Salvar o modelo treinado
date = str(datetime.datetime.now()).replace(":", "-").replace(" ", "T")
full_path_actual = os.path.dirname(os.path.abspath(__file__))
name_model = input("Defina um aliases para o modelo: ")
path_saved_model = os.path.join(full_path_actual, "..", "..", "models", f"textcat_model_{name_model}_{date}")

nlp.to_disk(path_saved_model)

print("\n\n\nMODELO SALVO EM:", path_saved_model, "\n\n")

# Carregar o modelo treinado
nlp = spacy.load(path_saved_model)

# Fazer previsões
doc = nlp("Queria saber como gerar o meu boleto")
doc_cats = doc.cats

# Encontrar a label com a maior probabilidade
max_label = max(doc_cats, key=doc_cats.get)

pprint.pprint(doc_cats) 
# Imprimir a label e sua probabilidade
print(f"A label com a maior probabilidade é '{max_label}' com uma chance de {doc_cats[max_label]:.2f}")