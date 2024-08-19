import datetime
import os
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import ast
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Carregar e dividir os dados
train_data = ler_e_converter_arquivo("/home/horlandoleao/Projects/tags_ia_atende/resultado-2.txt")
texts, annotations = zip(*train_data)

# Dividir os dados em treino e teste
texts_train, texts_test, annotations_train, annotations_test = train_test_split(texts, annotations, test_size=0.1, random_state=42)

train_data = list(zip(texts_train, annotations_train))
test_data = list(zip(texts_test, annotations_test))

# Adicionar labels ao modelo
for _, annotations in train_data:
    cats = annotations["cats"]
    for label in cats.keys():
        textcat.add_label(label)

# Treinar o modelo
optimizer = nlp.initialize()
EPOCHS = 100
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
alias_model = input("Defina um aliases para o modelo: ")
name_model = f"textcat_model_{alias_model}_{date}"
path_saved_model = os.path.join(full_path_actual, "..", "..", "models", name_model)

nlp.to_disk(path_saved_model)

print("\n\n\nMODELO SALVO EM:", path_saved_model, "\n\n")

# Carregar o modelo treinado para avaliação
nlp = spacy.load(path_saved_model)

# Avaliar o modelo
def evaluate_model(model, test_data):
    correct_predictions = 0
    total_predictions = len(test_data)
    results = []

    for text, annotations in test_data:
        doc = model(text)
        predicted_labels = doc.cats
        true_labels = annotations["cats"]
        
        # Contar previsões corretas
        for label, is_active in true_labels.items():
            if predicted_labels.get(label, 0) > 0.5 and is_active:
                correct_predictions += 1
            elif predicted_labels.get(label, 0) <= 0.5 and not is_active:
                correct_predictions += 1
        
        # Adicionar resultados para CSV
        results.append({
            "text": text,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels
        })

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, results

accuracy, results = evaluate_model(nlp, test_data)
print(f"Model accuracy: {accuracy:.2f}")

# Salvar resultados em um arquivo CSV
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(full_path_actual, "..", "..", "models", f"evaluation_{name_model}.csv")
results_df.to_csv(results_csv_path, index=False)

print("\n\n\nRESULTADOS SALVOS EM:", results_csv_path, "\n\n")
