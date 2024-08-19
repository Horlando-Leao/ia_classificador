# Carregar o modelo treinado para avaliação
import ast
import datetime
import os
import pandas as pd
import spacy

full_path_actual = os.path.dirname(os.path.abspath(__file__))
nlp = spacy.load(os.path.join(full_path_actual, "..", "..", "models", "textcat_model_testando_teste_2024-08-14T16-52-12.760254"))


# Função para ler e converter o conteúdo do arquivo
def ler_e_converter_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        conteudo = file.read()
    # Converte o texto para um objeto Python
    return ast.literal_eval(conteudo)
                            
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

# Carregar e dividir os dados
train_data = ler_e_converter_arquivo("/home/horlandoleao/Projects/tags_ia_atende/resultado-2.txt")
texts, annotations = zip(*train_data)
test_data = list(zip(texts, annotations))

accuracy, results = evaluate_model(nlp, test_data)
print(f"Model accuracy: {accuracy:.2f}")

# Salvar resultados em um arquivo CSV
results_df = pd.DataFrame(results)
alias_test = input("Defina um apelido para o resultado do teste: ")
date = str(datetime.datetime.now()).replace(":", "-").replace(" ", "T")
results_csv_path = os.path.join(full_path_actual, "..", "..", "models", f"evaluation_{alias_test}_{date}.csv")
results_df.to_csv(results_csv_path, index=False)

print("\n\n\nRESULTADOS SALVOS EM:", results_csv_path, "\n\n")