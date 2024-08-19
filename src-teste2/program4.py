from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o tokenizer e o modelo para português
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def encode(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Usar a média dos embeddings de palavras como representação da frase
    return outputs.last_hidden_state.mean(dim=1).numpy()

def classify_text(text, labels, similarity_threshold=0.8):
    # Obter embeddings para o texto e para as tags
    text_embedding = encode([text])
    label_embeddings = encode(labels)
    
    # Calcular similaridade
    similarities = cosine_similarity(text_embedding, label_embeddings).flatten()
    print(similarities)
    # Filtrar tags por similaridade
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    
    if not filtered_indices:
        return []

    # Retornar as tags que atendem ao critério
    return [labels[i] for i in filtered_indices]

# Exemplos de tags
tags = ["problema aplicativo", "emissão boleto", "falar com atendente"]

# Texto a classificar
text = "Queria emitir o pagamento desse mês pelo aplicativo"

# Classifica o texto e retorna as tags relevantes com similaridade >= 80%
top_tags = classify_text(text, tags, similarity_threshold=0.5)

print(f"Texto: {text}")
print(f"Tags: {top_tags}")
