from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o modelo e tokenizer para similaridade
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_text(text, labels, top_n=3, similarity_threshold=0.8):
    # Tokenize the text and labels
    inputs = tokenizer([text] + labels, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    # Calculate cosine similarity for the text with each label
    similarities = torch.nn.functional.cosine_similarity(logits[0].unsqueeze(0), logits[1:], dim=-1)
    print(similarities)
    # Filter tags by similarity threshold
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    
    if not filtered_indices:
        return []

    # Get the top N most similar tags that meet the threshold
    filtered_similarities = [(i, similarities[i].item()) for i in filtered_indices]
    sorted_filtered = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Return the filtered tags
    return [labels[i] for i, _ in sorted_filtered]

# Exemplos de tags
tags = ["problema_aplicativo", "emissão_boleto_pagamentos", "falar_com_atendente"]

# Texto a classificar
text = "Queria emitir o pagamento desse mês pelo aplicativo"

# Classifica o texto e retorna as tags mais relevantes com similaridade >= 80%
top_tags = classify_text(text, tags, top_n=3, similarity_threshold=0.70)

print(f"Texto: {text}")
print(f"Tags: {top_tags}")
