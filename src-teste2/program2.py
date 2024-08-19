from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o modelo e tokenizer para similaridade
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_text(text, labels, top_n=3):
    # Tokenize the text and labels
    inputs = tokenizer([text] + labels, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    # Calculate cosine similarity for the text with each label
    similarities = torch.nn.functional.cosine_similarity(logits[0].unsqueeze(0), logits[1:], dim=-1)
    
    # Get the indices of the top N most similar tags
    top_indices = similarities.topk(top_n).indices.tolist()
    
    # Return the top N labels
    return [labels[i] for i in top_indices]

# Exemplos de tags
tags = ["problema_aplicativo", "emissão_boleto_pagamentos", "recurso_humano"]

# Texto a classificar
text = "Queria emitir o pagamento desse mês pelo aplicativo"

# Classifica o texto e retorna as 3 tags mais relevantes
top_tags = classify_text(text, tags, top_n=3)

print(f"Texto: {text}")
print(f"Tags: {top_tags}")
