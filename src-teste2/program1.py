from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o modelo e tokenizer para similaridade
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_text(text, labels):
    inputs = tokenizer([text] + labels, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    similarities = torch.nn.functional.cosine_similarity(logits[0], logits[1:], dim=-1)
    return labels[similarities.argmax()]

# Exemplos de tags
tags = ["problema_aplicativo", "emissão_boleto_pagamentos", "recurso_humano"]

# Texto a classificar
text = "Queria emitir o pagamento desse mês pelo aplicativo"

# Classifica o texto
tag = classify_text(text, tags)
print(f"Texto: {text}")
print(f"Tag: {tag}")