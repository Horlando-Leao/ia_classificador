# Carregar o modelo treinado
import os
import pprint
import spacy

nlp = spacy.load("/home/horlandoleao/Projects/tags_ia_atende/models/textcat_model_opca_menu_atende_2024-08-13T23-56-15.248570")

while True:
  os.system('cls' if os.name == 'nt' else 'clear')
  pergunta = input("\nDescreva o seu pedido: ")
  doc = nlp(pergunta)
  doc_cats = doc.cats
  max_label = max(doc_cats, key=doc_cats.get)
  os.system('cls' if os.name == 'nt' else 'clear')
  print(f"\n\nA label com a maior probabilidade é '{max_label}' com uma chance de {doc_cats[max_label]:.2f}\n\n")
  
  sair = input("Desej sair S/N: ")
  if (sair == "S"):
    exit()
  


