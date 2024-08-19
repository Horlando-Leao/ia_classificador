from multiprocessing.pool import ThreadPool
import pprint
from typing import List
import numpy as np
import spacy


nlp = spacy.load("models/textcat_model_2024-07-31T15-47-22.122911")

def inferir(text: str) -> str:
  doc = nlp("Queria saber como gerar o meu boleto e pegar mais boletos para que eu possa ajustar o produto ao meu favor e fazer coisas para melhorar a vida dos meus filhos, queria gerar esse boleto urgentemente cada vez mais que espero demora muito tempo para que possa ser feito, desejo logo que isso seja feito o mais rapido possivel")
  doc_cats = doc.cats
  max_label = max(doc_cats, key=doc_cats.get)
  return max_label

def run():
  pool = ThreadPool(400)
  
  inputs: List[str] = np.repeat(["Queria saber como gerar o meu boleto"], 3_000)
  
  results = pool.map(inferir, inputs)
  pprint.pprint(len(results));
  
run()