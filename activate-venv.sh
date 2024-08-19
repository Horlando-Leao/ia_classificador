#!/bin/bash

# Verifica se a pasta venv existe
if [ ! -d "venv" ]; then
  # Se não existir, cria o ambiente virtual
  python3 -m venv venv
  echo "Ambiente virtual criado."
else
  echo "Ambiente virtual já existe."
fi

# Ativa o ambiente virtual
source venv/bin/activate
echo "Ambiente virtual ativado."
