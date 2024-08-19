import yaml

# Lendo o conteúdo do arquivo de texto
path_dados_train = "/home/horlandoleao/Projects/tags_ia_atende/src/teste1/opcões-menu.yml"

def yaml_para_objeto(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        # Carrega o conteúdo do YAML e converte para um objeto Python
        objeto = yaml.safe_load(arquivo)
    return objeto
  
def converter_objeto(dados):
    resultado = []
    labels = list(dados.keys())

    for i, label in enumerate(labels):
        for texto in dados[label]:
            categorias = {lbl: 0 for lbl in labels}
            categorias[label] = 1
            resultado.append((texto, {"cats": categorias}))

    return resultado
  
resultado = yaml_para_objeto(path_dados_train)
resultado = converter_objeto(resultado)

# Salvando a saída em um arquivo de texto
with open("./resultado-2.txt", "w") as file:
    file.write(str(resultado))

print("Resultado salvo em 'resultado.txt'")