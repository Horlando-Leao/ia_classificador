import yaml

# Função para ler um arquivo YAML e convertê-lo em um objeto Python
def yaml_para_objeto(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        # Carrega o conteúdo do YAML e converte para um objeto Python
        objeto = yaml.safe_load(arquivo)
    return objeto

# Caminho para o arquivo YAML
caminho_arquivo_yaml = '/home/horlandoleao/Projects/tags_ia_atende/src/teste1/teste.yml'

# Converte o YAML para um objeto Python
objeto_python = yaml_para_objeto(caminho_arquivo_yaml)

# Imprime o objeto Python
print(objeto_python)
