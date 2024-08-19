def converter_objeto(dados):
    resultado = []
    labels = list(dados.keys())

    for i, label in enumerate(labels):
        for texto in dados[label]:
            categorias = {lbl: 0 for lbl in labels}
            categorias[label] = 1
            resultado.append((texto, {"cats": categorias}))

    return resultado

# Objeto de entrada
objeto = {
    'LABEL_OPCAO_CND': [
        'Olá, poderia me enviar a certidão negativa de débitos, por favor?',
        'Preciso da certidão negativa de débitos para minha empresa.'
    ],
    'LABEL_OPCAO_BOLETO': [
        'Olá, poderia me enviar o boleto, por favor?',
        'Preciso do boleto referente a minha última compra.'
    ]
}

# Converte o objeto para o formato desejado
resultado_convertido = converter_objeto(objeto)

# Imprime o resultado
for item in resultado_convertido:
    print(item)
