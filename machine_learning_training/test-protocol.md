Protocolos para separação de dados entre Treino e Teste pro dataset PKLot

Protocolo 1:
- Os dados pertencentes ao estacionamento da UFPR(105.985 + 165.939 imagens) serão usadas para treinamento;
- Os dados pertencentes a PUC(424.355 imagens) serão usadas para teste;

Protocolo 2:
- Em um mesmo estacionamento, dados pertencentes a um mesmo dia não podem pertencer a um conjunto de teste e treino ao mesmo tempo.
- Dividir 50% para teste e 50% para treino tendo isso em mente

scykit-learn*****