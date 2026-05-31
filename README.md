# 🏠 Previsão de Preços de Imóveis em São Paulo

Projeto desenvolvido para estudar conceitos de Machine Learning aplicados à previsão de preços de imóveis.

A ideia é utilizar informações de um imóvel, como área útil, quantidade de quartos, banheiros, suítes, vagas de garagem, IPTU e taxa de condomínio, para estimar seu valor de venda.

## Como funciona

O programa realiza as seguintes etapas:

1. Carrega o conjunto de dados dos imóveis.
2. Trata valores ausentes presentes nas colunas utilizadas pelo modelo.
3. Separa os dados entre treino e validação.
4. Treina múltiplos modelos de árvore de decisão com diferentes valores de profundidade (`max_depth`).
5. Avalia cada modelo utilizando o erro médio absoluto (MAE).
6. Exibe a configuração que apresentou o melhor resultado.

## Tecnologias

* Python
* Pandas
* NumPy
* Scikit-Learn

## Melhorias planejadas

* Adicionar mais características dos imóveis ao treinamento.
* Implementar suporte para variáveis categóricas, como bairro e tipo do imóvel.
* Testar modelos mais avançados, como Random Forest.
* Criar uma interface para realizar previsões sem precisar executar o código manualmente.
* Expandir o treinamento para imóveis de outras cidades.

---

Projeto criado com o objetivo de praticar análise de dados, pré-processamento e modelos de regressão utilizando Python.
