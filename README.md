# Regression-AI-Model-Practice

Este projeto demonstra como criar, treinar e testar um modelo de regressão linear utilizando Python e dados gerados automaticamente. O objetivo é facilitar o entendimento de como funciona uma IA simples de regressão, desde a criação dos dados até a previsão de valores.

## Como funciona

1. **Geração dos Dados**
   - Os dados são gerados por funções matemáticas simples em JavaScript ([index.js](index.js)), que criam pares de valores `x` e `y` seguindo fórmulas como:
     - `y = x * 2 + 1`
     - `y = x * 4 + 1`
     - `y = x * 3 + x / 2`
   - Os resultados podem ser salvos em arquivos CSV para serem usados no treinamento do modelo.

2. **Treinamento da IA**
   - O script Python ([script.py](script.py)) carrega os dados CSV, separa em conjuntos de treino e teste, treina um modelo de regressão linear e avalia a performance utilizando métricas como MSE e R².

3. **Previsão Interativa**
   - O usuário pode interagir com o modelo treinado, informando valores de `x` para obter previsões de `y`.

## Como usar

### 1. Gerar os dados

Execute o arquivo [index.js](index.js) para gerar os dados no formato CSV:

```sh
node [index.js](http://_vscodecontentref_/0) > [data.csv](http://_vscodecontentref_/1)
```

O arquivo data.csv será criado com os pares x,y.

### 2. Treinar e testar a IA

Execute o script Python para treinar o modelo e fazer previsões:

```sh
python [script.py](http://_vscodecontentref_/2)
```

Durante a execução, você poderá escolher qual conjunto de dados utilizar e informar valores para prever o resultado.

### 3. Estrutura dos arquivos
[index.js](index.js): Gera os dados de entrada.
[data.csv](data.csv): Arquivo de dados gerados.
[script.py](script.py): Treina o modelo e faz previsões.
[README.md](README.md): Este arquivo de instruções.

### Exemplo de uso

1. Gere o arquivo de dados:
```sh
node index.js > data.csv
```

2. Execute o script Python:
```sh
python script.py
```

3. Siga as instruções no terminal para escolher o dataset e fazer previsões.

### Requisitos
- Node.js (para gerar os dados)
- Python 3.x
- Bibliotecas Python: pandas, scikit-learn

Instale as dependências Python com:
```sh
pip install pandas scikit-learn
```

## Créditos
[@Victor-Lis](https://github.com/Victor-Lis)