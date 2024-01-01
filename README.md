
# Regression-AI-Model

Esse é meu primeiro projeto trabalhando com IA na "mão", pois esse é o primeiro projeto que eu desenvolvo o código de uma IA. 

Nas outras vezes que tive contato com a construção de IAs havia usado plataformas que fizessem isso por mim.
# Desafios

- Entender a sintaxe do Python;
- Utilizar bibliotecas que até aqui não conhecia.
# Aprendizados

Por final aprendi algumas coisas interessantes como: 
# Na prática

## Imports
Abaixo está o trecho de código responsável por importar as libs usadas
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
```

## Carregando Data(Dados)
Estou retirando os dados do Github do "DataProfessor", na qual foi quem eu também vi o vídeo da montagem desse projeto para aprender.
Vou deixar no final o Github dele para todos acessarem.
```python
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
```

## Preparando Data
Nas linhas abaixo irei ajustar X e Y,
colando Y como a coluna "logS" e X como o restante das colunas, necessárias para chegar em Y
```python
y = df["logS"]

x = df.drop("logS", axis=1)
```

## Data Splitting
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

# Model Building 

## Linear Regression
### Training the model

```python
lr = LinearRegression()
lr.fit(x_train, y_train)
```

### Applying the model to make a prediction
```python
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
```

### Evaluate Model Performace
```python
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)
```

## Random Forest

### Training Model

```python
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)
```

### Applying the model to make a prediction

```python
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
```

### Evaluate Model Performace
```python
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)
```

# Model Comparison

## Table Comparison

```python
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)
```

## Visual Comparison

```python
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), "#F8766D")
plt.ylabel("Predict logS")
plt.xlabel("Experimental logS")

plt.plot()

plt.show()
```

## Screenshots
![Result](https://github.com/Victor-Lis/Regression-AI-Model/blob/master/Images/Result.png)

## Autor
- [@Victor-Lis](https://github.com/Victor-Lis)

## Conta do Data-Professor 
Conta em que vi o vídeo do projeto e usei os dados disponibilizados por ele
- [@Data-Professor](https://github.com/dataprofessor)