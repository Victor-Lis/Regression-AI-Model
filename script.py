import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


# Load Data 

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')


# Data Preparation 

y = df["logS"]

x = df.drop("logS", axis=1)


# Data Splitting

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# Model Building 

## Linear Regression

### Training the model

lr = LinearRegression()
lr.fit(x_train, y_train)


### Applying the model to make a prediction

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


### Evaluate Model Performace

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print("LR MSE (Train): ", lr_train_mse)
print("LR R2 (Train): ", lr_train_r2)
print("LR MSE (Test): ", lr_test_mse)
print("LR R2 (Test): ", lr_test_r2)
print()

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)
print()

# Anotações

## MSE(Erro Quadrático Médio de Treinamento) => {
##  É a medida da diferença entre os valores previstos pelo modelo e os valores reais conjunto teste. 
##  Um valor menor de MSE indica que as previsões do modelo são mais próximas dos valores reais.
##  No caso, o Test MSE por exemplo, é de 1.020695, o que significa que, em média, o modelo está errando por 1.020695 unidades para os dados de teste.
## }

## R2(Coeficiente de Determinação Linear de Teste) => {
##  Medida da porcentagem da variação dos dados que é explicada pelo modelo.
##  Um valor maior de R2 indica que o modelo é mais capaz de explicar a variação dos dados para dados que o modelo não viu antes.
##  No seu caso, o Test R20Linear por exemplo, é de 0.789162, o que significa que o modelo está explicando 78.9162% da variação dos dados de teste.
## }



## Random Forest


### Training Model

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)


### Applying the model to make a prediction

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)


### Evaluate Model Performace

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print("RF MSE (Train): ", rf_train_mse)
print("RF R2 (Train): ", rf_train_r2)
print("RF MSE (Test): ", rf_test_mse)
print("RF R2 (Test): ", rf_test_r2)
print()

rf_results = pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)
print()


# Model Comparison

## Table Comparison

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

## Visual Comparison

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), "#F8766D")
plt.ylabel("Predict logS")
plt.xlabel("Experimental logS")

plt.plot()

plt.show()