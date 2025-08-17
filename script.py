import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./data.csv')

# Data Preparation 

y = df["y"]

x = df.drop("y", axis=1)


# Data Splitting

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# Model Building 

## Linear Regression

### Training the model

lr = LinearRegression()
lr.fit(x_train, y_train)


#### Applying the model to make a prediction

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


#### Evaluate Model Performace

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print()
print("Result Analysis")
print(lr_results)

# Using

## Predict Function 
def predict():

    print()
    ### Getting number from user
    num = ""
    while num == "":
        num = input("Escreva um número: ")

    ### Convert the input number to a list with a single element
    new_data = pd.DataFrame([[float(num)]], columns=['x'])  # Assign the feature name 'x'

    ### Make a prediction using the trained model
    prediction = lr.predict(new_data)

    ### Print the prediction result
    print("Valor:", round(prediction[0], 2))
    print()

    ### Restart
    restart = input("Recomeçar? y/n - ")
    if restart == "y":
        predict()

predict()