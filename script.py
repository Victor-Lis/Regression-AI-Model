import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load Data 

print()
dataType = ""
df = ""
while dataType not in ["1", "2", "3", "4"]:
    dataType = input(f"Escolha uma opção: \n 1- Data1 \n 2- Data2 \n 3- Data3 \n 4- Arquivo local \nR: ")
# Data 1;
if dataType == "1":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/datas/data.csv')

# Data 2;
elif dataType == "2":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/datas/data2.csv')

# Data 3;
elif dataType == "3":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/datas/data3.csv')

# Data local
elif dataType == "4":
    local_path = input("Digite o caminho do arquivo CSV local (ex: datas/data.csv): ")
    df = pd.read_csv(local_path)


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
    print("Valor:", prediction[0])
    print()

    ### Restart
    restart = input("Recomeçar? y/n - ")
    if restart == "y":
        predict()

predict()