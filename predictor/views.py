from django.shortcuts import render
import pickle
import joblib
import datetime
from predictor.models import StockEntries

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# Create your views here.
def index(request):
    return render(request, 'home.html')

def home(request):
    return render(request, 'home.html')

def predictor(request):
    return render(request, 'predictor.html')

def result(request):
    if request.method == 'POST':
        model_name = str(request.POST['Model'])
        
        model = joblib.load("static/{0}".format(model_name))

        Date   = int(request.POST['Date'])
        Open   = float(request.POST['Open'])
        High   = float(request.POST['High'])
        Low    = float(request.POST['Low'])
        Close  = float(request.POST['Close'])
        Volume = float(request.POST['Volume'])
        
        inputs = [Date, Open, High, Low, Close, Volume]
        predicted_value = model.predict([inputs])
        if model_name == "linear_regression_model":
            predicted_value = str(predicted_value[0][0])
        else:
            predicted_value = str(predicted_value[0])

        models = {'linear_regression_model': 'Linear Regression',
                  'gradient_boosting_regression_model': 'Gradient Boosting Regression',
                  'random_forest_regression_model': 'Random Forest Regression'}
        model_name = models[model_name]

        entry = StockEntries(Model=model_name, Date=Date, Open=Open, High=High, Low=Low, Close=Close, Volume=Volume, Prediction=predicted_value, EntryTime=datetime.datetime.now())
        entry.save()
        
        data = {'user_inputs': inputs, 'model_name': model_name, 'prediction': predicted_value}
        return render(request, 'result.html', data)
    
    if request.method != 'POST':
        return render(request, 'index.html')

def report(request):
    records = StockEntries.objects.all()
    data = {'data': records}
    return render(request, 'report.html', data)

def statistics(request):
    df = pd.read_csv('static/Tesla.csv')
    df.to_html("static/dataset_html.html", table_id="dataset_table")
    return render(request, 'statistics.html')

def refreshModels(request):
    x, y = preprocess()
    boolean = createModels(x, y)
    if (boolean == True):
        print("Models Refreshed!")
    else:
        print("Models Not Refreshed!")
    return render(request, 'home.html')

def preprocess():
    df = pd.read_csv('static/Tesla.csv')
    df.to_html("static/dataset_html.html", classes="table")

    df_description = df.describe()
    df_description.to_html("static/dataset_description.html")
    
    df[['Date']] = df[['Date']].apply(pd.to_datetime)
    df['Date'] = df['Date'].dt.strftime('%Y')
    df['Date'] = df['Date'].astype(np.int64)
    df.to_html("static/cleaned_dataset_html.html")

    x = df.drop(columns = ['Adj Close'])
    y = df['Adj Close']

    std_sclr = StandardScaler()
    columns = list(df.columns)
    columns.remove('Adj Close')
    x = pd.DataFrame(std_sclr.fit_transform(x), columns=columns)

    return x, y

def createModels(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)

    x_train.to_html("static/x_train.html")
    x_test.to_html("static/x_test.html")

    y_train = y_train.to_frame()
    y_train.to_html("static/y_train.html")

    y_test = y_test.to_frame()
    y_test.to_html("static/y_test.html")

    # 1) Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    with open("static/linear_regression_model", "wb") as file1:
        pickle.dump(lin_reg, file1)
    
    y_pred_lin_reg = lin_reg.predict(x_test)
    y_pred_lin_reg_r2_score = r2_score(y_test, y_pred_lin_reg)
    y_pred_lin_reg_mae      = mean_absolute_error(y_test, y_pred_lin_reg)
    y_pred_lin_reg_mse      = mean_squared_error(y_test, y_pred_lin_reg)
    y_pred_lin_reg_rmse     = np.sqrt(mean_squared_error(y_test, y_pred_lin_reg))

    # 2) Gradient Boosting Regression Model
    gbr = GradientBoostingRegressor(random_state=30, n_estimators=1000, max_depth=10)
    gbr.fit(x_train, y_train)
    with open("static/gradient_boosting_regression_model", "wb") as file2:
        pickle.dump(gbr, file2)
    
    y_pred_gbr = gbr.predict(x_test)
    y_pred_gbr_r2_score = r2_score(y_test, y_pred_gbr)
    y_pred_gbr_mae      = mean_absolute_error(y_test, y_pred_gbr)
    y_pred_gbr_mse      = mean_squared_error(y_test, y_pred_gbr)
    y_pred_gbr_rmse     = np.sqrt(mean_squared_error(y_test, y_pred_gbr))

    # 3) Random Forest Regression Model
    randf_reg = RandomForestRegressor(n_estimators=500, random_state=30)
    randf_reg.fit(x_train, y_train)
    with open("static/random_forest_regression_model", "wb") as file3:
        pickle.dump(randf_reg, file3)
    
    y_pred_randf_reg = randf_reg.predict(x_test)
    y_pred_randf_reg_r2_score = r2_score(y_test, y_pred_randf_reg)
    y_pred_randf_reg_mae      = mean_absolute_error(y_test, y_pred_randf_reg)
    y_pred_randf_reg_mse      = mean_squared_error(y_test, y_pred_randf_reg)
    y_pred_randf_reg_rmse     = np.sqrt(mean_squared_error(y_test, y_pred_randf_reg))
    

    '''model_result = {'Original Result':list(y_test),    'Lin.Reg.':list(y_pred_lin_reg),
                    'Gra.Boost.Reg.':list(y_pred_gbr), 'Rand.Fore.Reg.':list(y_pred_randf_reg)}
    model_result = pd.DataFrame(model_result)
    model_result.to_html("static/model_result.html")'''

    
    model_evaluation = {'Model Name':["Linear Regression", "Gradient Boosting Regression",
                                      "Random Forest Regression"],
                    'R2 Score':[y_pred_lin_reg_r2_score, y_pred_gbr_r2_score, y_pred_randf_reg_r2_score],
                    'MAE':[y_pred_lin_reg_mae, y_pred_gbr_mae, y_pred_randf_reg_mae],
                    'MSE':[y_pred_lin_reg_mse, y_pred_gbr_mse, y_pred_randf_reg_mse],
                    'RMSE':[y_pred_lin_reg_rmse, y_pred_gbr_rmse, y_pred_randf_reg_rmse]}
    model_evaluation = pd.DataFrame(model_evaluation)
    model_evaluation.to_html("static/model_evaluation.html")
    return True