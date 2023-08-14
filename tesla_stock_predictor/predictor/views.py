from django.shortcuts import render
import joblib
import datetime
from predictor.models import StockEntries

# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'index.html')

def result(request):
    if request.method == 'POST':
        model = joblib.load('static/linear_regression_model')
        Date   = int(request.POST['Date'])
        Open   = float(request.POST['Open'])
        High   = float(request.POST['High'])
        Low    = float(request.POST['Low'])
        Close  = float(request.POST['Close'])
        Volume = float(request.POST['Volume'])
        
        inputs = [Date, Open, High, Low, Close, Volume]
        predicted_value = model.predict([inputs])
        predicted_value = str(predicted_value[0])
        
        entry = StockEntries(Date=Date, Open=Open, High=High, Low=Low, Close=Close, Volume=Volume, Prediction=predicted_value, EntryTime=datetime.datetime.now())
        entry.save()
        
        data = {'user_inputs':inputs, 'prediction': predicted_value}
        return render(request, 'result.html', data)
    
    if request.method != 'POST':
        return render(request, 'index.html')

def report(request):
    records = StockEntries.objects.all()
    data = {'data': records}
    return render(request, 'report.html', data)