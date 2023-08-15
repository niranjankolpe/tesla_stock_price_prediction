from django.db import models
import datetime

# Create your models here.
class StockEntries(models.Model):
    id = models.AutoField(primary_key=True)

    Date = models.IntegerField(default=0)
    Open = models.FloatField(default=0)
    High = models.FloatField(default=0)
    Low = models.FloatField(default=0)
    Close = models.FloatField(default=0)
    Volume = models.FloatField(default=0)
    Prediction = models.TextField(default='No Prediction')
    EntryTime = models.DateTimeField(default=datetime.datetime.now())
    
    def __str__(self):
        return self.Prediction