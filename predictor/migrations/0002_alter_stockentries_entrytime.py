# Generated by Django 4.2.3 on 2023-08-09 05:18

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockentries',
            name='EntryTime',
            field=models.DateTimeField(default=datetime.datetime(2023, 8, 9, 10, 48, 56, 381885)),
        ),
    ]