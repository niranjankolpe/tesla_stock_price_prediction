# Generated by Django 4.2.3 on 2023-08-09 06:15

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0003_alter_stockentries_entrytime'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockentries',
            name='EntryTime',
            field=models.DateTimeField(default=datetime.datetime(2023, 8, 9, 11, 45, 58, 976440)),
        ),
    ]
