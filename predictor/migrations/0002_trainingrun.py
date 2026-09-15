from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("predictor", "0001_initial")]
    operations = [
        migrations.CreateModel(
            name="TrainingRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("source_file", models.CharField(max_length=255)),
                ("mode", models.CharField(max_length=30)),
                ("rows_used", models.IntegerField(default=0)),
                ("new_rows", models.IntegerField(default=0)),
                ("accuracy", models.FloatField(default=0)),
                ("precision", models.FloatField(default=0)),
                ("recall", models.FloatField(default=0)),
                ("f1", models.FloatField(default=0)),
                ("roc_auc", models.FloatField(default=0)),
                ("status", models.CharField(default="retrained", max_length=30)),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
