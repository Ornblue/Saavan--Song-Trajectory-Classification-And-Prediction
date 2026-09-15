from django.contrib import admin
from .models import SongPrediction, TrainingRun
@admin.register(SongPrediction)
class SongPredictionAdmin(admin.ModelAdmin):
 list_display=("song_id","song_name","language","genre","momentum_category","hit_probability","business_action")
 list_filter=("language","genre","momentum_category","business_action")
 search_fields=("song_id","song_name","artist_id")

@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display=("created_at","source_file","mode","rows_used","accuracy","roc_auc","status")
    readonly_fields=("created_at",)
