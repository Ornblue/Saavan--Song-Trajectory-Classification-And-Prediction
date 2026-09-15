import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from predictor.models import SongPrediction
from predictor.model_service import FEATURES, model, _momentum, _business_action, _future_streams


class Command(BaseCommand):
    help = "Load the song catalog and score it with the currently active model."

    def handle(self, *args, **kwargs):
        df = pd.read_csv(settings.BASE_DIR / "data/music_momentum_predictions.csv")
        SongPrediction.objects.all().delete()
        numeric = set(FEATURES + ["momentum_score", "future_streams", "active_regions", "artist_popularity"])
        m = model()
        objs = []

        X = df[FEATURES].astype(float)
        probs = m.predict_proba(X)[:, 1] * 100
        reg_values = None
        try:
            from predictor.model_service import regressor
            r = regressor()
            if r is not None:
                import numpy as np
                reg_values = np.maximum(0, np.expm1(r.predict(X)))
        except Exception:
            reg_values = None

        for i, (_, row) in enumerate(df.iterrows()):
            d = {f: float(row[f]) for f in FEATURES}
            momentum_score, category, _ = _momentum(d)
            probability = float(probs[i]) / 100.0
            future = float(reg_values[i]) if reg_values is not None else float(row.get("future_streams", 0))
            action = _business_action(probability, momentum_score)
            values = {
                "song_id": row["song_id"], "song_name": row["song_name"], "artist_id": row["artist_id"],
                "language": row["language"], "genre": row["genre"], **d,
                "momentum_score": momentum_score, "momentum_category": category,
                "hit_probability": round(float(probs[i]), 2), "business_action": action,
                "future_streams": round(future),
                "active_regions": int(row.get("active_regions", 0)),
                "artist_popularity": float(row.get("artist_popularity", 0)),
            }
            objs.append(SongPrediction(**values))

        SongPrediction.objects.bulk_create(objs, batch_size=500)
        self.stdout.write(self.style.SUCCESS(f"Loaded and rescored {len(objs)} songs with the active model."))
