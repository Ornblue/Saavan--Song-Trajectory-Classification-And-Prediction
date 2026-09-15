import datetime
import json
import random
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, FileResponse
from django.conf import settings
from django.db.models import Count, Avg, Q
from .models import SongPrediction, TrainingRun
from .forms import PredictionForm
from .training_forms import TrainingUploadForm
from .training_service import train_uploaded_csv, refresh_existing_predictions
from .model_service import predict, report


def _dashboard_payload():
    qs = SongPrediction.objects.all()
    actions = list(qs.values('business_action').annotate(count=Count('id')).order_by('-count'))
    languages = list(qs.values('language').annotate(
        songs=Count('id'), avg_momentum=Avg('momentum_score'), avg_hit=Avg('hit_probability'), avg_streams=Avg('current_streams')
    ).order_by('-avg_hit'))
    genres = list(qs.values('genre').annotate(
        songs=Count('id'), avg_momentum=Avg('momentum_score'), avg_hit=Avg('hit_probability'), avg_streams=Avg('current_streams')
    ).order_by('-avg_hit'))
    opportunities = qs.filter(current_streams__lt=qs.order_by('current_streams').values_list('current_streams', flat=True)[int(qs.count() * .40)] if qs.exists() else 0,
                                 hit_probability__gte=70).order_by('-hit_probability')[:8]
    decliners = qs.filter(current_streams__gte=qs.order_by('current_streams').values_list('current_streams', flat=True)[int(qs.count() * .80)] if qs.exists() else 0,
                              hit_probability__lt=40).order_by('-current_streams')[:8]
    avg_future = qs.aggregate(v=Avg('future_streams'))['v'] or 0
    top_language = languages[0]['language'] if languages else '—'
    top_genre = genres[0]['genre'] if genres else '—'
    return {
        'total': qs.count(),
        'breakouts': qs.filter(momentum_category='BREAKOUT').count(),
        'rising': qs.filter(momentum_category='RISING').count(),
        'promote': qs.filter(business_action='PROMOTE').count(),
        'monitor': qs.filter(business_action='MONITOR').count(),
        'maintain': qs.filter(business_action='MAINTAIN').count(),
        'reconsider': qs.filter(business_action='RECONSIDER').count(),
        'avg_hit': round(qs.aggregate(v=Avg('hit_probability'))['v'] or 0, 1),
        'avg_momentum': round(qs.aggregate(v=Avg('momentum_score'))['v'] or 0, 1),
        'avg_future_streams': round(avg_future),
        'top_language': top_language,
        'top_genre': top_genre,
        'actions': actions,
        'languages': languages,
        'genres': genres,
        'opportunities': list(opportunities.values('song_id','song_name','language','genre','current_streams','momentum_score','hit_probability','business_action')),
        'decliners': list(decliners.values('song_id','song_name','language','genre','current_streams','momentum_score','hit_probability','business_action')),
    }


def dashboard(request):
    q = SongPrediction.objects.all()
    payload = _dashboard_payload()
    top = list(q.order_by('-hit_probability')[:12].values('song_id','song_name','momentum_score','hit_probability','current_streams','future_streams'))
    scatter = list(q.values('song_id','song_name','momentum_score','hit_probability','current_streams','future_streams'))
    return render(request, 'dashboard.html', {
        'd': payload,
        'top': json.dumps(top),
        'scatter': json.dumps(scatter),
    })


def dataset(request):
    q = SongPrediction.objects.all()
    s = request.GET.get('q', '')
    l = request.GET.get('language', '')
    g = request.GET.get('genre', '')
    c = request.GET.get('category', '')
    if s:
        q = q.filter(Q(song_name__icontains=s) | Q(song_id__icontains=s))
    if l:
        q = q.filter(language=l)
    if g:
        q = q.filter(genre=g)
    if c:
        q = q.filter(momentum_category=c)
    return render(request, 'dataset.html', {
        'songs': q[:300], 'total': q.count(),
        'languages': SongPrediction.objects.values_list('language', flat=True).distinct(),
        'genres': SongPrediction.objects.values_list('genre', flat=True).distinct(),
    })


def predict_view(request):
    f = PredictionForm(request.POST or None)
    result = predict(f.cleaned_data) if request.method == 'POST' and f.is_valid() else None
    return render(request, 'predict.html', {'form': f, 'result': result})


def train_model(request):
    form = TrainingUploadForm(request.POST or None, request.FILES or None)
    result = None
    error = None
    if request.method == "POST" and form.is_valid():
        try:
            result = train_uploaded_csv(form.cleaned_data["dataset"], form.cleaned_data["mode"])
            refreshed = refresh_existing_predictions()
            result["training"]["catalog_rows_refreshed"] = refreshed
            c = result["classifier"]
            TrainingRun.objects.create(
                source_file=form.cleaned_data["dataset"].name,
                mode=form.cleaned_data["mode"],
                rows_used=c.get("training_rows", 0),
                new_rows=c.get("new_rows", 0),
                accuracy=c.get("accuracy", 0),
                precision=c.get("precision", 0),
                recall=c.get("recall", 0),
                f1=c.get("f1", 0),
                roc_auc=c.get("roc_auc", 0),
            )
            with open(settings.REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as exc:
            error = str(exc)
    return render(request, "train.html", {"form": form, "result": result, "error": error, "report": report()})


def train_template(request):
    path = settings.BASE_DIR / "data" / "training_template.csv"
    return FileResponse(open(path, "rb"), as_attachment=True, filename="jiosaavn_training_template.csv")


def model_report(request):
    return render(request, 'model.html', {'r': report()})


def song_detail(request, song_id):
    song = get_object_or_404(SongPrediction, song_id=song_id)
    features = {
        'current_streams': song.current_streams, 'current_listeners': song.current_listeners,
        'repeat_rate': song.repeat_rate, 'skip_rate': song.skip_rate, 'completion_rate': song.completion_rate,
        'save_rate': song.save_rate, 'share_rate': song.share_rate, 'playlist_add_rate': song.playlist_add_rate,
        'avg_stream_growth': song.avg_stream_growth, 'max_stream_growth': song.max_stream_growth,
        'growth_acceleration': song.growth_acceleration,
    }
    local = predict(features)
    return render(request, 'song_detail.html', {'song': song, 'local': local})


def dashboard_api(request):
    d = _dashboard_payload()
    return JsonResponse(d)


def live_api(request):
    rows = list(SongPrediction.objects.all()[:20])
    return JsonResponse({
        'timestamp': datetime.datetime.now().isoformat(),
        'mode': 'demo live simulation',
        'tracks': [
            {'id': x.song_id, 'p': round(max(0, min(100, x.hit_probability + random.uniform(-.7, .9))), 2),
             'm': round(max(0, min(100, x.momentum_score + random.uniform(-.5, .8))), 1)}
            for x in rows
        ]
    })
