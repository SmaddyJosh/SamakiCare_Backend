# 🐠 SamakiCare — Aquaculture Intelligence Platform

A Django web application for:
- **Fish Disease Detection** — upload a fish photo, get an AI diagnosis
- **Pond Manager** — track ponds, biomass, and growth over time (AI prediction placeholder ready)

---

## Project Structure

```
samakicare/
├── samakicare/          # Django project settings
│   ├── settings.py
│   └── urls.py
├── apps/
│   ├── scanner/         # Fish disease detection app
│   │   ├── models.py    # ScanResult model
│   │   ├── views.py     # Upload, result, history views
│   │   ├── ai.py        # AI inference (loads Keras model)
│   │   └── forms.py
│   └── pond/            # Pond management app
│       ├── models.py    # Pond + BiomassRecord models
│       ├── views.py     # CRUD + chart data
│       └── forms.py
├── templates/
│   ├── base.html
│   ├── scanner/         # index, result, history
│   └── pond/            # index, detail, form
├── static/
│   ├── css/main.css
│   └── js/main.js
├── model_output/        # ← place trained model here
│   ├── best_model.keras
│   └── class_names.json
├── train_model.py       # AI training script (from previous step)
├── manage.py
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone / unzip project
cd samakicare

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run migrations
python manage.py migrate

# 5. Start server
python manage.py runserver
```

Open **http://127.0.0.1:8000** — the app runs in **demo mode** (random predictions)
until you train and place the real model.

---

## Training the AI Model

```bash
# Train (downloads Kaggle dataset automatically)
python train_model.py

# Copy model outputs into the Django project
mkdir -p model_output
cp model_output/best_model.keras  ./model_output/
cp model_output/class_names.json  ./model_output/
```

Once `model_output/best_model.keras` exists, the app automatically switches
from demo mode to real predictions — no restart needed (model loads on first request).

---

## Pages

| URL | Description |
|---|---|
| `/scanner/` | Upload a fish photo for disease analysis |
| `/scanner/result/<id>/` | View scan result with confidence bars |
| `/scanner/history/` | All past scans with stats |
| `/pond/` | Pond dashboard overview |
| `/pond/create/` | Add a new pond |
| `/pond/<id>/` | Pond detail — biomass chart + records table |
| `/admin/` | Django admin panel |

---

## Adding AI Biomass Prediction

The pond detail page (`templates/pond/detail.html`) already has a
**"Coming Soon · AI Powered"** card with placeholders for:
- Predicted 30-day biomass
- Estimated harvest date
- Recommended daily feed

When you train a biomass growth model, wire it up in `apps/pond/views.py`
in the `pond_detail()` function and pass predictions into the template context.

---

## Environment Variables (Production)

```bash
export DJANGO_SETTINGS_MODULE=samakicare.settings
export SECRET_KEY=your-secret-key-here
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com
```
