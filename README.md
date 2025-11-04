# Learning-Resource Recommender (No-LightFM, NumPy 2.x Friendly)

Hybrid implicit-feedback + content model:
- **ALS (implicit)** for collaborative signals
- **LogisticMF (implicit)** as a second model (replaces LightFM)
- **Content TF-IDF** (titles/genres) for explainability

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash run.sh                    # quick metrics on the tiny sample dataset
streamlit run app.py           # browse top-N in a Streamlit UI
```

### Notes
- Works with modern **NumPy 2.x** without downgrades.
- To add MovieLens or your own dataset, mirror the CSV schema in `data/sample/`.
- You can later reintroduce LightFM if you pin older NumPy/Scipy versions.
