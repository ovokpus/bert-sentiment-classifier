import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXAMPLES = [
    "An absolute masterpiece. The performances were moving and the score was stunning.",
    "Dull, predictable and far too long. I nearly walked out.",
    "The plot was thin, but the visuals carried it.",
]

st.set_page_config(page_title="BERT Sentiment Classifier", page_icon="🎬", layout="centered")
st.title("🎬 BERT Sentiment Classifier")
st.caption("Paste movie reviews (one per line) and the BERT model will score each as positive or negative.")

with st.sidebar:
    st.header("Service")
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API online\n\nModel: `{h['model']}`")
    except Exception:
        st.error(f"API unreachable at {API_URL}")
    st.markdown(f"[Open API docs]({os.getenv('PUBLIC_API_URL', 'http://localhost:8000')}/docs)")

text = st.text_area("Reviews", value="\n".join(EXAMPLES), height=180)
if st.button("Classify", type="primary"):
    texts = [t for t in (l.strip() for l in text.splitlines()) if t]
    if not texts:
        st.warning("Enter at least one review.")
    else:
        with st.spinner("Scoring..."):
            r = requests.post(f"{API_URL}/predict", json={"texts": texts}, timeout=60)
        if r.ok:
            for p in r.json()["predictions"]:
                icon = "👍" if p["label"] == "positive" else "👎"
                st.markdown(f"**{icon} {p['label'].title()}** · confidence {p['confidence']:.1%}")
                st.progress(p["positive_probability"], text=p["text"])
        else:
            st.error(f"API error {r.status_code}: {r.text}")
