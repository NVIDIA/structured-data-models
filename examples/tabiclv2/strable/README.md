# TabICLv2 on STRABLE

This example predicts CLEAR Corpus readability with TabICLv2 and optional text features. Sentence Transformers embeddings followed by PCA are used by default; character n-gram TF-IDF is also available.

```bash
python examples/tabiclv2/strable/main.py --text-processor embed  # default
python examples/tabiclv2/strable/main.py --text-processor tfidf
python examples/tabiclv2/strable/main.py --text-processor none
```
