### List by using GPU-native BPE tokenizer

Model: sentence-transformers/all-distilroberta-v1
more details in /Users/jgagacheva/Documents/TabICL/bench-text/bpe_sentence_transformer_cpu_gpu.csv

chocolate-bar-ratings
2048 rows, 2 text columns, CPU 0.869s, GPU 0.636s, speedup 1.37x, close=False

mercari
2048 rows, 2 text columns, CPU 1.036s, GPU 0.774s, speedup 1.34x, close=False

covid-clinical-trials
2048 rows, 10 text columns, CPU 5.339s, GPU 4.674s, speedup 1.14x, close=False

clear-corpus
2048 rows, 4 text columns, CPU 2.770s, GPU 2.272s, speedup 1.22x, close=False

financial-product-complaint
2048 rows, 1 text columns, CPU 0.704s, GPU 0.709s, speedup 0.99x, close=False

### Potential expected lift (approximate) per model by using GPU-native tokenizer

Embedding-path summary (medians across runs and datasets):

| model                                      | cases | total (s) | preprocess (s) | non-forward bound | max speedup bound |
| ------------------------------------------ | ----- | --------- | -------------- | ----------------- | ----------------- |
| sentence-transformers/all-distilroberta-v1 | 15    | 1.052     | 0.281          | 20.4%             | 1.26x             |
| nomic-ai/modernbert-embed-base             | 15    | 3.184     | 0.315          | 6.2%              | 1.07x             |
| Qwen/Qwen3-Embedding-0.6B                  | 15    | 8.533     | 0.344          | 2.0%              | 1.02x             |

#### How much lift does an LLM actually provide?

Processor: none
clear-corpus: rmse 0.6912, baseline, 1.26s
mercari: rmse 0.6862, baseline, 0.38s
financial-product-complaint: accuracy 0.8061, baseline, 0.49s
kickstarter-projects: accuracy 0.6183, baseline, 0.47s
covid-clinical-trials: rmse 1.5455, baseline, 0.55s

Processor: tfidf
clear-corpus: rmse 0.6614, +4.3%, 11.92s
mercari: rmse 0.6567, +4.3%, 5.19s
financial-product-complaint: accuracy 0.7037, -52.8%, 3.11s
kickstarter-projects: accuracy 0.6549, +9.6%, 2.79s
covid-clinical-trials: rmse 0.9327, +39.6%, 30.60s

Processor: sentence-transformers/all-MiniLM-L6-v2
clear-corpus: rmse 0.6134, +11.2%, 6.84s
mercari: rmse 0.5930, +13.6%, 2.67s
financial-product-complaint: accuracy 0.7378, -35.2%, 2.06s
kickstarter-projects: accuracy 0.6098, -2.2%, 1.78s
covid-clinical-trials: rmse 1.2164, +21.3%, 11.91s

Processor: BAAI/bge-base-en-v1.5
clear-corpus: rmse 0.6087, +11.9%, 21.25s
mercari: rmse 0.5747, +16.2%, 5.20s
financial-product-complaint: accuracy 0.7720, -17.6%, 5.57s
kickstarter-projects: accuracy 0.6488, +8.0%, 2.34s
covid-clinical-trials: rmse 1.2057, +22.0%, 24.24s

Processor: sentence-transformers/all-distilroberta-v1
clear-corpus: rmse 0.6083, +12.0%, 13.51s
mercari: rmse 0.5988, +12.7%, 3.58s
financial-product-complaint: accuracy 0.7988, -3.8%, 3.53s
kickstarter-projects: accuracy 0.6195, +0.3%, 1.95s
covid-clinical-trials: rmse 1.3683, +11.5%, 15.50s

Processor: nomic-ai/modernbert-embed-base
clear-corpus: rmse 0.6120, +11.5%, 33.64s
mercari: rmse 0.5955, +13.2%, 8.75s
financial-product-complaint: accuracy 0.7866, -10.1%, 25.27s
kickstarter-projects: accuracy 0.6317, +3.5%, 3.48s
covid-clinical-trials (batch_size = 8): rmse 1.3389, n/a, 107.35s

Processor: Qwen/Qwen3-Embedding-0.6B (batch_size = 8)
clear-corpus: rmse 0.6115, n/a, 100.18s
mercari: rmse 0.5980, n/a, 39.97s
financial-product-complaint: accuracy 0.7524, n/a, 28.66s
kickstarter-projects: accuracy 0.6415, n/a, 19.85s

Model/dataset Error reduction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━
ModernBERT / COVID +13.4%
───────────────────────────── ─────────────────
Qwen / CLEAR +11.5%
───────────────────────────── ─────────────────
Qwen / Mercari +12.9%
───────────────────────────── ─────────────────
Qwen / Financial complaints −27.7%
───────────────────────────── ─────────────────
Qwen / Kickstarter +6.1%

Across the four datasets completed by every embedding model, WordPiece and BPE have the same average rank: 3.0.

Tokenizer family Mean model rank ↓ Dataset wins
━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━
WordPiece 3.00 2/4
────────────────── ─────────────────── ──────────────
BPE 3.00 2/4

Dataset Best WordPiece Best BPE Winner
━━━━━━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━━━━━━━
CLEAR, RMSE ↓ BGE: 0.6087 DistilRoBERTa: 0.6083 BPE, effectively tied
───────────────────────── ──────────────── ─────────────────────── ───────────────────────
Mercari, RMSE ↓ BGE: 0.5747 ModernBERT: 0.5955 WordPiece
───────────────────────── ──────────────── ─────────────────────── ───────────────────────
Financial, accuracy ↑ BGE: 0.7720 DistilRoBERTa: 0.7988 BPE
───────────────────────── ──────────────── ─────────────────────── ───────────────────────
Kickstarter, accuracy ↑ BGE: 0.6488 Qwen: 0.6415 WordPiece

The stronger conclusion is model-specific:

- BGE, a WordPiece model, is the strongest overall neural embedding choice. It wins Mercari,
  Kickstarter, and COVID among embedding models.

- DistilRoBERTa is the strongest BPE candidate. It wins CLEAR and financial complaints among
  embedding models and is much cheaper than ModernBERT or Qwen.

- ModernBERT provides no clear quality advantage and is considerably slower.

- Qwen has not won any completed dataset and is dramatically slower. Missing its COVID
  result is unlikely to change the practical conclusion.

- TF-IDF is notably best on COVID, while the no-text baseline is best on financial
  complaints.
