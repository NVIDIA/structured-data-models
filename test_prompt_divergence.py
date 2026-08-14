import sentence_transformers as st
import torch

from sdm import StringTensor, TableTensor
from sdm.processing import SentenceTransformer

model_name = "intfloat/e5-base-v2"
device = torch.device("cuda:0")

model = st.SentenceTransformer(model_name)
processor = SentenceTransformer(model_name).to(device)

long_text = "word " * model.max_seq_length

texts = [
    ["hello world", "short"],
    ["", "a"],
    [None, "café naïve üñîçødé"],
    [long_text, "x y z"],
]

table = TableTensor(
    columns={"text": ("title", "body")},
    text=StringTensor.from_list(texts, device=device),
)
gpu_output = processor(table)
gpu_emb = gpu_output.numerical

flat_texts = [
    t if t is not None else ""
    for col_idx in range(2)
    for row in texts
    for t in [row[col_idx]]
]
model.eval()
with torch.inference_mode():
    ref_emb = model.encode(
        flat_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=str(device),
    )
ref_emb = (
    ref_emb.reshape(2, len(texts), -1).movedim(0, -2).reshape(len(texts), -1)
)

print(f"Model: {model_name}")
print(f"Default prompt: {model.default_prompt_name!r}")
print()

for i, row in enumerate(texts):
    print(f"Row {i}: {row}")
    print(f"  GPU: {gpu_emb[i, :5].tolist()}")
    print(f"  CPU: {ref_emb[i, :5].tolist()}")
    print(f"  Max diff: {(gpu_emb[i] - ref_emb[i]).abs().max().item():.6f}")
    print()
