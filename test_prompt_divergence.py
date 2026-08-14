import sentence_transformers as st
import torch

model_name = "intfloat/e5-base-v2"
model = st.SentenceTransformer(model_name, device="cuda")
model.eval()

text = "What is the capital of France?"

# CPU encode() path — applies default prompt automatically
with torch.inference_mode():
    emb_encode = model.encode(
        [text],
        convert_to_tensor=True,
        device="cuda",
        show_progress_bar=False,
    )

# Direct tokenize + module iteration — no prompt
tokenized = model.tokenizer(
    [text],
    padding=True,
    truncation=True,
    max_length=model.max_seq_length,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    features = dict(tokenized)
    for module in model:
        features = module(features)
    emb_direct = features["sentence_embedding"]

print(f"default_prompt_name: {model.default_prompt_name}")
print(f"prompts: {model.prompts}")
print(f"allclose: {torch.allclose(emb_encode, emb_direct, atol=1e-5)}")
print(f"max diff: {(emb_encode - emb_direct).abs().max().item()}")
