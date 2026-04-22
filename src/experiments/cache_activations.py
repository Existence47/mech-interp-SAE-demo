import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = get_device()

    print("=" * 80)
    print("Step 1: Load model")
    print("=" * 80)
    print(f"Device: {device}")

    # gpt2-small is a standard small GPT-style model supported by TransformerLens.
    # It is not tiny, but it is stable and widely used for mechanistic interpretability demos.
    model_name = "gpt2-small"
    model = HookedTransformer.from_pretrained(model_name, device=device)

    print(f"Loaded model: {model_name}")
    print(f"Number of layers: {model.cfg.n_layers}")
    print(f"Model dimension d_model: {model.cfg.d_model}")
    print(f"Vocabulary size d_vocab: {model.cfg.d_vocab}")

    print("\n" + "=" * 80)
    print("Step 2: Prepare prompt")
    print("=" * 80)

    prompt = "The capital of France is"
    tokens = model.to_tokens(prompt)

    print(f"Prompt: {prompt}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens: {tokens}")

    print("\n" + "=" * 80)
    print("Step 3: Run model with activation cache")
    print("=" * 80)

    logits, cache = model.run_with_cache(tokens)

    print(f"Logits shape: {logits.shape}")
    print(f"Number of cached activation entries: {len(cache)}")

    print("\nFirst 20 cache keys:")
    for i, key in enumerate(cache.keys()):
        if i >= 20:
            break
        print(f"{i:02d}: {key}")

    print("\n" + "=" * 80)
    print("Step 4: Inspect residual stream")
    print("=" * 80)

    # resid_pre is the residual stream at the beginning of a transformer block.
    # Layer 0 means the first transformer block.
    resid_pre_0 = cache["resid_pre", 0]

    print("cache['resid_pre', 0]")
    print(f"Shape: {resid_pre_0.shape}")
    print("Expected meaning: [batch, position, d_model]")

    batch_size, seq_len, d_model = resid_pre_0.shape
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Residual stream dimension: {d_model}")

    print("\n" + "=" * 80)
    print("Step 5: Inspect final prediction")
    print("=" * 80)

    # logits[:, -1, :] means:
    # - all batch items
    # - the final token position
    # - all vocabulary logits
    final_logits = logits[:, -1, :]
    predicted_token_id = final_logits.argmax(dim=-1)

    predicted_text = model.to_string(predicted_token_id)

    print(f"Predicted token id: {predicted_token_id}")
    print(f"Predicted next token: {predicted_text}")

    print("\nDone.")


if __name__ == "__main__":
    main()