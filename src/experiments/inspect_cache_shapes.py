import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"{name:<35} {tuple(tensor.shape)}")


def main() -> None:
    device = get_device()
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    prompt = "The capital of France is"
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)

    print("=" * 90)
    print("Important cache entries and their shapes")
    print("=" * 90)

    important_keys = [
        "hook_embed",
        "hook_pos_embed",
        "blocks.0.hook_resid_pre",
        "blocks.0.attn.hook_q",
        "blocks.0.attn.hook_k",
        "blocks.0.attn.hook_v",
        "blocks.0.attn.hook_attn_scores",
        "blocks.0.attn.hook_pattern",
        "blocks.0.attn.hook_z",
        "blocks.0.hook_attn_out",
        "blocks.0.hook_resid_mid",
        "blocks.0.mlp.hook_pre",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_mlp_out",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_pre",
    ]

    for key in important_keys:
        print_shape(key, cache[key])

    print("\n" + "=" * 90)
    print("Logits shape")
    print("=" * 90)
    print_shape("logits", logits)

    print("\nDone.")


if __name__ == "__main__":
    main()
