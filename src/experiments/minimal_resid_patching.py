import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_token_id(model: HookedTransformer, text: str) -> int:
    """
    Convert a text fragment into a single token id.

    We expect the input text to correspond to exactly one token.
    """
    tokens = model.to_tokens(text, prepend_bos=False)
    if tokens.shape[-1] != 1:
        raise ValueError(f"Text {text!r} is not a single token for this tokenizer.")
    return int(tokens[0, 0].item())


def main() -> None:
    device = get_device()
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    clean_prompt = "The capital of France is"
    corrupted_prompt = "The capital of Italy is"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    print("=" * 90)
    print("Prompts")
    print("=" * 90)
    print(f"Clean prompt:     {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Clean tokens shape: {tuple(clean_tokens.shape)}")
    print(f"Corrupted tokens shape: {tuple(corrupted_tokens.shape)}")

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError("Clean and corrupted prompts must have the same token shape for this demo.")

    target_text = " Paris"
    target_token_id = get_token_id(model, target_text)

    print("\n" + "=" * 90)
    print("Target token")
    print("=" * 90)
    print(f"Target text: {target_text}")
    print(f"Target token id: {target_token_id}")

    # Run clean and corrupted forward passes with cache
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

    # We care about the final position prediction
    clean_final_logits = clean_logits[:, -1, :]
    corrupted_final_logits = corrupted_logits[:, -1, :]

    clean_target_logit = clean_final_logits[0, target_token_id].item()
    corrupted_target_logit = corrupted_final_logits[0, target_token_id].item()

    print("\n" + "=" * 90)
    print("Before patching")
    print("=" * 90)
    print(f"Clean logit for {target_text!r}: {clean_target_logit:.4f}")
    print(f"Corrupted logit for {target_text!r}: {corrupted_target_logit:.4f}")

    # We will patch resid_pre at layer 0, for all positions
    def patch_resid_pre(resid_pre: torch.Tensor, hook) -> torch.Tensor:
        """
        Replace corrupted resid_pre with clean resid_pre at layer 0.
        """
        return clean_cache["resid_pre", 0]

    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[("blocks.0.hook_resid_pre", patch_resid_pre)],
    )

    patched_final_logits = patched_logits[:, -1, :]
    patched_target_logit = patched_final_logits[0, target_token_id].item()

    print("\n" + "=" * 90)
    print("After patching")
    print("=" * 90)
    print(f"Patched logit for {target_text!r}: {patched_target_logit:.4f}")

    print("\n" + "=" * 90)
    print("Interpretation")
    print("=" * 90)
    print("If the patched logit moves closer to the clean logit,")
    print("then the patched activation likely carries information relevant")
    print("to predicting the target token.")

    print("\nDone.")


if __name__ == "__main__":
    main()