import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_single_token_id(model: HookedTransformer, text: str) -> int:
    tokens = model.to_tokens(text, prepend_bos=False)
    if tokens.shape[-1] != 1:
        raise ValueError(f"{text!r} is not a single token for this tokenizer.")
    return int(tokens[0, 0].item())


def main() -> None:
    device = get_device()
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    clean_prompt = "The capital of France is"
    corrupted_prompt = "The capital of Italy is"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError("Clean and corrupted prompts must have the same token shape.")

    seq_len = clean_tokens.shape[1]

    target_text = " Paris"
    target_token_id = get_single_token_id(model, target_text)

    print("=" * 90)
    print("Prompts")
    print("=" * 90)
    print(f"Clean prompt:     {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Sequence length:  {seq_len}")
    print(f"Target token:     {target_text} (id={target_token_id})")

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

    clean_target_logit = clean_logits[0, -1, target_token_id].item()
    corrupted_target_logit = corrupted_logits[0, -1, target_token_id].item()

    print("\n" + "=" * 90)
    print("Baseline logits")
    print("=" * 90)
    print(f"Clean target logit:     {clean_target_logit:.4f}")
    print(f"Corrupted target logit: {corrupted_target_logit:.4f}")

    layer = 0

    print("\n" + "=" * 90)
    print(f"Position-wise patching on blocks.{layer}.hook_resid_pre")
    print("=" * 90)

    results = []

    for pos in range(seq_len):

        def patch_one_position(resid_pre: torch.Tensor, hook, pos=pos) -> torch.Tensor:
            patched = resid_pre.clone()
            patched[:, pos, :] = clean_cache["resid_pre", layer][:, pos, :]
            return patched

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_one_position)],
        )

        patched_target_logit = patched_logits[0, -1, target_token_id].item()
        results.append((pos, patched_target_logit))

        print(
            f"Position {pos:>2}: patched target logit = {patched_target_logit:.4f}"
        )

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    best_pos, best_logit = max(results, key=lambda x: x[1])
    print(f"Most influential patched position at layer {layer}: {best_pos}")
    print(f"Best patched target logit: {best_logit:.4f}")

    print("\nInterpretation:")
    print("A position is more influential if replacing only that position's residual")
    print("state moves the corrupted target logit more strongly toward the clean value.")

    print("\nDone.")


if __name__ == "__main__":
    main()