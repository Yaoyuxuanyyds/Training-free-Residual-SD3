import argparse
import torch
from diffusers import StableDiffusion3Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify T5 first token semantics in SD3.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path or HF repo id for the SD3 model (expects text_encoder_3 in subfolder).",
    )
    parser.add_argument(
        "--prompt",
        default="a red apple on a wooden table",
        help="Prompt to tokenize.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length for T5 tokenizer.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for loading the pipeline (cpu/cuda).",
    )
    args = parser.parse_args()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    tokenizer = pipe.tokenizer_3

    def encode_with_flag(add_special_tokens: bool) -> torch.Tensor:
        return tokenizer(
            args.prompt,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        ).input_ids[0]

    ids_with_special = encode_with_flag(True)
    ids_no_special = encode_with_flag(False)

    first_with_special = ids_with_special[0].item()
    first_no_special = ids_no_special[0].item()

    print("=== T5 tokenizer config ===")
    print(f"name_or_path: {tokenizer.name_or_path}")
    print(f"special_tokens_map: {tokenizer.special_tokens_map}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")

    print("\n=== With add_special_tokens=True ===")
    print(f"first_id: {first_with_special}")
    print(
        f"first_token: {tokenizer.decode([first_with_special], skip_special_tokens=False)}"
    )
    print(
        f"first_is_special: {first_with_special in tokenizer.all_special_ids}"
    )

    print("\n=== With add_special_tokens=False ===")
    print(f"first_id: {first_no_special}")
    print(
        f"first_token: {tokenizer.decode([first_no_special], skip_special_tokens=False)}"
    )
    print(
        f"first_is_special: {first_no_special in tokenizer.all_special_ids}"
    )

    del pipe
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
