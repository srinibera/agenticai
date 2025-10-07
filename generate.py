import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_argparser():
    p = argparse.ArgumentParser(description="Generate text from a causal LM")
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt to condition on")
    p.add_argument("--max_length", type=int, default=50)
    p.add_argument("--num_return_sequences", type=int, default=1)
    return p


def main():
    args = build_argparser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=args.max_length,
        do_sample=True,
        top_k=5,
        top_p=0.95,
        num_return_sequences=args.num_return_sequences,
    )

    for i, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        print(f"--- GENERATED {i+1} ---")
        print(text)


if __name__ == "__main__":
    main()
