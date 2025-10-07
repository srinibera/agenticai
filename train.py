import argparse
from pathlib import Path

def build_argparser():
    p = argparse.ArgumentParser(description="Simple fine-tune script for causal LM using Hugging Face Transformers")
    p.add_argument("--train_file", type=str, default="data/train.txt", help="Path to a plain-text training file (one example per line)")
    p.add_argument("--model_name_or_path", type=str, default="gpt2", help="Base model name or path")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to save the fine-tuned model")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    return p


def main():
    args = build_argparser().parse_args()

    # Lazy import heavy libraries so the script can be syntax-checked without installing deps
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path.resolve()}")

    print("Loading tokenizer and model (this may download weights)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Ensure a pad token exists (GPT2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    print("Loading dataset from text file...")
    ds = load_dataset("text", data_files={"train": args.train_file})

    def tokenize_batch(examples):
        # join lines to single string per example if dataset returns lines
        texts = examples["text"]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"]) 
    tokenized.set_format(type="torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
