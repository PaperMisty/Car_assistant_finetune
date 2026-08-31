import argparse
import json
import os
from typing import List


def parse_target_modules(value: str):
    if value == "all-linear":
        return value
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise argparse.ArgumentTypeError("target_modules 不能为空")
    return modules


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 微调单次实验")
    parser.add_argument("--experiment-name", default="lora_baseline")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=parse_target_modules, default="all-linear")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-dir", default="./logs/07_lora_demo")
    parser.add_argument("--output-dir", default="./finetuned/07_lora_demo")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def printable_config(args):
    return {
        "experiment_name": args.experiment_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "logging_dir": args.logging_dir,
        "output_dir": args.output_dir,
    }


def train(args):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    model = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-0.6B")

    data = load_dataset(
        "json",
        data_files={
            "train": "./data/keywords_data_train.jsonl",
            "test": "./data/keywords_data_test.jsonl",
        },
    )

    def map_function(samples: dict[str, List]):
        conversation_lists = samples["conversation"]
        message_lists = []
        for sample in conversation_lists:
            user_message = sample[0]["human"]
            assistant_message = sample[0]["assistant"]
            message_lists.append(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            )
        return {"messages": message_lists}

    mapped_data = data.map(
        map_function,
        batched=True,
        remove_columns=data.column_names["train"],
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir
    config = SFTConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        output_dir=args.output_dir,
        bf16=True,
        gradient_checkpointing=True,
        activation_offloading=False,
        max_length=700,
        assistant_only_loss=True,
        chat_template_path="./new_chat_template.jinja",
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=config,
        train_dataset=mapped_data["train"],
        eval_dataset=mapped_data["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.dry_run:
        print(json.dumps(printable_config(cli_args), ensure_ascii=False))
    else:
        train(cli_args)
