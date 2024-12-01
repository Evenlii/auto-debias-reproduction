import numpy as np
from logging import Logger
from argparse import Namespace, ArgumentParser
from auto_config import get_aa_args
from auto_utils import (
    clear_console,
    get_aa_logger,
    prepare_model_and_tokenizer,
    JSDivergence,
    load_words,
    clear_words,
    get_prompt_js_div,
    MALE_WORDS,
    FEMALE_WORDS,
    AFA,
    EUA,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser()

    # Add all required arguments
    parser.add_argument("--model_version", type=str, required=True, help="Model version")
    parser.add_argument(
        "--model_name", type=str, choices=["bert", "roberta", "gpt2"], required=True, help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument(
        "--debias_type", type=str, choices=["gender", "race"], required=True, help="Debiasing type"
    )
    parser.add_argument("--run_no", type=str, default="run00", help="Run identifier")  # Add this
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--save_dir", type=str, default="./prompts/", help="Directory to save prompts")  # Add this
    parser.add_argument("--ckpt_path", type=str, default="./ckpts/", help="Checkpoint path")  # Add this
    parser.add_argument("--max_prompt_len", type=int, default=5, help="Max length of prompts")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K prompts to select")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument("--max_epochs", type=int, default=100, help="Max number of epochs")
    parser.add_argument("--output_dir", type=str, default="./out/", help="Output directory")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--precision", type=int, default=16, help="Precision (e.g., 16 for fp16)")

    return parser.parse_args()


def generate_prompts(args: Namespace, logger: Logger):
    """Generate prompts and save them using JS-Divergence."""
    logger.info(f"Prepare pre-trained model and tokenizer: {args.model_version}")
    model, tokenizer = prepare_model_and_tokenizer(
        model_name=args.model_name, model_version=args.model_version
    )

    logger.info(f"Load and tokenize stereotype words from {args.data_dir + 'stereotype_words.txt'}")
    stereotype_words = clear_words(
        _words1=load_words(path=args.data_dir + "stereotype_words.txt", mode="stereotype"),
        _words2=None,
        tokenizer=tokenizer,
        mode="stereotype",
    )
    stereotype_ids = tokenizer.convert_tokens_to_ids(stereotype_words)

    logger.info(f"Load and tokenize prompt words from {args.data_dir + 'wiki_words_5000.txt'}")
    prompt_words = clear_words(
        _words1=load_words(path=args.data_dir + "wiki_words_5000.txt", mode="mode"),
        _words2=None,
        tokenizer=tokenizer,
        mode="mode",
    )

    current_prompts = prompt_words
    prompt_file = open(f"{args.save_dir}prompts_{args.model_version}_{args.debias_type}", "w")
    js_div_module = JSDivergence(reduction="none")

    for i in range(args.max_prompt_len):
        logger.info(f"Generating prompts for length {i+1}")
        if args.debias_type == "gender":
            current_prompts_js_div_values = get_prompt_js_div(
                prompts=current_prompts,
                targ1_words=MALE_WORDS,
                targ2_words=FEMALE_WORDS,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=stereotype_ids,
                js_div_module=js_div_module,
                args=args,
            )
        elif args.debias_type == "race":
            current_prompts_js_div_values = get_prompt_js_div(
                prompts=current_prompts,
                targ1_words=AFA,
                targ2_words=EUA,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=stereotype_ids,
                js_div_module=js_div_module,
                args=args,
            )

        selected_prompts = np.array(current_prompts)[
            np.argsort(current_prompts_js_div_values)[::-1][: args.top_k]
        ]

        for selected_prompt in selected_prompts:
            prompt_file.write(selected_prompt + "\n")

        temp_prompts = []
        for selected_prompt in selected_prompts:
            for prompt_word in prompt_words:
                temp_prompts.append(selected_prompt + " " + prompt_word)

        current_prompts = temp_prompts

    logger.info("Prompts generation completed.")
    prompt_file.close()


if __name__ == "__main__":
    clear_console()
    args = parse_arguments()
    logger = get_aa_logger(args)
    generate_prompts(args=args, logger=logger)
