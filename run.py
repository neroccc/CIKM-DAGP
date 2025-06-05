# run_training.py

import sys
import trainer  # Assuming your main training script is named trainer.py
import argparse

def run():
    sys.argv = [
        "trainer.py",
        "--output_dir", "./checkpoints/deepseekcoder",
        "--model_name_or_path", "deepseek-ai/deepseek-coder-1.3b-instruct",
        "--train_data_file", "leetcode-train-clean.jsonl",
        "--eval_data_file", "leetcode-train-clean.jsonl",
        "--do_train",
        "--max_length", "512",
        "--train_batch_size", "2",
        "--eval_batch_size", "2",
        "--learning_rate", "3e-5",
        "--num_train_epochs", "3"
    ]

    trainer.main()

if __name__ == "__main__":
    run()
