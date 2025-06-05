from __future__ import absolute_import, division, print_function

import os
import json
import random
import torch
import logging
import argparse
import numpy as np
import sys
sys.path.append("..")
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification)
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
REVERSE_DIFFICULTY_MAP = {v: k for k, v in DIFFICULTY_MAP.items()}

no_deprecation_warning = True
logger = logging.getLogger(__name__)
early_stopping = EarlyStopping()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, label, index):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.index = index


def convert_examples_to_features(js, tokenizer, args):
    source_tokens = tokenizer.encode_plus(
        js['content'],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    input_ids = source_tokens["input_ids"]
    attention_mask = source_tokens["attention_mask"]
    label = DIFFICULTY_MAP[js['difficulty']]
    return InputFeatures(input_ids, attention_mask, label, js['id'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                #logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        return (example.input_ids, example.attention_mask, torch.tensor(example.label), example.index)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataloader, eval_dataloader, model, tokenizer):
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    model.zero_grad()
    best_f1 = 0
    losses = []
    for idx in range(args.num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch[0].to(args.device).squeeze(1)
            attention_mask = batch[1].to(args.device).squeeze(1)
            labels = batch[2].to(args.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = F.cross_entropy(logits, labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(np.mean(losses[-100:]), 4)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results, eval_loss = evaluate(args, eval_dataloader, model)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            logger.info("  " + "*" * 20)
            logger.info("  Best f1:%s", round(best_f1, 4))
            logger.info("  " + "*" * 20)
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saved full model and tokenizer to %s", output_dir)
        early_stopping(eval_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def evaluate(args, eval_dataloader, model):
    logger.info("***** Running evaluation *****")
    eval_loss = 0.0
    model.eval()
    probs, labels = [], []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        input_ids = batch[0].to(args.device).squeeze(1)
        attention_mask = batch[1].to(args.device).squeeze(1)
        label = batch[2].to(args.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            prob = torch.nn.functional.softmax(logits, dim=-1)
            lm_loss = F.cross_entropy(logits, label)
            eval_loss += lm_loss.item()
            probs.append(prob.detach().to(torch.float32).cpu().numpy())
            labels.append(label.detach().to(torch.float32).cpu().numpy())

    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels, 0).astype(int)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    conf_matrix = confusion_matrix(labels, preds)

    df = pd.DataFrame({
        'pred': [REVERSE_DIFFICULTY_MAP[int(p)] for p in preds],
        'true': [REVERSE_DIFFICULTY_MAP[int(l)] for l in labels]
    })
    df.to_csv(f'../{args.output_dir.split("/")[-1]}.csv', index=False)

    results = {
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": conf_matrix[0][1] / (conf_matrix[0][0] + conf_matrix[0][1]) if (conf_matrix[0][0] + conf_matrix[0][1]) > 0 else 0
    }
    return results, eval_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-coder-1.3b", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3,
        trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = model.config.eos_token_id

    args.n_gpu = 1
    args.device = torch.device("cuda:3")
    model.to(args.device)
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info("Training/evaluation parameters %s", args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(4)))
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=args.train_batch_size, num_workers=4)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                     batch_size=args.eval_batch_size, num_workers=4)
        train(args, train_dataloader, eval_dataloader, model, tokenizer)

    if args.do_test:
        ckpt = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        model.load_state_dict(torch.load(ckpt, map_location='cuda:3'))
        test_dataset = TextDataset(tokenizer, args, args.test_data_file)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                     batch_size=args.eval_batch_size, num_workers=6)
        result, _ = evaluate(args, test_dataloader, model)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == "__main__":
    main()
