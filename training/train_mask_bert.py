#!/usr/bin/env python3
"""
Fine-tune BERTurk-Legal to label anonymisation masks in Danıştay decisions.

Input: the dataset built by build_mask_dataset.py (train/validation/test
.jsonl). Each row has a text where every mask is the token [SANSUR], and one
label per [SANSUR] in order. The model only ever sees `text`; `labels` are
used for the loss, at the [SANSUR] positions only (all other tokens -100).

Model: KocLab-Bilkent/BERTurk-Legal (uncased, 128k vocab) with a token
classification head. [SANSUR] is added as a special token, its embedding
initialised from [MASK]. Text is lowercased the Turkish way (I->ı, İ->i)
before tokenizing, since the model is uncased and the hub repo ships no
tokenizer config.

Usage:
  python train_mask_bert.py --data-dir ../data/datasets/danistay_mask_v1 --out runs/v1
  # quick smoke test on CPU:
  python train_mask_bert.py --data-dir ... --out runs/smoke --limit 64 --max-steps 5

Writes to --out: the best checkpoint (model + tokenizer), training_info.json,
test_report.json (accuracy, macro F1, per-label scores, confusion matrix) and
test_predictions.jsonl (every test chunk with gold and predicted labels).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import (AutoModelForTokenClassification, BertTokenizerFast,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)

MASK_TOKEN = "[SANSUR]"
LABELS = ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]


def tr_lower(text: str) -> str:
    """Turkish lowercasing that leaves [SANSUR] intact."""
    parts = text.split(MASK_TOKEN)
    return MASK_TOKEN.join(p.replace("I", "ı").replace("İ", "i").lower() for p in parts)


def load_tokenizer(model_name: str, max_length: int) -> BertTokenizerFast:
    vocab = hf_hub_download(model_name, "vocab.txt") if not Path(model_name).is_dir() \
        else str(Path(model_name) / "vocab.txt")
    tok = BertTokenizerFast(vocab_file=vocab, do_lower_case=False, strip_accents=False,
                            tokenize_chinese_chars=False, model_max_length=max_length)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    return tok


def make_preprocess(tok, label2id, max_length, ignore_mask_label):
    sansur_id = tok.convert_tokens_to_ids(MASK_TOKEN)

    def preprocess(batch):
        enc = tok([tr_lower(t) for t in batch["text"]], truncation=True, max_length=max_length)
        all_labels, lost = [], []
        for ids, labels in zip(enc["input_ids"], batch["labels"]):
            row = [-100] * len(ids)
            positions = [i for i, t in enumerate(ids) if t == sansur_id]
            lost.append(len(labels) - len(positions))   # masks cut off by truncation
            for pos, lab in zip(positions, labels):
                if not (ignore_mask_label and lab == "MASK"):
                    row[pos] = label2id[lab]
            all_labels.append(row)
        enc["labels"] = all_labels
        enc["lost_masks"] = lost
        return enc
    return preprocess


def make_metrics(id2label, ignore_mask_label):
    """Accuracy and macro F1 over the scored labels; the confusion matrix of
    the last evaluation is kept on the returned function (.confusion), since
    Trainer metrics must be numbers."""
    names = [id2label[i] for i in range(len(id2label))]
    scored = [i for i, n in enumerate(names) if not (ignore_mask_label and n == "MASK")]

    def compute(eval_pred):
        preds, labels = eval_pred
        keep = labels != -100
        p, y = preds[keep], labels[keep]
        cm = np.zeros((len(names), len(names)), dtype=int)
        np.add.at(cm, (y, p), 1)
        out = {"accuracy": float((p == y).mean()) if len(y) else 0.0, "n": int(len(y))}
        f1s = []
        for i in scored:
            tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            out[f"f1_{names[i]}"] = float(f1)
            if cm[i, :].sum():
                f1s.append(f1)
        out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
        compute.confusion = cm.tolist()
        return out
    compute.confusion = None
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="KocLab-Bilkent/BERTurk-Legal")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16, help="per device")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-mask-label", action="store_true",
                    help="also train/score the MASK (unknown) label; ignored by default")
    ap.add_argument("--limit", type=int, default=0, help="use only N rows per split (testing)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (testing)")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers")
    ap.add_argument("--tiny-random-model", action="store_true",
                    help="testing only: a 2-layer randomly initialised BERT with the same "
                         "vocab instead of the pretrained weights (checks the code, not quality)")
    args = ap.parse_args()
    set_seed(args.seed)
    ignore_mask = not args.keep_mask_label

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    tok = load_tokenizer(args.model, args.max_length)

    files = {s: str(args.data_dir / f"{s}.jsonl") for s in ("train", "validation", "test")}
    ds = load_dataset("json", data_files=files)
    if args.limit:
        ds = {s: d.select(range(min(args.limit, len(d)))) for s, d in ds.items()}
    keep_cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    proc = {}
    for s, d in ds.items():
        p = d.map(make_preprocess(tok, label2id, args.max_length, ignore_mask),
                  batched=True, remove_columns=d.column_names, desc=f"tokenize {s}")
        lost = sum(p["lost_masks"])
        print(f"{s}: {len(p)} chunks; masks lost to truncation: {lost}")
        proc[s] = p.remove_columns([c for c in p.column_names if c not in keep_cols])

    if args.tiny_random_model:
        from transformers import BertConfig
        cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=128,
                         num_labels=len(LABELS), id2label=id2label, label2id=label2id)
        model = AutoModelForTokenClassification.from_config(cfg)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(LABELS), id2label=id2label, label2id=label2id)
    old = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    if len(tok) > old:   # start [SANSUR] from [MASK] instead of noise
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[tok.convert_tokens_to_ids(MASK_TOKEN)] = emb[tok.mask_token_id]

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio, weight_decay=0.01,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        bf16=bf16, fp16=cuda and not bf16,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.max_steps if args.max_steps > 0 else None,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=50, report_to="none", seed=args.seed,
        dataloader_num_workers=args.workers if cuda else 0,
    )
    metrics = make_metrics(id2label, ignore_mask)
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=proc["train"], eval_dataset=proc["validation"],
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metrics,
        # keep only argmax ids, not full logits, to save memory during eval
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1),
    )
    trainer.train()

    pred = trainer.predict(proc["test"], metric_key_prefix="test")
    test = pred.metrics
    args.out.mkdir(parents=True, exist_ok=True)

    # per-chunk predictions on the test set, for error analysis: read the
    # prediction at every [SANSUR] (also the ones whose gold label is ignored)
    sansur_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    with open(args.out / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for row, ids, preds in zip(ds["test"], proc["test"]["input_ids"], pred.predictions):
            positions = [i for i, t in enumerate(ids) if t == sansur_id]
            predicted = [id2label[int(preds[p])] for p in positions]
            text = row["text"]
            for lab in predicted:        # [SANSUR] -> predicted label, in order
                text = text.replace(MASK_TOKEN, f"<{lab}>", 1)
            f.write(json.dumps({
                "id": row["id"], "text": row["text"], "gold": row["labels"],
                "predicted": predicted, "text_predicted": text,
                "errors": sum(g != p for g, p in zip(row["labels"], predicted)
                              if not (ignore_mask and g == "MASK")),
            }, ensure_ascii=False) + "\n")
    best = args.out / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    report = {"labels": LABELS, "ignored_labels": ["MASK"] if ignore_mask else [],
              **{k.replace("test_", ""): v for k, v in test.items()},
              "confusion (rows=true, cols=predicted)": metrics.confusion}
    (args.out / "test_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (args.out / "training_info.json").write_text(json.dumps({
        "base_model": args.model, "data_dir": str(args.data_dir), "mask_token": MASK_TOKEN,
        "lowercase": "Turkish: I->ı, İ->i, then .lower(); [SANSUR] kept as is",
        "args": {k: str(v) for k, v in vars(args).items()},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
    }, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("confusion")}, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
