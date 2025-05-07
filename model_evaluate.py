import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import json
import itertools

# for detailed ROUGE
from rouge_score import rouge_scorer
from statistics import mean

# for sacreBLEU and BERTScore
import evaluate

# ─── (Re-using existing retrieval/encoding logic) ─────────────────────────
from train import QuestionUnderstanding, TableEncoder, TableRetriever

class FeTaQAValDataset(Dataset):
    def __init__(self, path, tokenizer, max_in=512):
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.tr = TableRetriever()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        ctx = self.tr.get_context(ex['question'], ex['table_array'])
        table_text = '\n'.join([' | '.join(map(str, r)) for r in ctx])
        inp = f"question: {ex['question']} table: {table_text}"
        X = self.tokenizer(
            inp,
            max_length=self.max_in,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return X['input_ids'].squeeze(), X['attention_mask'].squeeze(), ex['answer']

# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(preds, refs):
    # Detailed ROUGE via rouge_score
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=True
    )
    r1, r2, rL, rLsum = [], [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1 .append(scores['rouge1'].fmeasure)
        r2 .append(scores['rouge2'].fmeasure)
        rL .append(scores['rougeL'].fmeasure)
        rLsum.append(scores['rougeLsum'].fmeasure)

    # sacreBLEU (gives overall score + n‑gram precisions)
    bleu = evaluate.load('sacrebleu')
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bleu_score      = bleu_res['score']
    bleu_precisions = bleu_res['precisions']

    # BERTScore
    bert = evaluate.load('bertscore')
    bert_res = bert.compute(predictions=preds, references=refs, lang='en')
    bert_p = mean(bert_res['precision'])
    bert_r = mean(bert_res['recall'])
    bert_f = mean(bert_res['f1'])

    return {
        'rouge1'            : mean(r1),
        'rouge2'            : mean(r2),
        'rougeL'            : mean(rL),
        'rougeLsum'         : mean(rLsum),
        'bleu'              : bleu_score,
        'bleu_precisions'   : bleu_precisions,
        'bertscore_precision': bert_p,
        'bertscore_recall'   : bert_r,
        'bertscore_f1'       : bert_f
    }

# ─── Evaluation ───────────────────────────────────────────────────────────────
def run_evaluation(model_dir, val_path, max_examples=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model     = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

    ds = FeTaQAValDataset(val_path, tokenizer)
    dl = DataLoader(ds, batch_size=1)

    preds, refs = [], []
    for inp_ids, attn_mask, ref in tqdm(
        itertools.islice(dl, max_examples) if max_examples else dl,
        desc="Evaluating"
    ):
        out = model.generate(
            input_ids=inp_ids.to(device),
            attention_mask=attn_mask.to(device),
            max_length=128,
            num_beams=4
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        preds.append(pred)

        ref_str = ref[0] if isinstance(ref, (tuple, list)) else ref
        refs.append(ref_str.strip())

    scores = compute_metrics(preds, refs)

    
    print("\n=== Evaluation Results ===\n")

    print("ROUGE Scores:")
    print(f"rouge1:    {scores['rouge1']:.4f}")
    print(f"rouge2:    {scores['rouge2']:.4f}")
    print(f"rougeL:    {scores['rougeL']:.4f}")
    print(f"rougeLsum: {scores['rougeLsum']:.4f}\n")

    print("BLEU Score:")
    print(f"sacreBLEU: {scores['bleu']:.4f}")
    print("Precisions:")
    for i, p in enumerate(scores['bleu_precisions'], start=1):
        print(f"Precision {i}: {p:.4f}")
    print()

    print("BERTScore:")
    print(f"precision: {scores['bertscore_precision']:.4f}")
    print(f"recall:    {scores['bertscore_recall']:.4f}")
    print(f"f1:        {scores['bertscore_f1']:.4f}")

if __name__ == "__main__":
    run_evaluation(
        model_dir='models/t5_fetaqa',
        val_path='fetaQA-v1_test.jsonl',
        max_examples=None      
    )
