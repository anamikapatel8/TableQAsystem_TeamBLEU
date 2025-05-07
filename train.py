# train.py

import json
import pandas as pd
import torch
import faiss
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    
)
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType


# ─── Data Loader 
# ───────────────────────────────────────────────────────────────
class FeTaQADataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        examples = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line.strip())
                examples.append({
                    'question': ex['question'],
                    'answer': ex['answer'],
                    'table': ex['table_array']
                })
        return examples

# ─── Question Understanding 
# ────────────────────────────────────────────────────
class QuestionUnderstanding:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.types = ['lookup','aggregation','comparison','reasoning']
        self.encoder = LabelEncoder().fit(self.types)

    def classify(self, q: str):
        ql = q.lower()
        if any(w in ql for w in ['how many','sum','average','total']):
            return 'aggregation'
        if any(w in ql for w in ['more than','less than','most','least']):
            return 'comparison'
        if any(w in ql for w in ['why','how','explain']):
            return 'reasoning'
        return 'lookup'

    def extract_keywords(self, q: str):
        stop = {'what','is','are','the','in','on','at','which','who','where','when','how','why'}
        return [w for w in q.lower().split() if w not in stop]

    def embed(self, q: str):
        return self.model.encode(q, convert_to_tensor=True)

    def process(self, q: str):
        return {
            'type': self.classify(q),
            'keywords': self.extract_keywords(q),
            'emb': self.embed(q)
        }

# ─── Table Encoder 
# ─────────────────────────────────────────────────────────────
class TableEncoder:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None

    def encode_rows(self, df: pd.DataFrame):
        texts = [' | '.join(map(str,row)) for _,row in df.iterrows()]
        embs = self.model.encode(texts, convert_to_tensor=True)
        arr = embs.cpu().numpy()
        if self.index is None:
            self.index = faiss.IndexFlatL2(arr.shape[1])
        self.index.reset()
        self.index.add(arr)
        return texts, embs

    def query_topk(self, q_emb, k=3):
        qn = q_emb.cpu().numpy().reshape(1, -1)
        _, idxs = self.index.search(qn, k)
        return idxs[0]

# ─── Retriever ────────────────────────────────────────────────────────────────
class TableRetriever:
    def __init__(self):
        self.qu = QuestionUnderstanding()
        self.te = TableEncoder()

    def get_context(self, question, table_array, k=10):
        df = pd.DataFrame(table_array[1:], columns=table_array[0])
        qinfo = self.qu.process(question)
        _, embs = self.te.encode_rows(df)
        topk = self.te.query_topk(qinfo['emb'], k=k)
        selected = [table_array[0]] + [table_array[i+1] for i in topk]
        return selected

# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class FeTaQADataset(Dataset):
    def __init__(self, examples, tokenizer, max_in=256, max_out=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_in, self.max_out = max_in, max_out
        self.retriever = TableRetriever()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        context = self.retriever.get_context(ex['question'], ex['table'])
        table_text = '\n'.join([' | '.join(map(str,row)) for row in context])
        inp = f"question: {ex['question']} table: {table_text}"
        tgt = ex['answer']
        model_inputs = self.tokenizer(inp, max_length=self.max_in, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt, max_length=self.max_out, truncation=True)
        # Replacing pad_token_id with -100 for loss masking
        labels = labels["input_ids"]
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        model_inputs["labels"] = labels
        
        return model_inputs

# ─── Training Loop 
# ─────────────────────────────────────────────────────────────
def train_model(
    train_path,
    out_dir='models/t5_fetaqa',
    epochs=7,
    bs=4,
    lr=3e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained('t5-large').to(device)

    # # --- LoRA adapter ---
    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q","v"],
        lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, peft_config)

    # Data & Dataloader with dynamic padding
    examples = FeTaQADataLoader(train_path).load()
    ds = FeTaQADataset(examples, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', label_pad_token_id=-100)

    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collator)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=lr, eps=1e-6)
    total_steps = epochs * len(dl)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()

    # Training
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        prog = tqdm(dl, desc=f"Epoch {ep}/{epochs}", dynamic_ncols=True, mininterval=2)
        for batch in prog:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with mixed precision
            if device.type == 'cuda':
                with autocast(dtype=autocast_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Skipping batch if loss is NaN or infinite
            if not torch.isfinite(loss):
                prog.write(f"NaN or infinite loss encountered at epoch {ep}, skipping batch.")
                continue

            # Backward and optimization step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1
            prog.set_postfix(loss=running_loss / step_count)

        if step_count > 0:
            avg_loss = running_loss / step_count
        else:
            avg_loss = float('nan')
        prog.write(f"Epoch {ep} avg loss: {avg_loss:.4f}")

    # Save
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nModel & tokenizer saved to {out_dir}")

if __name__ == "__main__":
    train_model('fetaQA-v1_train.jsonl')
