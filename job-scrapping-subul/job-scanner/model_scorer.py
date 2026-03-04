# model_scorer.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MAX_LEN    = 256
MODEL_PATH = "jobscan_model/finetuned_model.pt"
TOK_PATH   = "jobscan_model/tokenizer"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FineTunedMiniLM(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_name)
        # Architecture exacte depuis le checkpoint
        self.regressor = nn.Sequential(
            nn.Linear(1536, 768),      # 0 — 1536 = 768*2
            nn.BatchNorm1d(768),       # 1
            nn.GELU(),                 # 2
            nn.Dropout(dropout),       # 3
            nn.Linear(768, 256),       # 4
            nn.BatchNorm1d(256),       # 5
            nn.GELU(),                 # 6
            nn.Dropout(dropout),       # 7
            nn.Linear(256, 64),        # 8
            nn.GELU(),                 # 9
            nn.Linear(64, 1),          # 10
            nn.Sigmoid()               # 11
        )

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, r_ids, r_mask, j_ids, j_mask):
        r        = self.encode(r_ids, r_mask)
        j        = self.encode(j_ids, j_mask)
        combined = torch.cat([r, j], dim=1)   # 768+768 = 1536
        return self.regressor(combined).squeeze(-1)


_model     = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print("⏳ Chargement modèle fine-tuné...")
    ckpt       = torch.load(MODEL_PATH, map_location=device)
    cfg        = ckpt['config']
    _model     = FineTunedMiniLM(cfg['model_name'], cfg['dropout']).to(device)
    _model.load_state_dict(ckpt['model_state'])
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)
    print("✅ Modèle fine-tuné chargé !")
    return _model, _tokenizer


def predict_score(cv_text: str, job_text: str) -> float:
    m, tok = load_model()
    with torch.no_grad():
        r_enc = tok(cv_text,  max_length=MAX_LEN, padding='max_length',
                    truncation=True, return_tensors='pt').to(device)
        j_enc = tok(job_text, max_length=MAX_LEN, padding='max_length',
                    truncation=True, return_tensors='pt').to(device)
        pred  = m(r_enc['input_ids'], r_enc['attention_mask'],
                  j_enc['input_ids'], j_enc['attention_mask'])
    return round(pred.item() * 100, 2)
