import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from attention_seq2seq_captioning import AttentionCaptioner, Seq2SeqAttention, get_device
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Could not import attention_seq2seq_captioning.py. "
        "Make sure this file is in the same folder as the script or add the correct path to sys.path."
    )

warnings.filterwarnings("ignore")
SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)


class PairDataset(Dataset):
    def __init__(self, source_sentences: List[str], target_sentences: List[str], src_vocab: dict, trg_vocab: dict, sos_token: str = "<sos>", eos_token: str = "<eos>", pad_token: str = "<pad>", unk_token: str = "<unk>"):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

    def _encode(self, sentence: str, vocab: dict) -> List[int]:
        tokens = sentence.split()
        return [vocab.get(token, vocab[self.unk_token]) for token in tokens]

    def _build_tensor(self, sentence: str, vocab: dict) -> torch.Tensor:
        ids = [vocab[self.sos_token]] + self._encode(sentence, vocab) + [vocab[self.eos_token]]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        src_tensor = self._build_tensor(self.source_sentences[idx], self.src_vocab)
        trg_tensor = self._build_tensor(self.target_sentences[idx], self.trg_vocab)
        return src_tensor, trg_tensor


def build_vocab(source_sentences: List[str], target_sentences: List[str]):
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    src_tokens = set(" ".join(source_sentences).split()) | set(special_tokens)
    trg_tokens = set(" ".join(target_sentences).split()) | set(special_tokens)

    src_vocab = {token: idx for idx, token in enumerate(special_tokens)}
    trg_vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for token in sorted(src_tokens - set(special_tokens)):
        src_vocab[token] = len(src_vocab)
    for token in sorted(trg_tokens - set(special_tokens)):
        trg_vocab[token] = len(trg_vocab)

    return src_vocab, trg_vocab


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(x) for x in src_batch]
    trg_lengths = [len(x) for x in trg_batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=False, padding_value=0)
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=False, padding_value=0)
    return src_batch, trg_batch, src_lengths, trg_lengths


def train_seq2seq_real(data_path: str = None, epochs: int = 10, batch_size: int = 8, save_path: str = "seq2seq_model.pt"):
    device = get_device()

    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]
        source_sentences = [line[0] for line in lines]
        target_sentences = [line[1] for line in lines]
    else:
        source_sentences = ["hello world", "this is a test", "good morning", "i love python"]
        target_sentences = ["안녕 세상", "이것은 테스트입니다", "좋은 아침입니다", "파이썬을 사랑합니다"]

    src_vocab, trg_vocab = build_vocab(source_sentences, target_sentences)
    dataset = PairDataset(source_sentences, target_sentences, src_vocab, trg_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqAttention(input_dim=len(src_vocab), output_dim=len(trg_vocab), emb_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for src, trg, src_lengths, trg_lengths in loader:
            src = src.to(device)
            trg = trg.to(device)
            outputs, _ = model(src, trg, teacher_forcing_ratio=0.8, is_train=True)
            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), trg[1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / max(1, len(loader)))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {losses[-1]:.4f}")

    torch.save(model.state_dict(), save_path)
    return losses, model, src_vocab, trg_vocab


def plot_losses(losses: List[float], path: str = "seq2seq_loss.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, marker="o")
    plt.title("Seq2Seq Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_full_pipeline():
    losses, model, src_vocab, trg_vocab = train_seq2seq_real()
    plot_losses(losses)
    print("Saved model to seq2seq_model.pt")
    print("Saved chart to seq2seq_loss.png")
    return losses, model, src_vocab, trg_vocab


if __name__ == "__main__":
    run_full_pipeline()
