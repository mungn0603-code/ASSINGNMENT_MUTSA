import random
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import inception_v3


SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.W1 = nn.Linear(encoder_dim, attn_dim)
        self.W2 = nn.Linear(decoder_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        # encoder_outputs: [src_len, batch, encoder_dim]
        encoder_proj = self.W1(encoder_outputs)
        decoder_proj = self.W2(decoder_hidden).unsqueeze(0)
        energy = self.v(torch.tanh(encoder_proj + decoder_proj)).squeeze(-1)
        attention_distribution = F.softmax(energy.t(), dim=1)
        context = torch.bmm(
            attention_distribution.unsqueeze(1),
            encoder_outputs.permute(1, 0, 2),
        ).squeeze(1)
        return context, attention_distribution


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, emb_dim: int = 32, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.encoder_embedding = nn.Embedding(input_dim, emb_dim)
        self.decoder_embedding = nn.Embedding(output_dim, emb_dim)
        self.encoder_lstm = nn.LSTM(emb_dim, hidden_dim)
        self.decoder_lstm = nn.LSTMCell(emb_dim + hidden_dim, hidden_dim)
        self.attention = AdditiveAttention(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor):
        embedded = self.dropout(self.encoder_embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded)
        return encoder_outputs, (hidden, cell)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5, is_train: bool = True):
        src_len, batch_size = src.size()
        trg_len = trg.size(0)

        encoder_outputs, (hidden, cell) = self.encode(src)
        decoder_hidden = hidden.squeeze(0)
        decoder_cell = cell.squeeze(0)

        outputs = torch.zeros(trg_len - 1, batch_size, self.output_dim, device=src.device)
        attentions = torch.zeros(trg_len - 1, batch_size, src_len, device=src.device)

        decoder_input = self.decoder_embedding(trg[0])
        for t in range(trg_len - 1):
            context, attn_weights = self.attention(encoder_outputs, decoder_hidden)
            combined_input = torch.cat([decoder_input, context], dim=1)
            decoder_hidden, decoder_cell = self.decoder_lstm(combined_input, (decoder_hidden, decoder_cell))
            output = self.fc_out(torch.cat([decoder_hidden, context], dim=1))
            outputs[t] = output
            attentions[t] = attn_weights

            if is_train:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                next_token = trg[t + 1] if use_teacher_forcing else output.argmax(dim=1)
            else:
                next_token = output.argmax(dim=1)

            decoder_input = self.decoder_embedding(next_token)

        return outputs, attentions

    def predict(self, src: torch.Tensor, max_len: int = 20, sos_idx: int = 1, eos_idx: int = 2):
        src_len, batch_size = src.size()
        encoder_outputs, (hidden, cell) = self.encode(src)
        decoder_hidden = hidden.squeeze(0)
        decoder_cell = cell.squeeze(0)

        decoder_input = self.decoder_embedding(torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device))
        predictions = []
        for _ in range(max_len):
            context, _ = self.attention(encoder_outputs, decoder_hidden)
            combined_input = torch.cat([decoder_input, context], dim=1)
            decoder_hidden, decoder_cell = self.decoder_lstm(combined_input, (decoder_hidden, decoder_cell))
            output = self.fc_out(torch.cat([decoder_hidden, context], dim=1))
            next_token = output.argmax(dim=1)
            predictions.append(next_token)
            decoder_input = self.decoder_embedding(next_token)
            if (next_token == eos_idx).all():
                break
        return torch.stack(predictions, dim=0)


class AttentionCaptioner(nn.Module):
    def __init__(self, input_vocab_size: int, output_vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = inception_v3(weights=None, aux_logits=False)
        self.backbone.fc = nn.Identity()

        self.feature_dim = 2048
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.fc_h = nn.Linear(self.feature_dim, hidden_dim)
        self.fc_context = nn.Linear(self.feature_dim, hidden_dim)
        self.fc_attn = nn.Linear(self.feature_dim, 1)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def extract_features(self, images: torch.Tensor):
        features = None

        def _hook(_module, _inputs, output):
            nonlocal features
            features = output

        handle = self.backbone.Mixed_7c.register_forward_hook(_hook)
        _ = self.backbone(images)
        handle.remove()
        return features

    def compute_attention(self, feature_map: torch.Tensor):
        batch_size, patch_num, feature_dim = feature_map.size()
        attn_logits = self.fc_attn(feature_map).view(batch_size, patch_num)
        attn_weights = F.softmax(attn_logits, dim=1)
        return attn_weights

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: List[int], teacher_forcing_ratio: float = 0.5, is_train: bool = True):
        features = self.extract_features(images)
        batch_size = images.size(0)
        feature_map = features.view(batch_size, self.feature_dim, -1).permute(0, 2, 1)
        attn_weights = self.compute_attention(feature_map)
        context = (feature_map * attn_weights.unsqueeze(2)).sum(dim=1)
        context = self.fc_context(context)

        h0 = torch.tanh(self.fc_h(features.mean(dim=(2, 3))))
        c0 = torch.zeros_like(h0)
        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)

        if is_train:
            input_tokens = captions[:, :-1]
            target_tokens = captions[:, 1:]
            max_len = input_tokens.size(1)
            embeddings = self.dropout(self.embedding(input_tokens))
            lstm_input = torch.cat([embeddings, context.unsqueeze(1).expand(-1, max_len, -1)], dim=2)
            packed_inputs = pack_padded_sequence(lstm_input, [max(1, l - 1) for l in lengths], batch_first=True, enforce_sorted=False)
            packed_outputs, _ = self.lstm(packed_inputs, (h0, c0))
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
            logits = self.fc_out(outputs)
            return logits, attn_weights

        decoder_input = self.embedding(torch.full((batch_size,), 1, dtype=torch.long, device=images.device))
        h_t, c_t = h0, c0
        outputs = []
        for _ in range(20):
            lstm_input = torch.cat([decoder_input, context], dim=1)
            h_t, c_t = self.lstm(lstm_input.unsqueeze(1), (h_t, c_t))[1]
            logits = self.fc_out(h_t.squeeze(0))
            next_token = logits.argmax(dim=1)
            outputs.append(next_token)
            decoder_input = self.embedding(next_token)
            if (next_token == 2).all():
                break
        return torch.stack(outputs, dim=0), attn_weights


def build_toy_seq_data(num_samples: int = 64, max_len: int = 6):
    srcs = []
    trgs = []
    for _ in range(num_samples):
        src = [random.randint(1, 5) for _ in range(random.randint(2, max_len))]
        trg = [1] + [x + 1 for x in src] + [2]
        srcs.append(torch.tensor(src, dtype=torch.long))
        trgs.append(torch.tensor(trg, dtype=torch.long))

    max_src_len = max(len(x) for x in srcs)
    max_trg_len = max(len(x) for x in trgs)
    src_batch = torch.zeros(max_src_len, len(srcs), dtype=torch.long)
    trg_batch = torch.zeros(max_trg_len, len(trgs), dtype=torch.long)
    for i, (src, trg) in enumerate(zip(srcs, trgs)):
        src_batch[:len(src), i] = src
        trg_batch[:len(trg), i] = trg
    return src_batch, trg_batch


def build_vocab_and_tensors(source_sentences: List[str], target_sentences: List[str], sos_token: str = "<sos>", eos_token: str = "<eos>", pad_token: str = "<pad>", unk_token: str = "<unk>"):
    special_tokens = [pad_token, unk_token, sos_token, eos_token]
    src_vocab = {token: idx for idx, token in enumerate(sorted(set(" ".join(source_sentences).split()) | set(special_tokens)))}
    trg_vocab = {token: idx for idx, token in enumerate(sorted(set(" ".join(target_sentences).split()) | set(special_tokens)))}

    src_vocab = {**{token: idx for idx, token in enumerate(special_tokens)}, **{token: idx + len(special_tokens) for token, idx in src_vocab.items() if token not in special_tokens}}
    trg_vocab = {**{token: idx for idx, token in enumerate(special_tokens)}, **{token: idx + len(special_tokens) for token, idx in trg_vocab.items() if token not in special_tokens}}

    def encode_sentence(sentence: str, vocab: dict) -> List[int]:
        tokens = sentence.split()
        return [vocab.get(token, vocab[unk_token]) for token in tokens]

    src_tensors = [torch.tensor([vocab[sos_token]] + encode_sentence(sentence, src_vocab) + [vocab[eos_token]], dtype=torch.long) for sentence in source_sentences]
    trg_tensors = [torch.tensor([vocab[sos_token]] + encode_sentence(sentence, trg_vocab) + [vocab[eos_token]], dtype=torch.long) for sentence in target_sentences]

    max_src_len = max(len(x) for x in src_tensors)
    max_trg_len = max(len(x) for x in trg_tensors)
    src_batch = torch.zeros(max_src_len, len(src_tensors), dtype=torch.long)
    trg_batch = torch.zeros(max_trg_len, len(trg_tensors), dtype=torch.long)
    for i, (src, trg) in enumerate(zip(src_tensors, trg_tensors)):
        src_batch[:len(src), i] = torch.tensor(src, dtype=torch.long)
        trg_batch[:len(trg), i] = torch.tensor(trg, dtype=torch.long)

    return src_batch, trg_batch, src_vocab, trg_vocab


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_seq2seq_demo():
    torch.manual_seed(SEED)
    device = get_device()
    src, trg = build_toy_seq_data(96, 5)
    src = src.to(device)
    trg = trg.to(device)
    configs = [
        {"emb_dim": 16, "hidden_dim": 16, "teacher_forcing_ratio": 0.5},
        {"emb_dim": 24, "hidden_dim": 24, "teacher_forcing_ratio": 0.8},
    ]
    results = []
    for cfg in configs:
        model = Seq2SeqAttention(input_dim=6, output_dim=8, emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(12):
            model.train()
            current_tf_ratio = max(0.2, cfg["teacher_forcing_ratio"] - epoch * 0.03)
            outputs, _ = model(src, trg, teacher_forcing_ratio=current_tf_ratio, is_train=True)
            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), trg[1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model.predict(src[:, :1], max_len=8)
        results.append((cfg, float(loss.item()), preds[:, 0].tolist()))
    return results


def train_captioner_demo():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    images = torch.randn(batch_size, 3, 299, 299).to(device)
    captions = torch.tensor([
        [1, 3, 4, 2],
        [1, 5, 2, 2],
    ], dtype=torch.long).to(device)
    lengths = [4, 4]
    cfgs = [
        {"embedding_dim": 32, "hidden_dim": 64, "dropout": 0.0},
        {"embedding_dim": 48, "hidden_dim": 96, "dropout": 0.1},
    ]
    results = []
    for cfg in cfgs:
        model = AttentionCaptioner(1000, 10, embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(3):
            model.train()
            logits, _ = model(images, captions, lengths, is_train=True)
            logits = logits.reshape(-1, logits.size(-1))
            targets = captions[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        results.append((cfg, float(loss.item())))
    return results


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", get_device())
    print("Running Seq2Seq attention tuning demo...")
    seq_results = train_seq2seq_demo()
    for cfg, loss, preds in seq_results:
        print({"config": cfg, "loss": round(loss, 4), "sample_prediction": preds})

    print("\nRunning captioning tuning demo...")
    cap_results = train_captioner_demo()
    for cfg, loss in cap_results:
        print({"config": cfg, "loss": round(loss, 4)})

    print("\nExample real-data preparation:")
    print("source_sentences = ['hello world', 'this is a test']")
    print("target_sentences = ['안녕 세상', '이것은 테스트입니다']")
    print("src, trg, src_vocab, trg_vocab = build_vocab_and_tensors(source_sentences, target_sentences)")
