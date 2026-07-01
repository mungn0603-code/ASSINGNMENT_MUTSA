import unittest

import torch

from attention_seq2seq_captioning import AttentionCaptioner, Seq2SeqAttention


class AttentionModelTests(unittest.TestCase):
    def test_seq2seq_forward_shape(self):
        model = Seq2SeqAttention(input_dim=8, output_dim=10, emb_dim=16, hidden_dim=16)
        src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        trg = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.long)
        outputs, attentions = model(src, trg, teacher_forcing_ratio=0.0, is_train=True)
        self.assertEqual(outputs.shape, (2, 3, 10))
        self.assertEqual(attentions.shape, (2, 3, 2))

    def test_captioner_forward_shape(self):
        model = AttentionCaptioner(1000, 20, embedding_dim=16, hidden_dim=32)
        images = torch.randn(2, 3, 299, 299)
        captions = torch.tensor([[1, 3, 2], [1, 4, 2]], dtype=torch.long)
        logits, attn = model(images, captions, [3, 3], is_train=True)
        self.assertEqual(logits.shape, (2, 2, 20))
        self.assertEqual(attn.dim(), 2)
        self.assertEqual(attn.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
