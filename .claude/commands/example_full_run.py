import os
import sys
from pathlib import Path

# 현재 파일 기준으로 sibling 모듈 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import real_training_pipeline as pipeline


if __name__ == "__main__":
    print("Starting full example run...")
    losses, model, src_vocab, trg_vocab = pipeline.run_full_pipeline()
    print("Done.")
    print("Loss history:", losses)
    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(trg_vocab))
