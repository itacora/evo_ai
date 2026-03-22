# evo_ai — 進化するAIを育てるプロジェクト

小さなTransformerを**進化的アルゴリズム**・**勾配降下法**・**知識蒸留**で育てる実験プロジェクトです。

## モデル構成

### v1: EvoTransformer（密結合Transformer）

```
vocab_size=96 (ASCII文字レベル)
n_embd=32, n_layer=2, n_head=2, block_size=32
パラメータ数: 約32,000 (0.032M)
```

GPT-2（117M）の約3,600分の1。ファイルサイズ約128KB。

### v2: HybridTransformer（最新アーキテクチャ）

```
RoPE + GQA + GatedDeltaNet + SwiGLU + RMSNorm
n_embd=64, n_layer=2, n_head=4, n_kv_head=2
パラメータ数: 約100,000 (0.1M)
最大コンテキスト長: 10,000トークン
```

Qwen3 / Llama で採用されている最新技術をフルスクラッチ実装。

| 技術 | 役割 |
|---|---|
| RoPE | トークン間距離を回転行列で表現する位置エンコーディング |
| GQA | QヘッドよりKVヘッドを減らしてメモリ効率を改善 |
| GatedDeltaNet | O(n)メモリで動く線形Attention |
| SwiGLU | Llama/Qwen系FFN活性化関数 |
| RMSNorm | LayerNormより軽量な正規化 |

---

## スクリプト一覧

### v1（EvoTransformer）

| スクリプト | 内容 |
|---|---|
| `pretrain.py` | TinyShakespeareで事前学習 |
| `finetune.py` | QA形式でファインチューニング |
| `auto_train.py` | Qwen2.5-0.5B（先生AI）による自動進化学習 |
| `chat_learn.py` | 人間が応答を選んで手動で育てる |

### v2（HybridTransformer）

| スクリプト | 内容 |
|---|---|
| `pretrain_v2.py` | TinyShakespeareで事前学習 |
| `finetune_v2.py` | QA形式でファインチューニング |
| `auto_train_v2.py` | Qwen2.5-0.5B（先生AI）による自動進化学習 |
| `distill_v2.py` | **知識蒸留**（CPT + SFT）Qwen → HybridTransformer |
| `chat_v2.py` | インタラクティブ会話 |

### ユーティリティ

| スクリプト | 内容 |
|---|---|
| `check_model.py` | モデル構成・パラメータ数の確認（v1/v2両対応）|
| `upscale.py` | モデルの層数を増やして大型化 |

---

## 知識蒸留について

`distill_v2.py` は DeepSeek が大型モデルから小型モデルを作った方式と同じ **Output Distillation** を実装しています。

```
① CPT蒸留 (Continued Pre-Training)
   Qwen2.5-0.5B が自由文を生成 → テキストコーパスとして事前学習
   → 言語能力そのものを転移

② SFT蒸留 (Supervised Fine-Tuning)
   Qwen2.5-0.5B がQ&A応答を生成 → 会話パターンを学習
   → 特定の応答スタイルを習得
```

詳細は [HOW_TO_RUN.md](HOW_TO_RUN.md) を参照。

---

## クイックスタート

```bash
# 仮想環境セットアップ（初回のみ）
python3 -m venv .venv
source .venv/bin/activate
pip install -r evo_ai/requirements.txt

# v2モデルを蒸留で育てる（推奨フロー）
cd evo_ai
python pretrain_v2.py          # 事前学習
python distill_v2.py --mode both  # CPT + SFT 蒸留
python chat_v2.py              # 会話テスト
```

---

## 動作環境

- Apple Silicon Mac: MPS（Apple GPU）で自動高速化
- それ以外: CPU動作
- Python 3.8以上
- 初回実行時: 先生AIモデル（Qwen2.5-0.5B、約1GB）を自動ダウンロード
