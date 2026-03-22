# evo_ai 実行ガイド

## 目次

1. [環境構築](#1-環境構築初回のみ)
2. [モデル確認ツール](#2-モデル確認ツール)
3. [v1: EvoTransformer](#3-v1-evotransformer)
4. [v2: HybridTransformer](#4-v2-hybridtransformer)
5. [知識蒸留](#5-知識蒸留-distill_v2py)
6. [会話テスト](#6-会話テスト-chat_v2py)
7. [学習の再開・モデルの拡張](#7-学習の再開モデルの拡張)
8. [停止方法](#8-停止方法)

---

## 1. 環境構築（初回のみ）

Apple GPU（MPS）を使えるよう仮想環境で実行します。

```bash
cd /Users/apple/Documents/evo_ai

# 仮想環境を作成
python3 -m venv .venv

# 仮想環境を有効化
source .venv/bin/activate

# ライブラリをインストール
pip install -r evo_ai/requirements.txt
```

> `.venv/` は `.gitignore` に含まれているのでgitには追跡されません。

### 毎回の準備

```bash
cd /Users/apple/Documents/evo_ai
source .venv/bin/activate
cd evo_ai
```

プロンプトが `(.venv)` から始まれば有効化されています。

---

## 2. モデル確認ツール

学習済みモデルの構成・パラメータ数を確認できます（v1/v2両対応）。

```bash
python check_model.py                        # chat_model.pth を確認
python check_model.py --path chat_model_v2.pth  # v2 モデルを確認
```

出力例：
```
=== chat_model_v2.pth ===
  モデル種別         : HybridTransformer v2 (RoPE+GQA+DeltaNet)
  層数    (n_layer)  : 2
  ヘッド数 (n_head)  : 4
  埋め込み (n_embd)  : 64
  文脈長             : 10000
  KVヘッド数         : 2
  学習可能パラメータ : 100,416
```

---

## 3. v1: EvoTransformer

密結合の小型GPT風モデル（約32,000パラメータ）。

### 3-1. 事前学習

```bash
python pretrain.py
```

- TinyShakespeare（約1MB）を自動ダウンロード
- 500ステップごとに `chat_model.pth` へ保存
- `Ctrl+C` で途中停止しても保存される

### 3-2. ファインチューニング

```bash
python finetune.py
```

- 挨拶などのQAペアで会話パターンを学習
- `pretrain.py` 実行後に行うのを推奨

### 3-3. 自動進化学習（先生AI使用）

```bash
python auto_train.py
```

- Qwen2.5-0.5B が自動でベスト応答を選択（初回 約1GBダウンロード）
- 10世代ごとに保存
- `pretrain.py` → `finetune.py` → `auto_train.py` の順が推奨

### 3-4. 手動育成モード

```bash
python chat_learn.py
```

- 自分がトレーナーとなり、5つの候補から最良の応答を選ぶ
- `exit` で終了・保存

---

## 4. v2: HybridTransformer

RoPE + GQA + GatedDeltaNet + SwiGLU + RMSNorm の最新アーキテクチャ（約100,000パラメータ）。

### 4-1. 事前学習

```bash
python pretrain_v2.py
```

`chat_model_v2.pth` に保存されます。

オプション：
```bash
python pretrain_v2.py --layers 4 --embd 128 --heads 8 --kv_heads 4
python pretrain_v2.py --iters 10000 --lr 1e-4
```

### 4-2. ファインチューニング

```bash
python finetune_v2.py
```

- 挨拶などのQAペアで会話パターンを学習
- 事前学習済みモデルがあれば継続学習

### 4-3. 自動進化学習（先生AI使用）

```bash
python auto_train_v2.py
```

- Qwen2.5-0.5B が評価しながら進化学習

---

## 5. 知識蒸留（`distill_v2.py`）

**Qwen2.5-0.5B（先生）の知識を HybridTransformer v2（生徒）に転移します。**

DeepSeek が大型モデルから小型モデルを作った Output Distillation 方式の実装です。

### 蒸留の2段階

| ステップ | 方式 | 内容 |
|---|---|---|
| ① CPT蒸留 | Continued Pre-Training | Qwen が自由文を生成 → コーパスとして事前学習（言語能力の転移）|
| ② SFT蒸留 | Supervised Fine-Tuning | Qwen がQA応答を生成 → 会話パターンの学習 |

### 実行方法

```bash
# 推奨: CPT → SFT の順で両方実行
python distill_v2.py --mode both

# 個別に実行
python distill_v2.py --mode cpt   # 言語能力の転移のみ
python distill_v2.py --mode sft   # 会話パターンの学習のみ

# 段階的に実行（データ生成 → 学習を分ける）
python distill_v2.py --mode both --gen_only    # Qwen でデータ生成のみ
python distill_v2.py --mode both --train_only  # 生成済みデータで学習のみ

# ステップ数を変更
python distill_v2.py --mode both --cpt_iters 5000 --sft_iters 8000
```

### 生成されるファイル

| ファイル | 内容 |
|---|---|
| `distill_cpt.txt` | CPT用テキストコーパス（Qwen生成）|
| `distill_sft.jsonl` | SFT用Q&Aペア（Qwen生成）|
| `chat_model_v2.pth` | 蒸留済みモデル（更新）|

### なぜ Soft-label 蒸留（logits比較）を使わないのか

Qwen は BPE トークナイザー（語彙数 ~150,000）、HybridTransformer v2 は文字レベルトークナイザー（語彙数 96）のため語彙が異なり logits の直接比較が不可能。そのため出力テキストを学習データとして使う Output Distillation を採用しています。

---

## 6. 会話テスト（`chat_v2.py`）

```bash
python chat_v2.py
```

対話例：
```
You: Hello
AI: Hello! How are you?

You: Thank you
AI: You're welcome!

You: quit
終了します。
```

オプション：
```bash
python chat_v2.py --temp 0.0    # greedy（確定的）
python chat_v2.py --temp 1.2    # 高温度（多様な応答）
python chat_v2.py --max_tokens 80
```

---

## 7. 学習の再開・モデルの拡張

### 学習の再開

`chat_model_v2.pth` が存在する状態でスクリプトを起動すると、自動的に前回から継続されます。

```
Resuming from chat_model_v2.pth...
```

### 推奨フロー（v2）

```
pretrain_v2.py        ← 言語の基礎を学ぶ（TinyShakespeare）
       ↓
distill_v2.py --mode cpt  ← Qwen の言語知識を転移
       ↓
distill_v2.py --mode sft  ← 会話パターンを習得
       ↓
auto_train_v2.py      ← 進化学習でさらに磨く
       ↓
chat_v2.py            ← 会話テスト
```

### 層数を増やして大型化（upscale.py）

```bash
python upscale.py --layers 4   # 2層 → 4層に拡張
```

既存の重みをコピーして層を増やします。`pretrain_v2.py` や `distill_v2.py` で再学習することでより高い性能が期待できます。

---

## 8. 停止方法

全スクリプト共通で `Ctrl+C` を押すと、現在の世代・ステップが完了してからモデルを保存して終了します。

```
^C
Stop requested. Saving after this step...
Model saved to chat_model_v2.pth
```

---

## 動作環境

- Apple Silicon Mac: MPS（Apple GPU）で自動高速化
- その他: CPU動作
- Python 3.8以上
- 先生AIモデル（Qwen2.5-0.5B）: 初回実行時に約1GBダウンロード
