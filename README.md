# 学習の起動
/Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/venv/bin/python /Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/train.py

# インタラクティブ進化 (会話学習)
/Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/venv/bin/python /Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/chat_learn.py
- 保存先: `chat_model.pth`
- 終了時に自動保存されます。
- また、5世代ごとに自動保存されます。

# 会話してみる
/Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/venv/bin/python /Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/interact.py  


# 自動学習 (LLM Judge)
/Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/venv/bin/python /Users/itakuranobunosuke/Documents/antigravity_pj/evo_ai/auto_train.py
- **概要**: 0.5Bの軽量LLMが「良い返答」を自動で選別し続けます。
- **保存先**: `chat_model.pth` (インタラクティブ学習と共通)
- **注意**: 初回起動時にLLM (Qwen2.5-0.5B) のダウンロードが走ります。

初回: 1GB程度の先生モデルのダウンロードが行われます。
動作:
・システムが自動で「Hello」などの問いかけをします。
・赤ちゃんAIが5つの答えを出します。
・先生AIが「これが一番いい」と判断して選抜します。
・これを無限に繰り返します（Ctrl+C で止めるまで）。
就寝前などに実行しておけば、朝には少し賢くなっているかもしれません。 （もちろん、時々 chat_learn.py であなた自身が様子を見てあげることもできます）

