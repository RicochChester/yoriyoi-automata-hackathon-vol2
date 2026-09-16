# 確認必要性だけを変える対照実験の手順

## 何を変えたか

既存の情報移転モデルには、話題変更を使わない `topic_link=False` が既にあった。今回これを大規模な対照実験として実行し、全反応確率と全関係状態の一致を検査する処理を追加した。
新規 `task_need_matched_experiment.py` の `MatchedProvider` がこの設定を使う。
`ProbabilityResponder` は既存の反応式を実行し、乱数を加える前の反応点から全ての反応乱数の結果を数え、確率分布を保存する。診断用の乱数複製は本番の乱数状態を変更しない。
人物・興味・関係閾値・選択ルールは変更していない。情報の受け渡しは前回と同じく社会的反応と独立している。

「慣れた/初めて」の間で話題と接触を揃える。A/B間のオンライン既知関係に由来する差は保持する。
全員が手順を知ることと、実作業を完成させることは区別する。今回の進行指標は必要手順情報の取得状況である。
前回と同じ分散情報仕様を維持し、初めて条件でも各自が異なる断片情報を1つ持つ。`task_familiarity=0` は全情報を失っている意味ではない。情報充足率は各自の残り3情報の取得率を4人で平均した値である。

## PowerShellで実行する

リポジトリのルートで、既存の仮想環境を使う。追加のインストールやAPIキーは不要。
入力は `results/task-information-101-1100`。再実行は必ず新しい出力先を指定する。
今回の本計算は `results/task-need-matched-101-1100` に保存し、2プロセスで約11分だった。保存後監査の時間は別。所要時間は環境によって変わる。

試運転:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.task_need_matched_experiment --output results/task-need-retry-smoke --seed-end 102
```

本実験:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.task_need_matched_experiment --output results/task-need-retry-full
```

保存後監査:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_task_need_matched.py results/task-need-retry-full
```

報告生成（監査通過後、結果文書を再生成）:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/report_task_need_matched.py results/task-need-retry-full
```

## 診断と安全な停止

反応確率・情報移転・関係状態が一致すべきところで一致しないと、実験はassertionで停止する。ログのペア・seed、例外、入力/ソースのハッシュを確認してから原因を調べる。
検証でassertを使うため、`-O` を付けず、上記のコマンドで実行する。
`manifest.json` がないフォルダは未完了。途中からの再開は未実装だが、途中記録を残して新しい出力先で再実行できる。
この実験モジュールを起動しなければ通常のUIには影響しない。追加は同じリポジトリの実験・テスト・報告に限定され、外部サービスや課金への依存はない。
担当範囲は本リポジトリ内の実験実装・保存データ・監査・手順。新しい心理効果の設計や人間を対象とする検証は別途の研究判断が必要である。
