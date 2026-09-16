# 題材・履歴経路実験の実行方法

## 変更前 → 変更後

以前は必要情報に対応する4題材が固定されていた。今回、実験専用providerが題材を指定できるようにした。指定しなければ以前と同じ動作。
`poc/ab_poc/task_information_experiment.py` の `InformationInitiator.initiate` に、任意の題材リストと話題履歴固定の入口を追加した。通常の人物選択や関係評価は変更していない。

`poc/ab_poc/task_topic_pathway_experiment.py` は実験専用。`TOPICS` が題材、`PROFILES` が題材と2種類の履歴固定を指定する。通常UIのモデル設定ではない。
`TopicProvider` は承認済みの通常共同作業から全48接触を再生する。
`DiagnosticResponder` は反応履歴を選び、既存 `RuleResponseProvider.reaction_points` をそのまま呼ぶ。
乱数を複製して自然な話題の反応も計算するが、比較用の計算は実際の状態・乱数系列に影響しない。
乱数の複製は `clone_rng` で `getstate/setstate` を使う。再帰的コピーから置き換えた前後で、試運転288条件の保存トレース全体の一致を検証した。
関係は実際の会話履歴だけから従来の評価器で計算する。
情報の受け渡しは前回と同じく社会的な反応と独立して進む。会話が好反応でなかったから情報を渡さない、といった変更はしていない。

履歴固定は質問後だけを集計する方法とは異なる。毎ターン、該当ペアのその時点までの通常共同作業の履歴を読み、質問による履歴の変化が次の判断に使われるのを止める。関係評価の履歴は止めない。未来の履歴は使わない。

## 実行

リポジトリのルートで、既存の仮想環境を使う。追加パッケージ不要。
保存済み `results/task-information-101-1100` が必要。出力先は新規フォルダを指定し、既存結果を上書きしない。
今回の本実験の保存先は `results/task-topic-pathway-final-101-1100`。`results/task-topic-pathway-101-1100` は高速化前に中断した途中記録で、完了結果として使わない。
今回の本計算は2プロセスで約22分だった（保存後監査は別）。所要時間は環境によって変わる。

まず試運転（別名が必要なら末尾を変える）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.task_topic_pathway_experiment --output results/task-topic-retry-smoke --seed-end 102
```

本実験:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.task_topic_pathway_experiment --output results/task-topic-retry-full
```

保存後の独立した読み戻し監査:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_task_topic_pathway.py results/task-topic-retry-full
```

監査通過後の日本語報告生成（結果文書は再生成される）:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/report_task_topic_pathway.py results/task-topic-retry-full
```

## 検証・診断・復帰

`tests/test_task_topic_pathway.py` は全6ペア・全強度の再現、情報・反応の改ざん検知、興味登録外の題材を検査する。
保存後監査は反応点の各項を別計算し、全ターンの関係を再構築し、CSV/JSONと照合する。単なる同一生成コードの再実行だけで合否を決めない。
`design.json` は設定とソースハッシュ、`manifest.json` は完了件数と出力ハッシュを保存。manifestがなければ未完了として扱う。
エラー時は表示されたペア・seedと例外を確認し、出入力ハッシュ不一致ならファイルを変更せず実行版と保存版を照合する。途中からの自動再開は未実装。失敗したフォルダを残し、新しい出力先で再実行できる。
通常の起動方法ではこの実験providerを使用しないため、実験を無効にするにはこの実験コマンドを実行しなければよい。通常UIに新しい効果は加わらない。
保守対象はこのリポジトリ内の実験モジュール・検証・報告であり、外部サービスや追加課金への依存はない。

## 限界

興味登録外の4題材は相互に同等の不一致である。多様な人間の題材理解を表現していない。
履歴固定は機構を切り分ける人工的な操作。反応式を別の仮定に置き換える追試や人間での実験の代わりにはならない。
強条件で履歴経路を調べることを事前に限定した。弱・中の機構分解、他seed・人物への外挿は未確認。
