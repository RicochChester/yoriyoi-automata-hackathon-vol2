# 初期共同作業後の再接触：実験方法と実行手順

2026年9月15日。設計の正本は `NEXT_MODEL_ONLY_EXPERIMENTS.md`。今回実装した範囲は、その最優先の5条件比較。接触タイミング・人物特性の追加実験は含めない。

## 変更前と変更後

前回は48ターンの共同作業条件の後に24ターンを追加した。課題終了時に既に多くの関係が成立していたため、今回は最初の6会話で全6ペアを1回ずつ接触させ、その後の42会話で関係の形成を観測する。

- 4人、全6通りの既知ペア、A＝事前直接関係なし、B＝1組あり。人物・興味・慎重さ・オンライン経験・反応式・関係成立閾値は既存設定のまま。
- 1ターン＝1会話。第1〜6ターンだけ運営が話し手と相手を指定する。相手順と方向は既存の乱数規則を再利用し、同じseedなら全条件・A/Bで同一。個々人の話し手回数まで均等にする設計ではない。
- 第7〜48ターンは通常の話し手順に戻る。選択重みだけに実験倍率を掛ける。
- 情報は各人1個から開始。既存の情報移転規則を使い、話題への接続は無効。好反応・好感度・関係への直接加点はない。
- 情報授受は課題6会話内だけ。第6ターン終了時の方向付き情報提供履歴を固定する。
- seedは1101〜2100。事前の27設計ファイルのseed記録では1100まで。新しいseedも、人間社会への外部検証にはならない。

## 5条件と仮定

|条件ID|内容|theta|
|---|---|---:|
|neutral|通常の相手選択|0|
|actual_promote|実際の情報提供者への重みを増す|1|
|actual_suppress|実際の情報提供者への重みを減らす|-1|
|permuted_promote|提供者ラベルを入れ替えて重みを増す|1|
|permuted_suppress|提供者ラベルを入れ替えて重みを減らす|-1|

`新しい選択重み = 通常の選択重み × 2 ** (theta × c / 3)`

cは、その話し手が相手から受け取った異なる情報の個数。ラベル入れ替え条件では、候補3人のcのベクトルをseedに基づいて並べ替える。写像は全自由交流を通じて固定し、A/Bでも同一。元の情報・会話履歴は変更しない。同じ並びになる場合も残して、その割合を記録する。

**今回の6接触では各ペア1回だけなので、cは0または1。実際に使う最大倍率は約1.260、最小倍率は約0.794になる。** 前回と式は同じでも、情報量が少ないため作用の実効的な幅は同じではない。倍率は選択確率そのものの倍率ではない。

同じ倍率の集合でも、通常の候補重みとの対応が違えば分布変化量も違う。そのため候補確率の全変動距離と、実際の情報提供者への期待接触量の変化も記録する。この比較だけで感謝や新奇性という人間の心理作用を同定したとは扱わない。

## 評価と保存項目

48ターンの1実行から12・24・48ターンを取り出す。主評価は24ターン。5条件×6既知ペア×1000seed×A/B＝60,000条件実行、2,880,000会話。途中3時点を独立試行として数えない。

主指標は既知ペアの両端から第三者へ伸びる4本のBridge成立数と、4で割った率。AでもBと同じ4本を評価する。顔見知り以上を成立とする。

比較は、Bの各条件−中立、Aの各条件−中立、B−A、その差の変化、実際の提供者条件−ラベル入れ替え条件。副指標は以下。

- `unformed_turns`：成立直前までの累積ターン数を4本で平均。最後まで未成立なら評価時点のターン数。
- `all_formed`：4本すべて成立。
- `free_known_share` / `free_bridge_share`：課題終了後の会話に占める既知ペア／Bridge候補の割合。
- `source_contact_share`：実際に情報を受け取った相手への自由交流の割合。入れ替え条件でも実際の提供者を数える。
- `unformed_bridge_contacts`：会話直前にはまだ未成立だったBridge候補への自由交流回数。成立する当該会話も含む。
- `selection_total_variation`：同じ履歴の通常重みと介入重みの候補確率分布の距離（0〜1）。
- `expected_source_shift`：同じ履歴での確率変化による、実際の受領量c/3の期待値の差。
- `permutation_unchanged_fraction`：候補3人のcを入れ替えてもベクトルが変わらない話し手の割合。中立・実際条件でも同じ対照写像を診断用に評価。
- `information_transfers`：課題中の移転件数。

区間は同じseedの対応差から正規近似で計算した95% Monte Carlo区間。6ペア集約はseed内平均の後に1000seed間の区間を計算し、6000独立標本にはしない。

主評価24ターンのBについて、4条件対中立と実際対入れ替え2対比、計6対比×6ペア＝36対比を補正系列とする。集約値、A、別時点、副指標は探索的。CSVの他行にも便宜上同じ36比較の係数で区間を保存するが、すべての行を一括保証する区間ではない。差の区間が0を含むことを同等性の証明とはしない。

## 実装の置き場所

- `poc/ab_poc/early_recontact_experiment.py`：新規の実験ラッパー。5条件・3評価時点・重み倍率・情報履歴・実行と集計を担当。設定は同ファイルの定数と、出力の`design.json`に保存する。
- `validation/verify_early_recontact.py`：保存後監査。既存の関係評価、通常の選択重み、話題選択と反応式を利用して、全ターンを再検算。
- `tests/test_early_recontact.py`：通常モデルとの中立一致、全条件の課題部分一致、情報授受なしの一致、破損データの検出、指標の検算。
- `validation/report_early_recontact.py`：監査済みの集計から日本語報告を生成。

既存のエンジン・人物設定・UI・関係形成ロジックは編集していない。今回の実験コマンドを使わなければ従来の動作になる。無効化のために設定を戻す必要はない。

## PowerShellから再実行

リポジトリのルートに移動して、1コマンドずつ実行する。既存`.venv`を使用し、新しいパッケージは不要。出力先は未使用の名前にする。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.early_recontact_experiment --seed-start 101 --seed-end 102 --output results/early-recontact-check
```

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_early_recontact.py --output results/early-recontact-check --report validation/early-recontact-check-audit.json
```

本実験：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.early_recontact_experiment --output results/early-recontact-new --workers 2
```

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_early_recontact.py --output results/early-recontact-new --report validation/early-recontact-new-audit.json
```

報告生成：

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/report_early_recontact.py --output results/early-recontact-new --audit validation/early-recontact-new-audit.json --report docs/EARLY_RECONTACT_NEW_RESULTS.md
```

実行中は50seedごと、監査中は100seedごとに進捗が出る。圧縮JSONLに全会話・選択・情報・反応・関係状態、CSVに各seedの指標を保存する。完了manifestと保存後auditのpassがそろったものを完成データとする。

既存フォルダへの書き込みは`FileExistsError`で停止し、上書きしない。途中再開機能はない。エラー時はログの末尾とソース／出力ハッシュを確認し、原因を直して別の出力先へ実行する。完了前に結果を採用しない。実験中はソース・設定・計画・監査コードを変更しない。assertによる検証を使うため`-O`を付けない。

## 解釈の範囲

未推定の相手選択仮定に対するモデル内の感度実験である。前回とは課題量、情報取得量、介入開始時点、seedが違う。結果の違いを「天井効果が解消したから」だけに帰属させない。今回も、人が実際に長く滞在するかは測っていない。
