# 課題後の再接触：仕組みと実行方法

## 何を変えたか

前の対照実験では、話題・興味一致・好反応確率・接触を揃えると、情報の不足だけでは関係形成が変わらなかった。今回は、情報交換がその後の相手選択に影響すると仮定した場合の感度を測る。人の実測値を当てはめた実験ではない。

既存の共同作業・強条件48ターンをそのまま再現し、その後に24ターンの自由交流を追加する。1ターンは1会話であり、分や滞在時間への換算は行わない。seed 101〜1100、全6既知ペア、A/B、3仮定で36,000条件。人物・関係形成閾値・A/Bの定義は変えていない。

課題部分は、情報から話題への接続を無効にした分散情報条件。4人は各1つの情報から始め、接触に応じて情報を受け取る。課題中の話題・好反応・関係状態は通常共同作業の保存記録と一致させる。48ターン終了時の「誰から何個の情報を受け取ったか」を固定して、自由交流の選択だけに使う。

## 追加した仮定

話し手iが相手jから受け取った異なる情報の個数をcとすると、自由交流の選択重みは次になる。

`既存の選択重み × 2 ** (theta × c / 3)`

|theta|仮定|重みへの倍率|
|---:|---|---|
|-1|情報提供者への再接触を抑える|1〜0.5倍|
|0|再接触への追加作用なし|1倍|
|+1|情報提供者への再接触を促す|1〜2倍|

倍率は選択確率そのものの倍率ではない。他の候補の重みと合計して確率に直す。全候補が同量の情報を提供した場合は倍率が相殺される。方向のある情報授受を使用し、既知ペアやBridge候補を優先しない。感謝や好感度への直接加点はない。

thetaと倍率は未推定の仮定。相手選択が変わった後の話題や反応は、従来ルールに従って変わり得る。各条件で同じseed・ターンの乱数から開始するが、分岐後の会話内容まで同じに固定する実験ではない。

## ファイルと役割

- `poc/config/post_task_recontact_v1.json`: 課題48＋自由24、seed、theta、倍率の設定。
- `poc/ab_poc/post_task_recontact_experiment.py`: 48ターンの再現、24ターンの選択作用、記録・集計。
- `tests/test_post_task_recontact.py`: 中立条件と通常ルールの一致、情報授受なしの場合の一致、改ざん検出。
- `validation/verify_post_task_recontact.py`: 保存結果を読み直し、全ターンの選択・反応・情報・関係状態を再検算。
- `validation/report_post_task_recontact.py`: 検算済み集計から日本語報告を生成。

既存の通常UIは追加providerを呼ばない。実験の無効化はこの追加コマンドを使わないだけでよく、通常ルールを戻す操作は不要。

## 指標と区間

48/72ターン時点のBridge成立数（0〜4）、成立率、自由交流中の新規成立数、4本全成立率を記録する。未成立累積ターンは、各Bridgeが初めて成立する直前までのターン数（最後まで未成立なら72）を4本で平均する。

自由交流24ターンの既知ペア割合、Bridge候補割合、情報提供者への接触割合、選んだ相手からの情報量c/3を記録する。提供者が誰であるかは課題終了時点で固定する。

操作確認として、同じ時点・同じ履歴で、倍率あり/なしの候補分布から計算した期待情報量の差も保存する。これは実装が指定方向へ重みを動かしたかの診断であり、心理的効果ではない。

差は同じseed・既知ペアのtheta=0に対応づける。6ペア平均はseed内で平均してから1000 seed間の標準誤差を求め、6000独立標本とは扱わない。95%区間は正規近似のMonte Carlo区間。個別ペアの主要差は6ペア×2方向×A/Bの24比較、A/B差の変化は12比較のBonferroni区間も併記する。副指標すべてを覆う同時保証ではない。既存seedの再利用なので独立した外部検証ではない。

## PowerShellでの実行

リポジトリのルートで実行する。既存の `.venv` と `results/task-information-101-1100` の保存データが必要。新しいパッケージは不要。

まずテスト:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

小規模実行（出力先はまだ存在しない名前にする）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.post_task_recontact_experiment --seed-start 101 --seed-end 102 --output results/post-task-recontact-check
```

保存結果の再検算:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_post_task_recontact.py results/post-task-recontact-check
```

本実験:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.post_task_recontact_experiment --output results/post-task-recontact-new
```

本実験の再検算:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_post_task_recontact.py results/post-task-recontact-new
```

報告生成:

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/report_post_task_recontact.py results/post-task-recontact-new --report docs/POST_TASK_RECONTACT_NEW_RESULTS.md
```

出力には全会話・全関係状態の圧縮JSONL、seed単位CSV、集計CSV/JSON、ソースと入力のハッシュを残す。100 seedごとに進捗が出る。完了manifestと検算passが両方揃ったものだけを結果として扱う。

## エラー時と制約

`FileExistsError`は上書き防止。既存結果を削除せず、新しい出力名を指定する。ハッシュ不一致やassertionで停止した場合は、入力・ソース変更・ログの該当箇所を確認する。途中結果を完成扱いしない。途中再開機能はない。検証用assertを使うためPythonの`-O`を付けない。

実行中にモデルソース・設定・計画・検算コードを変更しない。設定範囲を変えた別研究は、計画とテストを改めて用意する。48ターンで既に多くのBridgeが成立しており、上限4による天井効果がある。

人の4条件比較は別の未実施の調査。`HUMAN_RECONTACT_PILOT_PROTOCOL.md`と空の記録票を参照する。人の新奇性効果、再接触傾向、滞在時間延長はこの実験から確定しない。
