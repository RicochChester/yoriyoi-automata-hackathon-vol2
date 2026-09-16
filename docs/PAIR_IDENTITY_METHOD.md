# 既知ペアを変える実験：方法と操作

## 何を調べるか

同じ4人のうち、誰がオンラインで知り合っていたかだけを変える。全6通りについて12・24・36・48ターン、各1000 seed（101〜1100）、A/Bを比較する。計48,000条件実行、288,000ペア行。seedは前回と共通で独立した新規追試ではない。

Aには事前交流がない。Bでは選んだ2人に「技術について1往復した」という同じ事前交流を与える。元のボーナス、話題、会話のルール、人物の属性は固定する。AでもBと同じ4本をBridgeとして集計する。既知ペアの変更に伴いBridgeの4本が変わるため、Aの形成数も条件ごとに異なり得る。Aの世界自体を変えたわけではない。

事前計画は [PAIR_IDENTITY_PLAN.md](PAIR_IDENTITY_PLAN.md)。正本は現在のローカルソースと実験フォルダの設定・SHA-256。元GitHub取得revisionは `1cc02b28f850b2843c464f91146fd6b89c34f9a9`。現在の作業コピーはGit追跡未登録なので、Gitコミットだけを再現性の根拠にしていない。

## 変更前 → 変更後

変更前は、実験用設定のBを「あかね―みどり」で固定していた。

```python
config = duration_config(turns)
```

変更後は、新規の `poc/ab_poc/pair_identity_experiment.py` の `pair_config` が、既知ペアとオンライン記憶の持ち主を同時に入れ替える。

```python
experience = replace(b.online_experience, participants=pair, id=experience_id)
b = replace(b, online_known_pairs=(pair,), online_experience=experience, label=label)
config = replace(config, conditions=(a, b))
```

関係フラグだけを変えると、記憶が元の2人に残ってしまう。それを避け、同じ経験を新しい2人に割り当てた。経験IDと表示ラベルは名前に合わせるが、話題や強さは変えない。元のペアの場合は前回の設定をそのまま返す。

同ファイルの `run_study` が6条件を実行し、`combined` が全24点とペア条件間60比較をまとめる。前回の `duration_experiment.py` の分類・時系列抽出・集計を再利用し、エンジン、通常UI、JSON loaderは編集していない。

## 統計と解釈

主指標は各seedのBridge成立数差B−A。顔見知りと親しみの両方を成立に含む。最大4本。

全24点には、通常95%区間に加え、24比較を一つのfamilyとするBonferroni補正の近似同時区間を保存する。係数は標準正規分布の `1−0.05/(2×24)` 分位点。1000個のseed内差の標準誤差を使う大標本近似であり、有限標本の厳密保証ではない。[NISTのBonferroni法](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm) の多重区間の考え方を適用した。

人物による効果量の違いは、各時点で6条件から2条件を選ぶ15通りについて「左条件のB−A」−「右条件のB−A」をseed内で計算する。4時点×15=60比較は別familyとして補正。複数family全体を一つの95%保証とは呼ばない。

比較は同じseedを対応させる。6条件や4時点を独立標本として人数を水増ししない。副指標は既知ペアへの会話割合、親しみ、全4本成立率、初成立時点、観察期間内の未形成回数。未成立の時点は空欄と打ち切りフラグで保存する。

今回分離しないものは、既知ペアの性格、興味、固定した技術話題との適合、第三者の構成、参加者IDと順序の影響である。人物組合せによる違いが見つかっても、性格だけが原因とは言えない。

## ファイルと再実行

結果の正規保存先は `results/pair-identity-101-1100/`。最上位の `summary.csv/json` は24点、`contrasts.csv/json` は60比較。各ペア名のサブフォルダに `pairs.csv`、`seed_metrics.csv`、`traces.jsonl.gz`、`design.json`、`manifest.json` を保存する。ペアの識別はフォルダ名で保持し、全体集計には `known_pair` 列も入る。CSVはUTF-8 BOM付き。

PowerShellで次を一つずつ実行する。

```powershell
Set-Location "<cloneしたリポジトリのフォルダ>"
```

少数で確認する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.pair_identity_experiment --seed-start 101 --seed-end 103
```

全1000 seedで実行する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.pair_identity_experiment
```

今回の保存済み本実験を検証する。

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_pair_identity.py
```

出力先を省略すると新しい時刻付きフォルダへ保存し、既存結果は上書きしない。少数試行にはテスト内の再構成検証を使用しており、最後の監査コマンドは前回の1000 seed全件との一致を確認する本実験向け。

途中エラー時はメッセージとフォルダを残し、最上位manifestがない実行を完了と扱わない。新しい出力先で再実行できる。無効化は追加コマンドを使わないだけでよく、元モデルの復元は不要。保守と結果の所有先はこのローカルYoriyoiプロジェクト。外部API、LLM、課金、公開は使用しない。

## 検証の範囲

`tests/test_pair_identity_experiment.py` は6条件の設定、A不変、1/4/1分類、再構成、上書き拒否、多重区間を確認する。`validation/verify_pair_identity.py` は既存のduration監査を全6フォルダへ適用し、全状態を会話から再計算する。さらにA全記録の一致、元ペアの前回全記録との一致、24推定と60比較の再集計を検証する。

33テスト成功。別の共有検証環境のpytestでも33テストと8 subtests成功。コンパイル確認成功。元の43ソースファイルと前の100 seed成果物も再検証で一致した。

統合closeoutはGit追跡済みPythonファイルがないこと、公開向け個人パス検査などでblockedのまま。ローカル実験の検証と公開準備は区別する。先行実装調査用依存スキルの不足も解消済みとはしない。Git設定変更、push、公開は実施していない。
