# Yoriyoi 共通ペア比較：調査・実装・実験報告

2026-09-10。今回の範囲は、4人・12ターン・ルール版・各条件1試行のローカル実験です。

## まず結果を読む

seed 1〜100では、橋渡し候補4ペアの成立数はAが平均2.33件、Bが2.12件でした。
平均差B−Aは−0.21件、成立率の差は−5.25パーセントポイントです。
「成立」には顔見知りと親しみがある状態の両方を含め、各ペアを1回だけ数えます。

| 比較対象 | 候補数／seed／条件 | A平均件数 | B平均件数 | B−A | A率 | B率 |
|---|---:|---:|---:|---:|---:|---:|
| 橋渡し候補 | 4 | 2.33 | 2.12 | −0.21 | 58.25% | 53.00% |
| 共通の新規候補 | 5 | 2.61 | 2.40 | −0.21 | 52.20% | 48.00% |
| 第三者同士 | 1 | 0.28 | 0.28 | 0 | 28.00% | 28.00% |
| 参照ペア（あかね・みどり） | 1 | 0.21 | 0.83 | +0.62 | 21.00% | 83.00% |

橋渡し候補の成立総数はA=233/400、B=212/400です。
Bが多いseedは1件（66）、同数79件、少ないseedは20件です。
差の分布は−2件が2 seed、−1件が18 seed、0件が79 seed、+1件が1 seedでした。
橋渡し候補の「親しみがある」だけに限ると平均A=1.50件、B=1.19件です。
孤立者が1人以上出たのはA=16 seed、B=15 seedです（孤立者の延べ人数はA=18人、B=15人）。

この100 seedの平均はBridge仮説の増加予想を支持する方向ではなく、Island仮説の「増えない・減る」という予想と整合します。
ただし、交流集中が減少を引き起こしたという媒介機序まではこの集計から確定できません。
人間の行動への一般化、統計的有意性の判定、別の人物設定・ターン数・LLM版への一般化は行っていません。
同じseedを何度実行しても同じ結果なので、それを独立の追加試行として数えません。
集計の比較単位は同じseedのA/Bであり、1200行を独立な1200標本とは扱いません。

## どこを調べたか

対象正本は [RicochChester/yoriyoi-automata-hackathon-vol2](https://github.com/RicochChester/yoriyoi-automata-hackathon-vol2/tree/1cc02b28f850b2843c464f91146fd6b89c34f9a9)、取得revisionは `1cc02b28f850b2843c464f91146fd6b89c34f9a9` です。
今回の作業開始時、親のYoriyoiフォルダーはコードのないGitリポジトリでした。
そこで指定された公開ソースのアーカイブを取得し、このサブフォルダーを作業コピーにしました。
元の手動実験に使ったclone先・commitは未確認です。ただし、提示されたseed 1〜10の結果は全項目一致しました。
この作業コピー自体はGit cloneではありません。出典は `UPSTREAM.json`、元ソースは親フォルダーの `upstream.zip` に保存しています。
上流にAGENTS.md、CONTRIBUTING、CI定義、既存テストスイートはありませんでした。

| 調査対象 | ファイル・関数 | 結論 |
|---|---|---|
| 状態の内部保持 | `poc/ab_poc/domain.py` の `PairState.onsite_relationship` | 1ペア1状態。「未形成」「顔見知り」「親しみがある」 |
| 関係の更新 | 同ファイルの `update_relationship()` | 会話回数、前向き反応数、発話方向から状態を更新 |
| 実行中の保持 | `poc/ab_poc/engine.py` の `run_condition()` 内の `states` | 並び順を統一した2人のIDをキーにした辞書 |
| 終了時の保存 | 同関数の `final_relationships`、`snapshots` の `phase=end` | 未形成を含む全6ペアが既に保存される |
| A/B実行 | 同ファイルの `run_ab()` | 同じseed・人物・ルール・発話順でA/Bを実行 |
| UIの5指標 | `poc/ab_poc/metrics.py` の `calculate_metrics()` | 下記の定義でPythonが計算 |
| 設定 | `poc/ab_poc/config.py` の `load_config()` | 既定で `poc/config/ab_poc_v2.json` を読む |
| Bの既知ペア | 設定の `conditions.B.online_known_pairs` と `online_experience` | あかね・みどりが技術について1往復した事前経験 |
| ブラウザー表示 | `poc/web/ab-lab/ab-lab.js` の指標表示処理 | Pythonから受け取った値を表示 |
| UIとの接続 | `poc/server.py` の `POCRequestHandler.do_GET()` | `/api/ab-poc/compare?seed=1&participant_provider=rule` から同じ `run_ab()` を実行 |

注意：`poc/engine.py` という別のエンジンもあります。今回のAB LABが使うのは `poc/ab_poc/engine.py` です。

### 既存指標の意味

- 新規顔見知り：その条件でオンライン既知でないペアのうち、顔見知り以上のペア数。親しみも含む。
- 新規の親しみ：同じ候補集合のうち「親しみがある」のペア数。
- 既知ペアの対面化：あかね・みどりが条件に既知ペアとして含まれる場合、その最終状態。Aでは該当なし。
- 孤立者：既知・新規を問わず、対面の顔見知り以上の辺を1本も持たない人。オンライン既知だけでは孤立が解消しない。
- 第三者への観察経路：既知ペアの端点から第三者へ成立した対面の辺の一覧。会話イベント数でも、紹介という行為の検出でもない。Bでは今回のBridge成立数と一致する。Aでは既存出力は該当なし。

既存コードは状態が欠けた場合に未形成として扱う箇所があります。新しい出力では全6ペアがなければエラーにし、欠損を未形成として埋めません。

### 既知関係の優位はどこにあるか

`poc/ab_poc/rule_providers.py` と既定設定にあります。

- `RuleTalkInitiator.selection_weight()`：既知の相手に選択重み+1.0。確率そのものに1を足す意味ではない。
- `RuleTalkInitiator._choose_topic()`：相手とのオンライン記憶がある場合、その話題（技術）を優先する。
- `RuleResponseProvider.reaction_points()`：既知関係に反応点+2。慎重な人物の減点を免除する条件にも使う。
- 関係判定の `update_relationship()` 自体は対面イベントから判定し、オンライン既知だけで顔見知りにはしない。

既定の顔見知り判定は会話2回以上・前向き反応1回以上。親しみは会話3回以上・前向き反応2回以上・両者からの発話です。
これは自然法則ではなく、このモデルに設定された判定基準です。
今回測るのはこの事前経験条件全体の差です。ボーナス1種類だけの効果を分離する実験ではありません。

## 何を・なぜ変えたか

問題の原因は、既存指標が条件ごとに候補を除外するため、A=6ペア、B=5ペアになることです。
この指標は各条件内の記述には使えますが、同じペアへの波及効果という問いには直接対応しません。
小さい分母だけで常にBが有利になるわけではありません。重要なのは、数えているペア集合がA/Bで異なることです。

変更前：`run_ab(seed=1)` の結果に全6ペアはあるものの、共通集合の比較表はない。

変更後：新設した `bridge_experiment.py` の `analyze_ab()` が既存結果を読み取り、Bの既知ペアを基準としてA/B共通の分類を付ける。
`run_experiment()` がseedを順番に実行して保存し、`aggregate()` がseedごとの差を集計する。
既存のエンジン、状態更新、設定、指標計算、UI関数は変更していません。

処理の違いを短いコードで示すと次のとおりです。これは理解用で、手作業で貼り付ける必要はありません。

変更前（シミュレーションの結果を得る）：

```python
from poc.ab_poc.engine import run_ab
result = run_ab(seed=1)
```

変更後（同じ結果から共通ペアの比較表も得る）：

```python
from poc.ab_poc.engine import run_ab
from poc.ab_poc.bridge_experiment import analyze_ab
result = run_ab(seed=1)
pair_rows, seed_summary = analyze_ab(result)
```

| 分類 | 両条件で対象にするペア | 候補数 |
|---|---|---:|
| `known_pair` | あかね–みどり | 1 |
| `bridge_edge` | あかね–こはる、あかね–くるみ、みどり–こはる、みどり–くるみ | 4 |
| `third_party_pair` | こはる–くるみ | 1 |
| `matched_new`（集計用） | 上のbridgeとthird_partyを合わせた集合 | 5 |

`known_pair` はAでは「Bで既知となる参照ペア」という意味です。Aにオンライン既知関係があることを意味しません。
CSVの `online_known` はAでは全てFalse、Bでは参照ペアだけTrueです。
Bridgeという分類は候補の区分であり、その辺が紹介によって生まれたことを意味しません。
表示で「こはる–みどり」と「みどり–こはる」が入れ替わっても同じ無向ペアです。IDで順序を統一しています。

変更後はAの参照ペアも分析から除外するため、共通5ペア・Bridge4ペアの分母と対象がA/Bで一致します。
これが、元の分母を表示上だけ書き換える方法と異なり、問いに対応した修正である根拠です。

## 保存したもの

正式な今回の結果は `results/bridge-001-100/` です。`results/seeds-001-100/` は実装途中の先行出力です。

| ファイル | 読み方 |
|---|---|
| `bridge_summary.csv` | 最初に開く比較表。100行。seed、A/B件数、A/B率、B−A |
| `pairs.txt` | 人が読む全6ペアの一覧。seedごと・A/Bごとに表示 |
| `pairs.csv` / `pairs.json` | 1200行＝100 seed × 2条件 × 6ペア。状態と分類、会話回数など |
| `seed_summary.csv` | Bridge、共通5ペア、参照ペア、第三者、旧指標、孤立者の詳細 |
| `summary.json` | 全体の平均・合計・差の分布 |
| `raw_runs.jsonl` | 元の会話・状態・指標を含むA/B実行結果。1行1 seed、全100行、約8.2MB |
| `config.json` | 実行に使用した人物・条件・ルール |
| `manifest.json` | seed、Python版、出典、設定上の候補数、ソース・出力のSHA-256照合値 |

`B_minus_A` はBridgeの「件数差」です。率の差は `bridge_edge_rate_B_minus_A`。
`familiar_count` は親しみだけ、`count` は顔見知り以上です。
`legacy_` の列は旧UIと同じ分母の指標なので、主比較には使わず照合用に残しました。
小数は丸めず保存するため、0.2が0.19999999999999996のように表れる場合があります。件数と分母から確認できます。

CSVはWindowsのExcelで日本語を認識しやすいUTF-8 BOM付きです。
標準の [Python csv.DictWriter](https://docs.python.org/3/library/csv.html#csv.DictWriter) と [json](https://docs.python.org/3/library/json.html) を利用し、外部集計ライブラリは追加していません。
既存のM11集計は別シナリオも扱い、既存指標を中心に集計するため、今回の共通ペア抽出には直接流用せず、実行と集計を分ける構成を踏襲しました。

## PowerShellで実行する（コードの編集は不要）

1. この作業コピーへ移動します。

```powershell
Set-Location "<cloneしたリポジトリのフォルダ>"
```

2. まず1 seedで試す場合は次の1行です。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.bridge_experiment --seed-start 1 --seed-end 1
```

3. seed 1〜100を実行する場合は次の1行です。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.bridge_experiment --seed-start 1 --seed-end 100
```

サーバーの起動は不要です。各seedの件数と最後に保存先が表示されます。毎回日時の異なるフォルダーを作るので過去の結果は残ります。
保存先を指定する場合は `--output results/my-next-experiment` を付けます。既に存在するフォルダーへの出力はエラーで止まります。
補助の `Run-BridgeExperiment.ps1` も同じ処理を呼びますが、PowerShell実行ポリシーの変更は必要ありません。上のPython直接実行を使えます。

4. 今回保存済みの比較表を開く場合は次の1行です。

```powershell
Invoke-Item .\results\bridge-001-100\bridge_summary.csv
```

5. 全6ペアを文字で読む場合は次の1行です。

```powershell
Get-Content .\results\bridge-001-100\pairs.txt -Encoding UTF8
```

元のUIを使う場合も、この作業コピーから次を実行できます（使用中の8000番ポートがあれば先に確認）。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.server --port 8000
```

ブラウザーで `http://127.0.0.1:8000/ab-lab/` を開きます。追加集計の閲覧はCSV/TXTです。既存UIに新しいボタンは追加していません。

今回の `.venv` はCodex同梱Python 3.12.14を元に、このフォルダーだけへ作った環境です。追加パッケージはありません。
この `.venv` は別PCへコピーして使うものではありません。別PCではPython 3.10以上を使って `python -m venv .venv --without-pip` で作り直してください。

## エラーが出たら

- `No module named poc`：最初のSet-Locationでこの作業コピーへ移動できているか確認する。
- Pythonのパスが見つからない：`.venv` の存在を確認する。PC全体のPythonを変更する前に、使用可能な実行ファイルを確認する。
- `output already exists`：既存結果は保護されている。出力先指定を省略するか、新しいフォルダー名を使う。
- `Experiment failed`：そのエラー文を保存し、処理を止める。欠損、設定不整合、書き込み失敗を空の成功結果に置き換えない。
- `manifest.json` がないフォルダー：途中で失敗した未完了の出力。集計結果として採用せず、新しい保存先で再実行する。

今回の取得時にはCodex同梱GitのHTTPS補助プログラム不足が発生したため、GitHubのソースアーカイブ取得に切り替えました。
ネットワーク制限は読み取り取得に限る実行許可で解決済み。PC全体のGit、PATH、認証情報は変更していません。

## 検証・制約・引き継ぎ

- 標準ライブラリのunittest：12テスト成功。別の既存テスト環境でpytestも12テスト成功（8 subtests成功）。
- 提示された手動seed 1〜10：新規件数、親しみ、既知ペア対面化、孤立者、第三者経路を全行照合して一致。
- 旧実行結果：改修前に保存したseed 1〜100のJSON照合値と、改修後の結果が全て一致。
- 元の `poc/` 配下のファイル：取得アーカイブとバイト単位で一致。追加モジュールのみが新規。
- 保存結果：1200ペア行、100 seed、200条件実行。CSV/JSONと元データからの再集計が一致。照合値も確認。
- サービス確認：新しいプロセスで専用 `.venv` を使用し、空きloopbackポートでhealth、AB LAB HTML/JS、比較APIが200。APIのseed=1が直接実行と一致。
- 人によるブラウザー目視と、元のclone先での動作は未確認。手動実験の集計一致を、元cloneの同一性の証明とはしない。
- 対象は4人、Aに既知ペアなし、Bに既知ペア1組の設計。異なる人数や複数既知ペアに対応する実験は別途設計が必要。
- engineering-brainの共通closeoutではpytestは成功したが、Git追跡Pythonがないため総合判定はblocked。ソースアーカイブのための制約で、明示パスのコンパイルとアーカイブ照合でローカル検証を補った。Git公開準備完了とは扱わない。
- engineering-autopilotが参照する `implementation-precedent-research` スキルは見つからず、そのリサーチゲートはholdのまま。代わりに対象正本・既存集計・Python公式機能を直接調査した。ユーザーの安全なローカル作業継続の明示指示に従い、この依存不足を全体の実装停止とはしなかった。
- GitHubへのpush、PR、merge、公開、LLM/API呼び出しは実施していない。

再検証コマンドは1行ずつ実行できます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/verify_artifacts.py
```

```powershell
.\.venv\Scripts\python.exe -X utf8 validation/smoke_http.py
```

証拠は `validation/` に保存しています。実験完了の目印はmanifest、失敗の診断入口はコンソールのエラー文と未完了フォルダーです。
ロールバックは追加コマンドを使うのをやめればよく、既存サーバーやモデルの設定を戻す必要はありません。
ローカル成果物の所有者はこの作業フォルダーの利用者です。上流リポジトリへの採用は上流所有者の判断になります。
今回の依頼範囲の実装・実験は完了。次の研究判断としては、B−Aが減るseed（例2・9）と増えるseed（66）の会話順を比較し、交流集中の説明が実ログと一致するかを見ることを推奨します。
