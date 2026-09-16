# Yoriyoi つながりの実験室

オンラインでの事前のつながりが、オフラインの場での会話や関係にどう影響するかを比べるシミュレーションです。

- 条件A：4人に事前の直接交流がない
- 条件B：2人だけ事前にオンラインで交流している
- 4人・12ターンの会話を比較する

## 起動方法

Python 3.10以上をインストールした環境で、リポジトリのルートから実行します。追加パッケージは不要です。

```text
python -m poc.server --port 8000
```

ブラウザで [http://127.0.0.1:8000/ab-lab/](http://127.0.0.1:8000/ab-lab/) を開いてください。

## シミュレーションの実行

画面で実行範囲、seed、参加者Providerを選び、「再実行」を押します。

- **ルール**：APIキー不要。同じseedでは同じ結果になります。
- **LLM**：DeepSeekによる会話生成を行います。

LLM版を使う場合は、自分のAPIキーを環境変数 `YORIYOI_DEEPSEEK_API_KEY` に設定してからサーバーを起動してください。APIキーはリポジトリへ保存しないでください。

## ローカル追加実験：共通ペアでA/Bを比較する

全6ペアの最終状態、橋渡し候補4ペア、共通新規候補5ペアの比較を保存できます。既存のルールとUIは変更していません。
調査結果・実験結果・初心者向け操作手順は [共通ペア比較の報告](docs/BRIDGE_EXPERIMENT.md) にあります。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.bridge_experiment --seed-start 1 --seed-end 100
```

専用venvがない環境ではPython 3.10以上の `python -X utf8 -m poc.ab_poc.bridge_experiment` でも動きます。追加パッケージとAPIキーは不要です。

## 追加実験：原因経路とボーナスの強さ（段階4・5）

34条件×1100 seedで、会話配分、作用の除去、会話相手の固定、2種類のbonusの感度を調べられます。
[結果と初心者向け手順](docs/MECHANISM_RESULTS.md)、[結果を見る前の計画](docs/MECHANISM_PLAN.md)、[人間への検証に必要な別研究の案](docs/HUMAN_VALIDATION_DRAFT.md) を用意しています。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.mechanism_experiment
```

小さく動作確認する場合は `--seed-start 101 --seed-end 103` を付けます。元のUI・モデル設定は変更していません。

## 追加実験：12・24・36・48ターン × 1000 seed

会話機会だけを増やし、同じ4本のBridgeの形成数と形成までの遅れを比較します。
[結果と操作手順](docs/DURATION_RESULTS.md)、[実行前計画](docs/DURATION_PLAN.md) を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.duration_experiment
```

全8,000条件を実行済み。差は縮小しましたが、48ターンでも残りました。既知ペア変更・滞在施策・人間への効果は未検証です。

## 追加実験：既知ペア6通り × 4時点 × 1000 seed

上記の時間実験に続き、全6通りの既知ペアを比較しました。
[結果](docs/PAIR_IDENTITY_RESULTS.md)、[方法と再実行手順](docs/PAIR_IDENTITY_METHOD.md)、[実行前計画](docs/PAIR_IDENTITY_PLAN.md) を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.pair_identity_experiment
```

全24条件で平均Bridge差は負でした。人物によって効果量と時間推移は異なります。元のモデル・UIは変更していません。

## 追加実験：全既知ペアの作用除去・会話相手固定

全6ペアへ通常・4作用の個別除去・全部除去・A/B相手系列固定の8条件を適用し、12/24/36/48ターン時点を比較します。
[結果](docs/PAIR_MECHANISM_RESULTS.md)、[方法と再実行手順](docs/PAIR_MECHANISM_METHOD.md)、[実行前計画](docs/PAIR_MECHANISM_PLAN.md) を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.pair_mechanism_experiment
```

選択優位を外すと全24ペア時点で差が縮小し、相手系列固定では全seedでBridge差がゼロになりました。人間への効果や唯一の媒介経路は未同定です。

## 追加実験：5種類のイベント形式（24ターン）

自由交流と、シャッフル・単純共同作業・ホスト紹介・共通話題の弱/強を比較します。
[結果](docs/EVENT_FORMAT_RESULTS.md)、[方法と再実行手順](docs/EVENT_FORMAT_METHOD.md)、[実行前計画](docs/EVENT_FORMAT_PLAN.md) を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.event_format_experiment
```

全6既知ペア×1000 seedを実行済み。強いシャッフル/共同作業はBのBridgeを増やしました。A/B差の縮小だけでは施策成功と判断できない結果もあります。通常動作は従来の結果と一致します。

## 追加実験：48ターン・弱/中/強と連続した共同作業

自由交流と4施策×3強度の13条件、全6既知ペア、seed 101〜1100を比較します。
共同作業は承認された6ターン連続ブロック、それ以外は指定された分散配置です。
[実行前計画](docs/EVENT_FORMAT_48_PLAN.md)、[方法と実行手順](docs/EVENT_FORMAT_48_METHOD.md)、
[結果](docs/EVENT_FORMAT_48_RESULTS.md)を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.event_format_48_experiment
```

共同作業とシャッフルの差には時間配置の違いが含まれます。課題の心理的効果や滞在時間延長を測る実験ではありません。

## 追加実験：初めての共同作業と分散情報

通常共同作業の48ターンの接触を固定し、全員が1つずつ情報を持つ状態から
必要情報を質問・取得する条件を比較します。関係への直接加点はありません。
[仕様](docs/TASK_INFORMATION_PLAN.md)、[仕組みと実行手順](docs/TASK_INFORMATION_METHOD.md)、
[結果](docs/TASK_INFORMATION_RESULTS.md)を参照してください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m poc.ab_poc.task_information_experiment
```

情報の必要性が質問の話題を変えるモデルです。新奇性の心理効果そのものは測っていません。

## 追加実験：課題の題材と履歴経路

元の題材、情報への題材割り当て変更、興味登録外の題材を比較します。
強条件では話題履歴・反応履歴を個別に固定し、質問時の反応と後続への波及を切り分けます。
[計画](docs/TASK_TOPIC_PATHWAY_PLAN.md)、[実行方法](docs/TASK_TOPIC_PATHWAY_METHOD.md)、
[結果](docs/TASK_TOPIC_PATHWAY_RESULTS.md)を参照してください。
感謝・達成感の加点は追加していません。通常UIでは新しい実験providerは使いません。

## 追加実験：興味一致と反応確率を揃えた確認必要性

慣れた/初めての共同作業で、話題・接触・興味一致・反応確率を揃え、手順情報の不足と確認必要性だけを変える対照実験です。
[計画](docs/TASK_NEED_MATCHED_PLAN.md)、[実行手順](docs/TASK_NEED_MATCHED_METHOD.md)、[結果](docs/TASK_NEED_MATCHED_RESULTS.md)を参照してください。
情報から話題への接続を使わない現モデルでは、関係形成に別の作用経路がないため、構造上の一致を検証します。人間の新奇性の効果を否定する実験ではありません。

## 追加実験：課題後の情報提供者への再接触

共同作業48ターンの後に自由交流24ターンを追加し、情報提供者への選択重みを抑制/中立/促進する3仮定を比較します。
[計画](docs/POST_TASK_RECONTACT_PLAN.md)、[方法と実行手順](docs/POST_TASK_RECONTACT_METHOD.md)、[結果](docs/POST_TASK_RECONTACT_RESULTS.md)を参照してください。
係数は人の観察値ではなく、モデル内の感度分析です。

人を対象とする新奇性×相互依存の4条件比較は、[未実施の予備調査案](docs/HUMAN_RECONTACT_PILOT_PROTOCOL.md)と[空の記録票](study_materials/recontact_pilot/)を用意しています。人の実測結果はまだありません。

## 追加実験の共有・再現

[実験ガイド](docs/EXPERIMENTS.md)に、これまでの実験と必要な再生成順序をまとめています。[公開用集計CSV](research_results/README.md)と、小さな検証用データを同梱しています。大規模な生ログは`results/`へ再生成します。

最新の[初期共同作業後の再接触実験](docs/EARLY_RECONTACT_RESULTS.md)は、初期6会話の後に提供者への再接触とラベル入れ替えを比較したものです。全6既知ペア・seed 1101〜2100・5条件・A/Bの60,000条件を検証しました。

clone後の検証方法は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。
