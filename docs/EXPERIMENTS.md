# 追加実験ガイド

このPRは、事前オンライン関係と第三者への対面関係（Bridge）について、これまでに作成したモデル内実験をまとめたものです。人物・関係成立閾値・通常UIの挙動を保ち、比較対象のペアをA/Bで揃えます。

## 最初に読むもの

- 最新の追加検証：[初期共同作業後の再接触](EARLY_RECONTACT_RESULTS.md)
- 一般向けの全体説明：[記事原稿](NOTE_ARTICLE_YORIYOI.md)（執筆時点までの結果。最新の再接触結果は上記を参照）
- 機械可読な集計：[research_results](../research_results/README.md)
- テスト・CI：[CONTRIBUTING.md](../CONTRIBUTING.md)

## 実験の順序と内容

|段階|調べたこと|結果|実装モジュール（poc.ab_poc配下）|
|---|---|---|---|
|1|A/Bで同じ4本のBridgeを比較|[指標と調査](BRIDGE_EXPERIMENT.md)|bridge_experiment|
|2|12・24・36・48ターン|[時間](DURATION_RESULTS.md)|duration_experiment|
|3|相手選択・作用除去|[作用経路](MECHANISM_RESULTS.md)|mechanism_experiment|
|4|既知ペアを全6通り変更|[組み合わせ](PAIR_IDENTITY_RESULTS.md)|pair_identity_experiment|
|5|全既知ペアで作用を除去|[組み合わせと作用](PAIR_MECHANISM_RESULTS.md)|pair_mechanism_experiment|
|6|5種類のイベント・24ターン|[イベント24](EVENT_FORMAT_RESULTS.md)|event_format_experiment|
|7|48ターン・弱/中/強|[イベント48](EVENT_FORMAT_48_RESULTS.md)|event_format_48_experiment|
|8|共同作業での分散情報と質問|[情報](TASK_INFORMATION_RESULTS.md)|task_information_experiment|
|9|課題の題材・話題と反応の履歴経路|[題材と履歴](TASK_TOPIC_PATHWAY_RESULTS.md)|task_topic_pathway_experiment|
|10|興味一致・反応確率を固定|[対応条件](TASK_NEED_MATCHED_RESULTS.md)|task_need_matched_experiment|
|11|課題後の情報提供者への再接触|[課題後](POST_TASK_RECONTACT_RESULTS.md)|post_task_recontact_experiment|
|12|初期6会話後・提供者ラベル対照|[初期再接触](EARLY_RECONTACT_RESULTS.md)|early_recontact_experiment|

各結果文書に対応する`*_METHOD.md`と`*_PLAN.md`が手順・仮定の正本です。過去の文書内の`results/`参照は、当時のローカル実験生成物です。全文ログやローカル運用記録はPRに同梱していません。元のソース変更前コピー・差分は`validation/event-source-before/`と関連patchにあります。

## 新しい環境での検証

Python 3.10以上。追加パッケージは不要です。リポジトリのルートから順に実行します。

```text
python -m unittest discover -s tests -v
python validation/smoke_http.py
python -m poc.ab_poc.early_recontact_experiment --seed-start 101 --seed-end 102 --workers 2 --output results/early-check
python validation/verify_early_recontact.py --output results/early-check --report validation/early-check-audit.json
```

初期再接触実験は他の大規模データなしで実行できます。本実験の既定seedは1101〜2100、全6既知ペア・5条件・A/Bで60,000条件です。別の未使用出力名を指定してください。

## 過去実験の再生成に必要な順序

共同作業の比較系列は、まず`event_format_48_experiment`で`results/event-formats-48-101-1100`を生成し、次に`task_information_experiment`で`results/task-information-101-1100`を生成します。題材・対応条件・課題後再接触は、この情報実験の保存データを参照します。各モジュールの`--help`とMETHODの出力名を確認してください。

テストはこの大規模生成を要求しません。テスト用データは、当時の検証済み記録の抜粋をハッシュで固定しています。モデルを変えたときに、自動でテスト用の正解データを書き換えないでください。

元のsource archiveに依存する`validation/verify_artifacts.py`等は、当時の生成物も用意した上で使う履歴監査です。軽量CIは、回帰テスト・HTTP・小規模の新規生成と全保存結果の再検算を実行します。本実験全量をCIで再実行したとは扱いません。

## 解釈と未実施事項

1ターンは1会話です。現実の分数や滞在時間とは換算していません。人への一般化や、「情報提供者を避けるべき」等の提言は行いません。

人の予備調査文書・空CSVも当時の準備物として収録していますが、実施しておらず、今回のモデル実験にも必要ありません。接触を配る時期と人物特性の変更は[NEXT_MODEL_ONLY_EXPERIMENTS.md](NEXT_MODEL_ONLY_EXPERIMENTS.md)の次段階です。
