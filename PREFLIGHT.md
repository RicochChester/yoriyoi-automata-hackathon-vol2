<!-- repo-preflight:review-record -->
# 追加実験PRの検証記録

2026年9月16日。対象baseは`RicochChester/yoriyoi-automata-hackathon-vol2`の`additional/hibiki-experiments`、コミット`1cc02b28f850b2843c464f91146fd6b89c34f9a9`。headは`codex/hibiki-experiments`の今回の差分。

## 確認済み

- 元の実験環境とは別のcloneで69テスト成功。大規模な`results/`は不要。
- HTTPのhealth、UI、JavaScript、ルール比較APIが200。API出力とPythonの`run_ab(seed=1)`が一致。
- 初期再接触実験を120条件実行し、5,760会話、180指標行、1,260集計行、84対比行を保存後監査。
- 最新の本実験は元の環境で60,000条件・2,880,000会話の保存後監査と38,220項目の独立集計照合が成功。過去実験の結果文書・集計CSVも同梱。
- 通常モデルへの変更は、介入時の話し手選択hookとshared-interestの拡張点。人物・関係閾値・通常UIを変更していない。
- 公開用の文章から個人PCの絶対パスを除去。人の観察記録・APIキー・仮想環境・全量生ログは含めない。
- CIは読み取り権限、公式ActionsのSHA固定、外部APIなし。大規模本実験はCI対象外。

## 機械検査と人間判断の区別

ローカル`repo-preflight`は、実験IDの`task-...`内の部分文字列を`sk-...`型秘密情報と誤検出する。候補は実験ファイル名・識別子との対応を確認する。これはAPIキーの検出を無視したという意味ではない。

元のbaseにはLICENSEがない。今回の追加実験PRで、提出先全体のライセンスを独断で新設しない。ライセンスの選択は管理者の判断事項として残す。

同梱するSECURITY/CONTRIBUTINGは、データの扱いと検証手順を文書化したもの。GitHub側の非公開報告機能、branch保護、required checksを設定したものではない。

## 未確認・外部依存

- GitHub Actionsの実行結果は、PR作成後のcheckを正本とする。初回の外部貢献で管理者による実行承認が必要になる場合がある。
- 人間のPRレビューとmerge判断は未実施。本記録はmerge承認ではない。
- 人間の心理効果・実際の滞在時間は検証していない。

無効化は追加実験を使わないことで可能。PR全体を戻す場合は、この追加のcommitをrevertする。既存履歴の書き換えは行わない。
