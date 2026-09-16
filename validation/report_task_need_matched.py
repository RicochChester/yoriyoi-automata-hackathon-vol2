"""Build a readable, explicitly structural negative-control report."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.bridge_experiment import _sha256, _write_json

def report(output):
    out=Path(output).resolve()
    audit=json.loads((ROOT/'validation'/f'{out.name}-audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='pass' and audit['manifest_sha256']==_sha256(out/'manifest.json')
    design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    by={(r['known_pair'],r['case_id'],r['metric']):r for r in data}
    n=design['end']-design['start']+1
    lines=['# 興味一致・話題・反応確率を揃えた確認必要性の比較','',
        f"48会話、4人、全6既知ペア、seed {design['start']}–{design['end']}。共同作業の弱・中・強で、慣れた条件と初めて条件をA/Bそれぞれ比較した。",'',
        '## 結果の要点','',
        '**確認の必要性と情報取得は変わったが、慣れた条件と初めて条件のBridge差は全試行でゼロだった。** 話題・接触・興味一致・反応確率を揃えると、現在のモデルには情報の不足が対人関係へ影響する別の経路が残らない。', '',
        'この結果は「人間では初めての共同作業に効果がない」という証拠ではない。新奇性の心理効果を実装していないモデルで、作用経路を外したときの整合性を確かめた結果である。単なる有意差なしではなく、既存式から一致が予測される構造上のゼロである。', '',
        '## 1. Bridge形成','',
        '成立数は4本が上限。以下は同一seed内で6ペアを平均してから集計した値。慣れた/初めて間の差を示しており、A/B差がゼロという意味ではない。', '',
        '|強度|慣れたA|初めてA|慣れたB|初めてB|初めて−慣れた B|',
        '|---|---:|---:|---:|---:|---:|']
    for case,label in (('task_weak','弱'),('task_medium','中'),('task_strong','強')):
        r=by['pooled',case,'bridge_count']
        lines.append(f"|{label}|{r['familiar_A_mean']:.4f}|{r['novel_A_mean']:.4f}|{r['familiar_B_mean']:.4f}|{r['novel_B_mean']:.4f}|{r['novel_minus_familiar_B_mean']:.4f}|")
    lines+=['','Bridge率、未成立だった累積ターン数の4本平均、既知ペア/Bridge候補への会話割合、全4本成立率も、慣れた/初めて間で全試行一致した。差のMonte Carlo区間は[0,0]になるが、人間の効果の不確実性がなくなったことを意味しない。','',
        '## 2. 本当に変わったもの','',
        '慣れた条件では全員が全4情報を知っている。初めて条件では各自1情報のみを持ち、残りの手順情報は他者への確認で取得する。相手が必要情報を持っていなければ、その会話では取得できない。新しい会話を追加せず、既存の共同作業枠だけで情報を渡す。','',
        '|強度|慣れた条件の確認要求|初めて条件の確認要求|初めて条件の情報移転|初めて条件の最終手順情報充足率|全員が全情報を取得した割合|',
        '|---|---:|---:|---:|---:|---:|']
    for case,label in (('task_weak','弱'),('task_medium','中'),('task_strong','強')):
        q=by['pooled',case,'confirmation_requests']; t=by['pooled',case,'information_transfers']; p=by['pooled',case,'information_coverage']; c=by['pooled',case,'information_complete']
        lines.append(f"|{label}|{q['familiar_B_mean']:.3f}|{q['novel_B_mean']:.3f}|{t['novel_B_mean']:.3f}|{p['novel_B_mean']:.1%}|{c['novel_B_mean']:.1%}|")
    lines+=['','情報の指標はA/Bで一致する。充足率は、各自が初期に持つ1情報以外の3情報について、何割を取得したかを4人で平均した値である。全員が必要情報を取得すると、それ以降の確認要求はなくなる。慣れた条件は初期から情報充足率100%だが、これは「作業を最初から完成させている」という意味ではない。実作業の完成度・成功/失敗・達成感はモデル化していない。','',
        '## 3. なぜ差がゼロになるのか','',
        '初期の対人状態が同じなら、同じ相手・話題・乱数を既存の反応式に入れると同じ反応になる。反応が同じなら、その後の対人履歴と関係状態も同じになる。これを48ターン繰り返す。情報状態は異なるが、反応式と関係判定がその情報状態を参照しないため、違いが関係へ伝わらない。','',
        '回答や関係状態をコピーして一致させたのではない。両条件を別々に実行し、既存の反応式で計算した。そのうえで、有限個の反応乱数を全列挙し、各ターンの前向き・ふつう・噛み合わない反応の確率分布まで一致することを検証した。','',
        '前回の題材実験では、必要情報が会話の話題を変え、その話題と興味の一致・過去の好反応が後続へ波及した。今回はその話題変更を使わないため、前回の減少は再現しない。心理的な新奇性や相互依存の効果を別々に同定できたわけではない。','',
        '## 4. 今後の検証に必要なこと','',
        'この制約を保つ限り、現在のモデルに新奇性の独立した関係効果を推定する能力はない。次に効果を表現する場合は、他者への依存がどの観測可能な行動をどう変えるかを、根拠と仮定とともに先に決める必要がある。感謝や達成感で望ましい結果にする加点は今回追加していない。人間の新奇性・共同作業の効果を知るには、モデル外のデータと検証が必要である。','',
        '## 5. 実装・検証記録','',
        '- 新規実験: `poc/ab_poc/task_need_matched_experiment.py`。既存の `InformationProvider(topic_link=False)` を再利用。',
        '- 既存モデル、人物、関係閾値、A/B条件の変更なし。通常UIの動作変更なし。',
        f"- 保存後監査: {audit['condition_runs']:,}条件、{audit['matched_comparisons']:,}組の慣れた/初めて比較、{audit['turn_probability_distributions_matched']:,}ターン分の確率分布一致。",
        f"- 生記録・CSV・設定・ハッシュ: `{out.relative_to(ROOT).as_posix()}`。",
        f'- 監査: `validation/{out.name}-audit.json`。',
        '- 計画: `docs/TASK_NEED_MATCHED_PLAN.md`。実行手順: `docs/TASK_NEED_MATCHED_METHOD.md`。',
        '- テスト結果: `validation/task-need-matched-tests.log`。','']
    name='TASK_NEED_MATCHED_RESULTS.md' if n==1000 else 'TASK_NEED_MATCHED_SMOKE_RESULTS.md'
    if n<1000: lines.insert(2,'> 少数seedの試運転記録。研究結果の本報告ではない。\n')
    path=ROOT/'docs'/name; path.write_text('\n'.join(lines),encoding='utf-8')
    _write_json(ROOT/'validation'/f'{out.name}-report.json',dict(report_sha256=_sha256(path),generator_sha256=_sha256(Path(__file__)),
        test_log_sha256=_sha256(ROOT/'validation/task-need-matched-tests.log'),audit=audit))
    print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); report(p.parse_args().output)
