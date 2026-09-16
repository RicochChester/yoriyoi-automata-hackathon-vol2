"""Report audited sensitivity results without interpreting assumptions as observations."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.bridge_experiment import _sha256, _write_json

LABELS={-1:'再接触抑制',0:'中立（通常選択）',1:'再接触促進'}
NAMES={'akane':'あかね','koharu':'こはる','midori':'みどり','kurumi':'くるみ'}

def ci(r,p,sim=False):
    suffix=('simultaneous_low','simultaneous_high') if sim else ('ci_low','ci_high')
    return f"{r[p+'_mean']:+.4f} [{r[p+'_'+suffix[0]]:+.4f}, {r[p+'_'+suffix[1]]:+.4f}]"

def report(output,destination):
    out=Path(output).resolve(); destination=Path(destination).resolve()
    ap=ROOT/'validation'/f'{out.name}-audit.json'
    audit=json.loads(ap.read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert audit['status']=='pass' and manifest['status']=='complete'
    assert audit['manifest_sha256']==_sha256(out/'manifest.json')
    test_log=(ROOT/'validation/post-task-recontact-tests.log').read_text(encoding='utf-8-sig')
    assert 'Ran 64 tests' in test_log and test_log.strip().endswith('OK')
    for key in ('summary.json','summary.csv','design.json'):
        assert _sha256(out/key)==manifest['output_sha256'][key]
    design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    by={(r['known_pair'],r['theta'],r['metric']):r for r in data}
    def row(theta,metric,key='pooled'): return by[key,theta,metric]
    start,end=design['start'],design['end']; smoke=(start,end)!=(101,1100)
    lines=['# 共同作業後の再接触：仮定を変えた追加実験','',
        '**小規模動作確認用。研究結果として解釈しない。**' if smoke else '**モデルの追加実験は完了。人を対象とした観察は未実施。**','',
        f"4人、共同作業48ターン＋自由交流24ターン、全6既知ペア、seed {start}〜{end}。3仮定×A/Bで{audit['condition_runs']:,}条件を実行し、保存結果を再検算した。",'',
        '今回変えたのは「課題で情報をくれた相手への、その後の話しかけやすさ」。再接触を抑制/中立/促進する係数は人の実測値ではなく、感度分析のための仮定である。関係値・好反応への直接ボーナスはない。','',
        '## 1. まずBの新しいつながりが増えるか','',
        'Aは事前オンライン関係なし、Bは1組あり。Bridgeはその組の両端から第三者へ伸びる4本で、最大4本。数値は6既知ペアをseed内で平均してから1000 seedで平均したもの（小規模確認では指定seed数）。','',
        '|再接触の仮定|48ターン時点 B|72ターン時点 B|後半の追加成立 B|中立との差 B [95% MC区間]|',
        '|---|---:|---:|---:|---|']
    for t in (-1,0,1):
        r=row(t,'bridge_after')
        lines.append(f"|{LABELS[t]}|{row(t,'bridge_before')['B_mean']:.4f}|{r['B_mean']:.4f}|{row(t,'new_bridge')['B_mean']:.4f}|{ci(r,'difference_B')}|")
    for t in (-1,1):
        r=row(t,'bridge_after'); lo,hi=r['difference_B_ci_low'],r['difference_B_ci_high']
        interpretation='増加側にある' if lo>0 else '減少側にある' if hi<0 else 'ゼロを含み、平均的な増減を明確に判別できない'
        lines+=['',f"Bの{LABELS[t]}では、中立との差の95% MC区間は{interpretation}。この区間はモデルの乱数による誤差についてのもので、人への効果を示さない。"]
    lines+=['','## 2. A/B両方と、Island差の変化','',
        '差の変化は `(各仮定のB−A) − (中立のB−A)`。正ならB−Aが正方向に動くが、Aの悪化だけで生じる場合もあるため、A/Bの水準と併読する。','',
        '|仮定|最終A|最終B|Aの中立との差|B−A [95% MC区間]|B−Aの変化 [95% MC区間]|',
        '|---|---:|---:|---:|---|---|']
    for t in (-1,0,1):
        r=row(t,'bridge_after')
        lines.append(f"|{LABELS[t]}|{r['A_mean']:.4f}|{r['B_mean']:.4f}|{r['difference_A_mean']:+.4f}|{ci(r,'B_minus_A')}|{ci(r,'gap_change')}|")
    lines+=['','## 3. ペアごとの違い','',
        '最終Bridge数の中立との差。角括弧は6ペア×2方向×A/Bの24比較を補正した区間。ゼロを含むペアについて方向を確定しない。','',
        '|既知ペア|抑制 A|抑制 B|促進 A|促進 B|','|---|---|---|---|---|']
    keys=sorted({r['known_pair'] for r in data}-{'pooled'})
    for k in keys:
        name='―'.join(NAMES[x] for x in k.split('-'))
        values=[ci(row(t,'bridge_after',k),'difference_'+c,True) for t,c in ((-1,'A'),(-1,'B'),(1,'A'),(1,'B'))]
        lines.append('|'+name+'|'+'|'.join(values)+'|')
    includes_zero=sum(row(t,'bridge_after',k)['difference_'+c+'_simultaneous_low']<=0<=row(t,'bridge_after',k)['difference_'+c+'_simultaneous_high'] for k in keys for t in (-1,1) for c in ('A','B'))
    lines+=['',f'24比較中{includes_zero}比較の補正区間がゼロを含む。区間がゼロを含むことを「効果が厳密にゼロ」とは解釈しない。']
    lines+=['','## 4. 何が動いたか','',
        '以下の割合は後半24会話について計算。「情報提供者」は前半で自分へ情報を渡した人。接触構造は前半48ターンで同一、後半は相手選択に応じて変わる。','',
        '|仮定|条件|情報提供者への接触|選択相手の情報量/3|重み変更直後の期待情報量差|既知ペア会話割合|Bridge会話割合|',
        '|---|---|---:|---:|---:|---:|---:|']
    for t in (-1,0,1):
        for c in ('A','B'):
            def v(m): return row(t,m)[c+'_mean']
            lines.append(f"|{LABELS[t]}|{c}|{v('source_contact_share'):.2%}|{v('source_exposure_mean'):.4f}|{v('immediate_expected_exposure_shift'):+.4f}|{v('free_known_share'):.2%}|{v('free_bridge_share'):.2%}|")
    lines+=['',f"Bの情報提供者への接触は、中立の{row(0,'source_contact_share')['B_mean']:.2%}から、促進で{row(1,'source_contact_share')['B_mean']:.2%}、抑制で{row(-1,'source_contact_share')['B_mean']:.2%}へ変化した。もともと提供者への接触割合が高い。再接触の操作が効いたことと、新しいBridgeが増えることは区別する。",'',
        '重み変更直後の期待値の差は、同じ履歴・同じ候補に倍率だけを適用した操作確認。促進なら正、抑制なら負になる設計だが、Bridge形成の増減はそれ自体からは決まらない。','',
        '|仮定|条件|最終Bridge率|4本全成立率|未成立累積ターン（4本平均）|','|---|---|---:|---:|---:|']
    for t in (-1,0,1):
        for c in ('A','B'):
            lines.append(f"|{LABELS[t]}|{c}|{row(t,'bridge_rate')[c+'_mean']:.2%}|{row(t,'all_formed')[c+'_mean']:.2%}|{row(t,'unformed_turns')[c+'_mean']:.3f}|")
    lines+=['','## 5. 解釈できる範囲','',
        f"課題終了時点でBは平均{row(0,'bridge_before')['B_mean']:.4f}/4本成立しており、追加できる余地は平均{4-row(0,'bridge_before')['B_mean']:.4f}本しかない。天井効果のある強い共同作業条件の結果であり、弱い課題・他の人数・他の時間へ一般化しない。",'',
        '情報交換→後の相手選択という経路を仮定すると、関係への直接加点がなくても選択の変化が後続の会話・関係へ伝わり得る。これはその経路が人に実在することを示さない。人が情報提供者に近づくのか、避けるのか、変わらないのかは未測定である。','',
        '新奇性と相互依存の2×2をモデル上で独立に操作した実験ではない。今回のモデルに「手順への慣れ」と「今回必要な情報」の独立した実証的な作用式はないため、ラベルだけの4条件を新奇性検証と呼ばない。','',
        '区間は同じseedに対応づけたMonte Carlo誤差。6ペアを6000独立標本とは扱わない。正規近似を使用し、小規模動作確認の区間は推論に使わない。副指標や全解析を覆う同時保証ではない。現実の人の不確実性やモデルの妥当性はこの区間に含まれない。既存seedを再利用した探索であり、未知データへの予測検証ではない。','',
        '## 6. 人の追加実験は準備まで','',
        '観察記録はまだないとの回答を受け、[4条件の予備調査案](HUMAN_RECONTACT_PILOT_PROTOCOL.md)と[ヘッダーのみの記録票](../study_materials/recontact_pilot/)を用意した。架空の観察データは作っていない。手順への慣れ×情報の相互依存を分け、接触枠と成果物形式を統一し、その後の自由交流で情報提供者への再接触を観察する案である。','',
        '実施責任者・場所・参加者・課題教材・同意/記録方法・必要人数は人間側で確定が必要。実際の観察後、AI側で記録品質・条件操作を確認し、グループ単位で集計する。滞在時間延長を調べる実験は別途必要。','',
        '## 7. 実装と検証証拠','',
        '[変更箇所・設定値・PowerShell手順](POST_TASK_RECONTACT_METHOD.md)、[実行前計画](POST_TASK_RECONTACT_PLAN.md)を参照。通常UIと人物設定・関係閾値は維持した。','',
        f"- 保存後検算: {audit['condition_runs']:,}条件、{audit['summary_rows']}集計行、全72ターンの選択・反応・関係状態、前半48ターン一致を確認。",
        '- 自動テスト: 64件通過。中立条件と通常選択の一致、情報授受なしで全係数一致、不正な重みを検出するテストを含む。',
        f"- データ: `{out.relative_to(ROOT).as_posix()}`。集計表は `summary.csv`、各ペアの `seed_metrics.csv` と圧縮会話記録を保存。",
        f"- 検算: `validation/{ap.name}`。manifest SHA256: `{audit['manifest_sha256']}`。",
        '- 主な残務: 人の観察（人間側の実施責任者）、実測に基づくモデル係数の検証（観察後にAIと共同）。','']
    destination.write_text('\n'.join(lines),encoding='utf-8')
    _write_json(ROOT/'validation'/f'{out.name}-report-provenance.json',dict(
        report=destination.relative_to(ROOT).as_posix(),report_sha256=_sha256(destination),
        generator_sha256=_sha256(Path(__file__)),audit_sha256=_sha256(ap),
        tests_sha256=_sha256(ROOT/'validation/post-task-recontact-tests.log'),manifest_sha256=audit['manifest_sha256']))
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(); report(a.output,a.report)
