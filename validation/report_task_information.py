"""Generate readable tables from the finished information comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from poc.ab_poc.bridge_experiment import _sha256, _write_json

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def ci(r,k): return f"{r[k+'_mean']:+.3f} [{r[k+'_ci_low']:+.3f}, {r[k+'_ci_high']:+.3f}]"

def main():
    out=ROOT/'results/task-information-101-1100'
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    audit_path=ROOT/'validation/task-information-101-1100-audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert m['condition_runs']==72000 and audit['status']=='pass'
    data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    levels={'task_weak':'弱（6接触）','task_medium':'中（12接触）','task_strong':'強（24接触）'}
    names={'akane':'あかね','koharu':'こはる','midori':'みどり','kurumi':'くるみ'}
    pooled={(r['case_id'],r['metric']):r for r in data if r['known_pair']=='pooled'}
    sections=['# 分散情報を必要とする初めての共同作業：結果',
        '4人・48ターン・全6既知ペア・seed101〜1100・弱/中/強。通常/初めて×A/Bで72,000実行を完了。通常と初めての間で48ターン全体の話し手・相手・順序を完全に固定した。',
        '通常は課題の4情報を既知、初めては各人1情報だけを持ち習熟度0。情報を必要とする質問で話題が変わり、既存の反応・関係ルールで判定した。情報・習熟度による直接加点はない。新奇性の心理効果や感謝、共同達成感の実験ではない。',
        '## 最初にBの新規形成を比較する',
        'Bridge成立数は顔見知り以上の0〜4本。率は件数÷4。区間は同seedでの初めて−通常の正規近似95% Monte Carlo区間。全6ペアをseed内で等重み平均した1000seedの集計。',
        table(['強度','通常A','初めてA','通常B','初めてB','初めてBの成立率','Bの差 [95%区間]'],[[levels[c],f"{pooled[c,'bridge_count']['familiar_A_mean']:.3f}",f"{pooled[c,'bridge_count']['novel_A_mean']:.3f}",f"{pooled[c,'bridge_count']['familiar_B_mean']:.3f}",f"{pooled[c,'bridge_count']['novel_B_mean']:.3f}",f"{pooled[c,'bridge_rate']['novel_B_mean']:.1%}",ci(pooled[c,'bridge_count'],'novel_minus_familiar_B')] for c in levels]),
        '## A/B差の変化',
        '差の差＝初めての(B−A)−通常の(B−A)。正でも、Aの悪化によって差が縮む場合があるため、上のBの絶対比較を先に読む。',
        table(['強度','Aの初めて−通常 [95%区間]','通常B−A','初めてB−A','差の差 [95%区間]'],[[levels[c],ci(pooled[c,'bridge_count'],'novel_minus_familiar_A'),f"{pooled[c,'bridge_count']['familiar_B_minus_A_mean']:+.3f}",f"{pooled[c,'bridge_count']['novel_B_minus_A_mean']:+.3f}",ci(pooled[c,'bridge_count'],'island_change')] for c in levels]),
        '## 既知ペア別のBの差',
        '＊は6ペア×3強度の18比較を補正した正規近似区間も0を含まない。印なしは同等性の証拠ではない。',
        table(['既知ペア','強度','Bの初めて−通常 [95%区間]'],[['―'.join(names[p] for p in r['known_pair'].split('-')),levels[r['case_id']],ci(r,'novel_minus_familiar_B')+('＊' if r['novel_minus_familiar_B_simultaneous_low']>0 or r['novel_minus_familiar_B_simultaneous_high']<0 else '')] for r in data if r['known_pair']!='pooled' and r['metric']=='bridge_count']),
        '## 形成の遅れと全4本成立',
        '遅れは未成立だった累積ターンの4本平均。最後まで未成立は48を含む。小さいほど早い。',
        table(['強度','B遅れ 通常→初めて','B遅れ差 [95%区間]','B全4本成立率 通常→初めて'],[[levels[c],f"{pooled[c,'unformed_turns']['familiar_B_mean']:.2f} → {pooled[c,'unformed_turns']['novel_B_mean']:.2f}",ci(pooled[c,'unformed_turns'],'novel_minus_familiar_B'),f"{pooled[c,'all_formed']['familiar_B_mean']:.1%} → {pooled[c,'all_formed']['novel_B_mean']:.1%}"] for c in levels]),
        'Aの副指標、既知ペア・Bridgeへの会話割合、すべての対応差と区間もsummary.csvに保存。会話割合は通常/初めて間で完全一致するので差は0。A/B間まで同一に固定した実験ではない。',
        '## 情報交換は実際に起きたか',
        table(['強度','初めてBの質問数','取得情報数','最終情報進行度','全員の情報が揃った割合','全48ターンの話題変更数'],[[levels[c],f"{pooled[c,'questions']['novel_B_mean']:.3f}",f"{pooled[c,'information_transfers']['novel_B_mean']:.3f}",f"{pooled[c,'task_progress']['novel_B_mean']:.1%}",f"{pooled[c,'task_complete']['novel_B_mean']:.1%}",f"{pooled[c,'changed_topics']['novel_B_mean']:.3f}"] for c in levels]),
        '情報進行度は他者由来の3情報×4人＝12取得の充足率。全員に必要情報が揃ったことを課題完了と呼ぶ。通常は最初から100%であり、実際の作業成果物の完成時間を測ったわけではない。情報所有と作業接触はA/B共通で、情報提供は社会的反応から独立なので、情報進行もA/B共通。',
        '## 何を仮定したモデルか',
        '課題情報の話題は地域・展示・ものづくり・食。所有者をseedで割り当て、相手の情報目録を参照できると仮定した。必要な情報が相手にあれば質問し、返答で1つ取得する。話題への反応は従来の人物の興味・慎重さ・既知関係・過去の反応で決まる。',
        '通常/初めてで相手選択を固定したため、「必要性によって接触数や相手が変わる」経路は遮断している。情報共有の価値そのものを好意に換算する経路もない。情報を取得しながら社会的反応が改善しない、または悪化することも起こり得る。',
        '題材の選択、情報提供方式、通常を全情報既知とする仮定に依存する。差が出ても人間の初めてのワークショップ一般には直接適用できない。新奇性と相互依存を別々に操作した4条件実験ではない。',
        '## 検証・再実行',
        f"全{audit['condition_runs']:,}実行・{audit['pair_rows']:,}ペア行を保存後に監査。全接触固定、通常の元記録との完全一致、情報の移動、全関係状態、CSVと集計、入力出力ハッシュを確認した。",
        '情報を動かしつつ話題への接続だけを切ったテストでは、関係状態が元記録と一致した。つまり情報を持つだけで関係が上がる処理はない。既存テストを含む54テストが成功。',
        '結果：results/task-information-101-1100。設定と元記録ハッシュはdesign.json、完了証拠はmanifest.json。全会話・全情報状態は各既知ペア下のtraces.jsonl.gz、ペア表はpairs.csv、seed集計はseed_metrics.csv。全主副指標と区間はsummary.csv/json。監査はvalidation/task-information-101-1100-audit.json。',
        '仕組みと実行コマンドはTASK_INFORMATION_METHOD.md、事前仕様はTASK_INFORMATION_PLAN.md。ローカル実験は完了。人間への外的妥当性、題材を変えた頑健性、感謝・共同達成感の経路は未検証。公開・課金・UI改修は行っていない。']
    report=ROOT/'docs/TASK_INFORMATION_RESULTS.md'; report.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    _write_json(ROOT/'validation/task-information-report.json',dict(status='pass',report_sha256=_sha256(report),manifest_sha256=_sha256(out/'manifest.json'),audit_sha256=_sha256(audit_path),script_sha256=_sha256(Path(__file__))))
    for c in levels:
        r=pooled[c,'bridge_count']; print(c,ci(r,'novel_minus_familiar_B'),r['familiar_B_mean'],r['novel_B_mean'])

if __name__=='__main__': main()
