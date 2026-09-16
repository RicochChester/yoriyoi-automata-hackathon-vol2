"""Paired secondary contrasts and Japanese tables from completed event48 outputs."""
import csv
import json
from pathlib import Path
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.event_format_experiment import MEASURES, PAIRS, estimate_ci
from poc.ab_poc.bridge_experiment import _sha256, _write_csv, _write_json
from poc.ab_poc.event_format_48_experiment import DESIGN

LABELS={'free':'自由交流',**{f'{kind}_{s}':name+'・'+level for kind,name in [('shuffle','シャッフル'),('task','共同作業'),('host','紹介'),('topic','共通話題')] for s,level in [('weak','弱'),('medium','中'),('strong','強')]}}
NAMES={'akane':'あかね','koharu':'こはる','midori':'みどり','kurumi':'くるみ'}
def pair_name(p): return '―'.join(NAMES[x] for x in p.split('-'))
def interval(r,k): return f"{r[k+'_mean']:+.3f} [{r[k+'_ci_low']:+.3f}, {r[k+'_ci_high']:+.3f}]"
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])

def main():
    source=ROOT/'results/event-formats-48-101-1100'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['condition_runs']==156000
    out=ROOT/'results/event-formats-48-analysis-101-1100'
    out.mkdir(exist_ok=False)
    by={}
    for p in PAIRS:
        key='-'.join(p)
        with (source/key/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                numeric={k:float(v) for k,v in row.items() if k not in ('case_id','known_pair')}
                by[key,int(row['seed']),row['case_id']]=numeric
    assert len(by)==78000
    seeds=range(101,1101); cases=DESIGN.cases; keys=['-'.join(p) for p in PAIRS]
    for seed in seeds:
        for case in cases:
            six=[by[p,seed,case] for p in keys]
            by['pooled',seed,case]={k:mean(r[k] for r in six) for k in six[0]}
    contrasts=[]; task=[]
    for key in keys+['pooled']:
        for case in cases:
            for m in MEASURES:
                for contrast in ('B_vs_free','A_vs_free','B_minus_A','island_change'):
                    values=[]
                    for s in seeds:
                        r=by[key,s,case]; b=by[key,s,'free']
                        if contrast=='B_vs_free': value=r[m+'_B']-b[m+'_B']
                        elif contrast=='A_vs_free': value=r[m+'_A']-b[m+'_A']
                        elif contrast=='B_minus_A': value=r[m+'_B']-r[m+'_A']
                        else: value=r[m+'_B']-r[m+'_A']-b[m+'_B']+b[m+'_A']
                        values.append(value)
                    contrasts.append(dict(known_pair=key,case_id=case,metric=m,contrast=contrast,**estimate_ci(values,78 if contrast=='B_minus_A' else 72)))
        for strength in DESIGN.strengths:
            for c in ('A','B'):
                for metric in ('bridge_count','unformed_turns'):
                    values=[by[key,s,'task_'+strength][metric+'_'+c]-by[key,s,'shuffle_'+strength][metric+'_'+c] for s in seeds]
                    task.append(dict(known_pair=key,strength=strength,condition=c,metric=metric,**estimate_ci(values,18)))
    for name,rows in [('secondary_contrasts',contrasts),('task_vs_shuffle',task)]:
        _write_csv(out/(name+'.csv'),rows); _write_json(out/(name+'.json'),rows)
    pooled=json.loads((source/'pooled.json').read_text(encoding='utf-8'))
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    # Independent checks against primary saved contrasts, and rate/count scaling.
    for r in pooled:
        for contrast in ('B_vs_free','A_vs_free','B_minus_A','island_change'):
            x=next(x for x in contrasts if x['known_pair']=='pooled' and x['case_id']==r['case_id'] and x['metric']=='bridge_count' and x['contrast']==contrast)
            assert abs(x['mean']-r[contrast+'_mean'])<1e-12
            rate=next(y for y in contrasts if y['known_pair']=='pooled' and y['case_id']==r['case_id'] and y['metric']=='bridge_rate' and y['contrast']==contrast)
            assert abs(rate['mean']-x['mean']/4)<1e-12
        for c in ('A','B'):
            if r['case_id'].startswith('task_'):
                assert abs(sum(r[f'task_formed_{k}_{c}'] for k in ('before','during','between','after'))-r[f'bridge_count_{c}_mean'])<1e-12
    sections=['# 48ターン・イベント形式比較の結果',
        '4人、全6既知ペア、seed 101〜1100、13条件、A/Bで156,000実行。数値は6ペアを等重みで集約した1000 seedの平均。Bridgeは顔見知り以上の4本を数える。これは固定48会話のモデル内比較であり、人間や滞在時間延長の効果ではない。',
        '## まず、既知関係があるBで新しいつながりが増えたか',
        '件数は0〜4、成立率は件数÷4。角括弧は同seed差に基づく通常の95% Monte Carlo区間。Bの改善とAの改善を並べて示す。',
        table(['条件','A件数','B件数','B成立率','B−自由B [95%区間]','A−自由A'],[[LABELS[r['case_id']],f"{r['bridge_count_A_mean']:.3f}",f"{r['bridge_count_B_mean']:.3f}",f"{r['bridge_rate_B_mean']:.1%}",interval(r,'B_vs_free'),f"{r['A_vs_free_mean']:+.3f}"] for r in pooled]),
        '## 次に、既知関係の差を打ち消したか',
        'B−Aが負なら既知関係がある方が少ない。差の差が正なら自由交流よりB−Aが上向いたことを示す。ただしAだけが悪化しても正になるため、上のBの絶対比較と合わせて読む。',
        table(['条件','B−A [95%区間]','差の差 [95%区間]'],[[LABELS[r['case_id']],interval(r,'B_minus_A'),interval(r,'island_change')] for r in pooled]),
        '## 誰が既知ペアかによる違い',
        'まずB−自由Bの平均件数差。末尾の＊は、その比較系列の全72比較に対するBonferroni正規近似区間が0を含まない。印がないことは効果ゼロの証明ではない。',
        table(['条件']+[pair_name(k) for k in keys],[[LABELS[case]]+[next(f"{r['B_vs_free_mean']:+.3f}"+('＊' if r['B_vs_free_simultaneous_low']>0 or r['B_vs_free_simultaneous_high']<0 else '') for r in summary if r['known_pair']==key and r['case_id']==case) for key in keys] for case in cases if case!='free']),
        '以下は各形式のB−A。＊は全78比較の補正区間が0を含まない。',
        table(['条件']+[pair_name(k) for k in keys],[[LABELS[case]]+[next(f"{r['B_minus_A_mean']:+.3f}"+('＊' if r['B_minus_A_simultaneous_low']>0 or r['B_minus_A_simultaneous_high']<0 else '') for r in summary if r['known_pair']==key and r['case_id']==case) for key in keys] for case in cases]),
        '## 形成までの遅れ・会話配分・全4本成立',
        '遅れは各ターン終了時に未成立だった累積ターンの4本平均。最後まで未成立のBridgeは48を含めるので、成立例だけを選んだ待ち時間ではない。小さいほど早い。',
        table(['条件','遅れ A / B','既知ペア会話率 A / B','Bridge会話率 A / B','全4本成立率 A / B'],[[LABELS[r['case_id']],f"{r['unformed_turns_A_mean']:.2f} / {r['unformed_turns_B_mean']:.2f}",f"{r['known_share_A_mean']:.1%} / {r['known_share_B_mean']:.1%}",f"{r['bridge_share_A_mean']:.1%} / {r['bridge_share_B_mean']:.1%}",f"{r['all_formed_A_mean']:.1%} / {r['all_formed_B_mean']:.1%}"] for r in pooled]),
        '副指標のA/Bそれぞれの95%区間はpooled.csv、同seedでの差の区間は分析フォルダのsecondary_contrasts.csvに保存した。Aの「既知ペア」はBで既知になる同じ1ペアを指し、Aで初期関係を与えた意味ではない。',
        '## 共同作業とシャッフルの違い',
        '両形式は同じ順序・方向の6ペアを同じ回数使う。共同作業だけ、承認された6ターン連続ブロックに配置し、情報伝達・確認・引き継ぎの役割を記録する。人物の反応や関係閾値への課題加点はない。実際の成果物や工程成功はモデル化しない。',
        '次の補助的な対比は「共同作業の連続配置−シャッフルの分散配置」。心理的な共同作業効果ではなく、配置が変えた履歴・話し手機会を含む差である。同じ配置に揃えるテストでは関係状態が一致した。',
        table(['強度','条件','指標','共同作業−シャッフル [95%区間]'],[[{'weak':'弱','medium':'中','strong':'強'}[r['strength']],r['condition'],{'bridge_count':'成立数','unformed_turns':'遅れ'}[r['metric']],f"{r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"] for r in task if r['known_pair']=='pooled']),
        'BのBridge初成立時期（平均件数）：',
        table(['共同作業','開始前','作業中','ブロック間','最終作業後','終了後の観察ターン'],[[LABELS[r['case_id']]]+[f"{r[f'task_formed_{k}_B']:.3f}" for k in ('before','during','between','after')]+[48-max(DESIGN.positions(r['case_id']))] for r in pooled if r['case_id'].startswith('task_')]),
        '各行の4区分はB成立数に合計一致する。最終作業後の観察期間は21・7・2ターンと異なるため、この区分の件数だけで強度を比較しない。時点の分類であり、作業による後続効果の因果推定ではない。',
        '## 施策が実際にどう使われたか',
        table(['条件','B強制時の接触2回未満率','B紹介実現率','B介入中の初回Bridge会話','B介入中のBridge成立'],[[LABELS[r['case_id']],f"{r['forced_undercontacted_fraction_B']:.1%}" if r['case_id'].startswith(('shuffle_','task_')) else '対象外',f"{r['introduction_realization_B']:.1%}" if r['case_id'].startswith('host_') else '対象外',f"{r['intervention_new_bridge_contacts_B']:.3f}",f"{r['intervention_bridge_formations_B']:.3f}"] for r in pooled if r['case_id']!='free']),
        '紹介bonusは全強度で1.0。話題提示は既存の共有興味判定を一時的に真にするが、相手の本来の興味やオンライン記憶の話題優先は変えない。したがって、提示した共通話題が本来の興味と一致するとは限らない。各人の実際の話し手回数も保存している。',
        '## 読み取りの限界と保存証拠',
        '95%区間はseed変動についての正規近似区間。モデルの正しさ・人物設定の代表性・人間一般への推論を保証しない。多重比較補正は各対比系列ごとで、全副指標を一括した検定ではない。0を含む区間から同等性を断定しない。48ターン版は24ターン版と介入配置も違うため、過去版との差を時間延長だけの効果と呼ばない。',
        '本体データ：results/event-formats-48-101-1100。設定・ソースハッシュはdesign.jsonとmanifest.json。各既知ペア下に全会話/状態/介入ログ、全ペアCSV、seed集計CSV。今回の補助対比：results/event-formats-48-analysis-101-1100。操作手順：EVENT_FORMAT_48_METHOD.md。全件監査：validation/event-format-48-audit.json。']
    (ROOT/'docs/EVENT_FORMAT_48_RESULTS.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    evidence=dict(status='pass',source_manifest_sha256=_sha256(source/'manifest.json'),script_sha256=_sha256(Path(__file__)),seed_rows=78000,
        primary_contrasts_reproduced=True,rate_count_scaling_verified=True,task_timing_partition_verified=True,
        output_sha256={p.name:_sha256(p) for p in out.iterdir()},report_sha256=_sha256(ROOT/'docs/EVENT_FORMAT_48_RESULTS.md'))
    _write_json(ROOT/'validation/event-48-analysis-audit.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False))

if __name__=='__main__': main()
