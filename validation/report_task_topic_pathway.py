"""Generate the Japanese report from audited saved metrics."""
import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.event_format_experiment import estimate_ci
from poc.ab_poc.bridge_experiment import _write_json, _write_csv, _sha256
from poc.ab_poc.task_topic_pathway_experiment import PAIRS

def report(output):
    out=Path(output).resolve()
    audit=json.loads((ROOT/'validation'/f'{out.name}-audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='pass' and audit['manifest_sha256']==_sha256(out/'manifest.json')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    by={(r['known_pair'],r['case_id'],r['profile'],r['metric']):r for r in summary}
    rows=[]
    for pair in PAIRS:
        with (out/'-'.join(pair)/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f: rows.extend(csv.DictReader(f))
    data={(r['known_pair'],int(r['seed']),r['case_id'],r['profile']):r for r in rows}
    seeds=sorted({int(r['seed']) for r in rows}); keys=['-'.join(p) for p in PAIRS]
    weights={'all_history_effect':{'original':1,'freeze_TR':-1},
             'topic_history_removal':{'original':1,'freeze_T':-1},
             'response_history_removal':{'original':1,'freeze_R':-1},
             'history_interaction':{'original':1,'freeze_T':-1,'freeze_R':-1,'freeze_TR':1},
             'rotated_minus_original':{'rotated':1,'original':-1},
             'outside_minus_original':{'outside':1,'original':-1}}
    contrasts=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for name,terms in weights.items():
            for c in ('A','B'):
                values=[mean(sum(w*float(data[k,s,'task_strong',p]['bridge_count_'+c]) for p,w in terms.items()) for k in selected) for s in seeds]
                contrasts.append(dict(known_pair=key,contrast=name,condition=c,**estimate_ci(values,72)))
    dest=ROOT/'validation'/f'{out.name}-contrasts.csv'; _write_csv(dest,contrasts)
    def ci(r,prefix):
        if r[prefix+'ci_low'] is None: return f"{r[prefix+'mean']:+.3f} [区間未算出]"
        return f"{r[prefix+'mean']:+.3f} [{r[prefix+'ci_low']:+.3f}, {r[prefix+'ci_high']:+.3f}]"
    labels={'original':'元の題材','rotated':'題材の割り当て変更','outside':'興味登録外の題材','freeze_T':'話題履歴を固定','freeze_R':'反応履歴を固定','freeze_TR':'両方の履歴を固定'}
    strength={'task_weak':'弱','task_medium':'中','task_strong':'強'}
    lines=['# 課題の題材と、その後の会話への波及を切り分ける', '',
        f'48会話・4人・全6既知ペア・seed {seeds[0]}–{seeds[-1]}。Aは既知関係なし、Bは1組のみ既知。通常共同作業と同一の全48接触を再生した。結果はこのルールベースモデル内のものである。', '',
        '## 1. 題材を変えた結果', '',
        'Bridge成立数は4本が上限。括弧内はseed単位の対応差の正規近似95% Monte Carlo区間。6ペアを独立な試行とせず、同一seed内で平均してから区間を計算した。', '',
        '|強度|題材|A平均|B平均|Bの通常共同作業との差|B−A|',
        '|---|---|---:|---:|---|---:|']
    for case in strength:
        base=by['pooled',case,'original','bridge_count']
        a=base['A_mean']-base['difference_A_mean']; b=base['B_mean']-base['difference_B_mean']
        lines.append(f"|{strength[case]}|通常共同作業（比較元）|{a:.3f}|{b:.3f}|0.000|{b-a:+.3f}|")
        for profile in ('original','rotated','outside'):
            r=by['pooled',case,profile,'bridge_count']
            lines.append(f"|{strength[case]}|{labels[profile]}|{r['A_mean']:.3f}|{r['B_mean']:.3f}|{ci(r,'difference_B_')}|{r['B_minus_A_mean']:+.3f}|")
    lines+=['','original＝地域・展示・ものづくり・食。rotated＝情報IDへの割り当てを2位置ずらしたもの。outside＝天文・古典・地質・法律。outsideの題材は全員の興味登録外であり、不利な境界条件を意図した診断対照である。現実の題材の人気や難しさを表してはいない。', '',
        '元の題材以外でも同じ方向かは下表で確認できる。負の平均は減少方向、補正区間の上端も0未満なら今回の比較群で減少を支持する。題材の網羅的な一般性までは示さない。', '',
        '|題材|強度|B差が負のペア数/6|144比較補正後も減少を支持/6|', '|---|---|---:|---:|']
    for profile in ('original','rotated','outside'):
        for case in strength:
            rs=[by[k,case,profile,'bridge_count'] for k in keys]
            lines.append(f"|{labels[profile]}|{strength[case]}|{sum(r['difference_B_mean']<0 for r in rs)}/6|{sum(r['difference_B_simultaneous_high'] is not None and r['difference_B_simultaneous_high']<0 for r in rs)}/6|")
    lines+=['','## 2. 質問した場面と、後続への波及（強条件）','','|条件|BのBridge平均|通常共同作業との差|','|---|---:|---|']
    for profile in ('original','freeze_T','freeze_R','freeze_TR'):
        r=by['pooled','task_strong',profile,'bridge_count']
        lines.append(f"|{labels[profile]}|{r['B_mean']:.3f}|{ci(r,'difference_B_')}|")
    lines+=['','両方固定では、質問以外の話題と反応は通常共同作業と一致する。それでも関係評価には実際の質問への反応が残る。「両方固定−通常共同作業」は後続の履歴反応を通常に保った制御比較、「元の題材−両方固定」は履歴の波及を許したことによる差である。後者には話題履歴と反応履歴の相互作用が含まれる。', '',
        '|強条件の対応比較|A差 [95%区間]|B差 [95%区間]|','|---|---|---|']
    names={'all_history_effect':'元の題材 − 両方固定','topic_history_removal':'元の題材 − 話題履歴固定',
           'response_history_removal':'元の題材 − 反応履歴固定','history_interaction':'履歴2経路の相互作用',
           'rotated_minus_original':'割り当て変更 − 元の題材','outside_minus_original':'興味登録外 − 元の題材'}
    for name in names:
        selected={r['condition']:r for r in contrasts if r['known_pair']=='pooled' and r['contrast']==name}
        lines.append(f"|{names[name]}|{ci(selected['A'],'')}|{ci(selected['B'],'')}|")
    lines+=['','これらは履歴を外部記録に差し替えるモデル内部の制御比較であり、現実の介入効果や唯一の媒介割合ではない。話題固定と反応固定の差を単純に足して全作用と呼ばない。', '',
        '## 3. その場の反応を同じ状況で比較する', '',
        '質問時に、その時点の履歴と乱数を変えず「課題の話題」と「自然に選ばれたはずの話題」を二重評価した。下表はBの1試行あたりの平均回数。即時差は好反応の増加数−減少数。質問場面/それ以外の差は通常共同作業との好反応数差であり、即時差とは比較対象が異なる。', '',
        '|強条件|質問数|即時の好反応差|質問場面の好反応差|質問以外の好反応差|', '|---|---:|---:|---:|---:|']
    for profile in labels:
        values=[by['pooled','task_strong',profile,m]['B_mean'] for m in ('questions','immediate_positive_difference','question_positive_difference','other_positive_difference')]
        lines.append('|'+labels[profile]+'|'+'|'.join(f'{v:.3f}' for v in values)+'|')
    lines+=['','### 最終成立数と、成立するまでの遅れは別の指標','','未成立ターンは、4本のBridgeがまだ成立していなかった累積ターン数の平均。小さいほど早い。下表はB・強条件。','','|条件|未成立ターン平均|通常共同作業との差 [95%区間]|全4本成立率|','|---|---:|---|---:|']
    for profile in ('original','rotated','outside','freeze_T','freeze_R','freeze_TR'):
        r=by['pooled','task_strong',profile,'unformed_turns']; a=by['pooled','task_strong',profile,'all_formed']
        lines.append(f"|{labels[profile]}|{r['B_mean']:.3f}|{ci(r,'difference_B_')}|{a['B_mean']:.1%}|")
    lines+=['','最終的なBridge数の減少がなくなっても、形成が早まるとは限らない。好反応総数・成立したペア数・形成の遅れは区別する。既知ペア/Bridge候補への会話割合は、全48接触を固定したため比較条件間で一致する。','']
    lines+=['','## 4. 何を結論にできるか','','興味との一致、過去の好反応、慎重さなどの既存ルールを通して、課題の話題が反応とその後の関係形成を変える経路を調べた。情報交換と接触回数は全条件で一致する。感謝、共同達成感、新奇性そのものの心理効果は追加していない。', '',
        '質問文の丁寧さや「質問されたこと」自体を評価する仕組みはない。今回の結果を「人は質問されると嫌がる」と読み替えることはできない。必要情報への質問が題材を変え、その題材と履歴が既存の反応式に入力される実験である。', '',
        '興味登録外条件では、どの単語を使っても既存の興味一致は起きない。したがって「広い題材の意味を理解した結果」ではなく、文字列一致と履歴ルールの挙動である。前回と同じseedを使った機構診断であり、新しいseed・人物・反応モデルでの追試は残る。滞在時間や人間一般への効果は未検証。', '',
        '## 5. 保存先・再現・検証','','- 計画: `docs/TASK_TOPIC_PATHWAY_PLAN.md`',
        '- 手順・変更内容: `docs/TASK_TOPIC_PATHWAY_METHOD.md`',
        f'- 生記録・seed別/ペア別集計: `{out.relative_to(ROOT).as_posix()}`',
        f'- 保存後監査: `validation/{out.name}-audit.json`',
        f'- 履歴経路と題材の対応比較: `validation/{out.name}-contrasts.csv`',
        f"- 読み戻し監査: {audit['condition_runs']:,}条件、前回original {audit['original_reproduced']:,}条件完全再現。全接触、情報、反応式、全ターン関係、集計表を照合。",
        '- 単体・回帰テスト: `validation/task-topic-pathway-tests.log`。公開や外部書き込みは行っていない。','']
    report_path=ROOT/'docs'/('TASK_TOPIC_PATHWAY_RESULTS.md' if len(seeds)==1000 else 'TASK_TOPIC_PATHWAY_SMOKE_RESULTS.md')
    if len(seeds)<1000: lines.insert(2,'> これは少数seedの動作確認用出力です。研究上の結論には使いません。\n')
    else:
        original=by['pooled','task_strong','original','bridge_count']
        fixed=by['pooled','task_strong','freeze_TR','bridge_count']
        delay=by['pooled','task_strong','freeze_TR','unformed_turns']
        all_negative=all(by[k,c,p,'bridge_count']['difference_B_simultaneous_high']<0 for k in keys for c in strength for p in ('original','rotated','outside'))
        intro=['## 今回分かったこと','']
        if all_negative:
            intro+=['今回の3題材条件×3強度×全6ペアでは、BのBridge成立数はすべて通常共同作業より減少した。元の題材だけに限った現象ではない。ただし、題材を変更すると減少幅も変わり、特に興味登録外では大きく減少した。','']
        intro+=[f"元の題材・強条件では、BのBridge平均は通常共同作業の{original['B_mean']-original['difference_B_mean']:.3f}本から{original['B_mean']:.3f}本へ変わった。ところが、後続の話題・反応履歴を通常共同作業に固定すると{fixed['B_mean']:.3f}本となった。質問場面だけでなく、その後に過去の会話を参照する経路が、最終成立数の減少に大きく関わっている。",'',
            f"同じ元の題材・強条件で、既知関係のないAも{original['A_mean']-original['difference_A_mean']:.3f}本から{original['A_mean']:.3f}本へ減った。オンライン既知関係だけに特有の減少とは捉えない。",'',
            f"一方、両方の履歴を固定しても、未成立ターンは通常より平均{delay['difference_B_mean']:.3f}ターン多かった。最終的な減少が消えることと、形成の遅れが消えることは同じではない。",'',
            'これはこのモデル内の制御比較である。「質問すること自体が悪い」「初めての共同作業は人間関係を悪くする」とは結論できない。','']
        lines[4:4]=intro
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    _write_json(ROOT/'validation'/f'{out.name}-report.json',dict(report_sha256=_sha256(report_path),contrast_sha256=_sha256(dest),
        generator_sha256=_sha256(Path(__file__)),test_log_sha256=_sha256(ROOT/'validation/task-topic-pathway-tests.log'),audit=audit))
    print(report_path)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); report(p.parse_args().output)
