"""Generate a Japanese report from completed, audited early-recontact outputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.bridge_experiment import _sha256

LABELS={'neutral':'中立（通常選択）','actual_promote':'実際の提供者を促進','actual_suppress':'実際の提供者を抑制','permuted_promote':'ラベル入れ替え・促進','permuted_suppress':'ラベル入れ替え・抑制'}
NAMES={'akane':'あかね','koharu':'こはる','midori':'みどり','kurumi':'くるみ'}
def pair_name(key): return '―'.join(NAMES[p] for p in key.split('-'))
def ci(r,prefix): return f"{r[prefix+'mean']:+.4f} [{r[prefix+'ci_low']:+.4f}, {r[prefix+'ci_high']:+.4f}]"
def direction(r,prefix=''):
    return '増加' if r[prefix+'simultaneous_low']>0 else '減少' if r[prefix+'simultaneous_high']<0 else '方向未確定'

def report(output,audit,report_path):
    out=Path(output); a=json.loads(Path(audit).read_text(encoding='utf-8'))
    assert a['status']=='pass' and a['manifest_sha256']==_sha256(out/'manifest.json')
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for p,h in manifest['output_sha256'].items(): assert _sha256(out/p)==h
    s=json.loads((out/'summary.json').read_text(encoding='utf-8')); contrasts=json.loads((out/'contrasts.json').read_text(encoding='utf-8'))
    by={(r['known_pair'],r['horizon'],r['case_id'],r['metric']):r for r in s}
    cases=list(LABELS); pairs=list(dict.fromkeys(r['known_pair'] for r in s if r['known_pair']!='pooled'))
    primary=[by['pooled',24,c,'bridge_count'] for c in cases]
    text=['# 初期の共同作業後に、誰へ再接触するかを変えた実験', '', '2026年9月15日。人を対象とした実験ではなく、Yoriyoiのルールモデル内での比較。', '',
          '## 要点', '',
          f"24ターン時点のBのBridgeは、中立で平均{primary[0]['B_mean']:.3f}/4本、実際の情報提供者を選びやすくすると{primary[1]['B_mean']:.3f}本、選びにくくすると{primary[2]['B_mean']:.3f}本だった。Aもほぼ同程度に変わり、A/B差を打ち消す効果は明確ではなかった。",
          '提供者のラベルだけを入れ替える対照では、中立との差の区間が0を含んだ。今回の設定では、重みを偏らせること一般より、実際の情報提供履歴に対応させることが結果に関係している。ただし、これは仮定した選択ルールのモデル内の結果であり、人が情報提供者を避けた方がよいという結論ではない。', '',
          '## 今回確かめたこと', '',
          '最初の6会話で全6ペアを1回ずつ接触させた後、情報をくれた相手を選びやすくする／選びにくくする仮定を比較した。提供者のラベルだけを入れ替える対照も設け、単に相手選択を偏らせることと、実際の提供者への対応を分けて調べた。', '',
          '## 主結果：24ターン時点', '',
          'Bridgeは、既知ペアの2人と残り2人をつなぐ4本の関係。顔見知り以上になった本数を数える。下表は同じseed内で6既知ペアを平均し、その後1000seedで平均したもの。差の区間は対応差の95% Monte Carlo区間。', '',
          '|条件|Aの本数|Bの本数|Bの率|B−中立B［95%区間］|A−中立A|B−A|B−Aの変化|',
          '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in primary:
        text.append(f"|{LABELS[r['case_id']]}|{r['A_mean']:.4f}|{r['B_mean']:.4f}|{r['B_mean']/4:.2%}|{ci(r,'difference_B_')}|{r['difference_A_mean']:+.4f}|{r['B_minus_A_mean']:+.4f}|{r['gap_change_mean']:+.4f}|")
    text += ['', 'Bの変化は「既知関係がある場で新しいつながりが増えたか」。B−Aの変化は「既知関係の有無による差がどう変わったか」。A/B両方が変わるため、後者だけで施策の良し悪しを判断しない。', '',
             '### 実際の提供者と、ラベルを入れ替えた場合の差', '',
             '|比較（24ターン、6ペア集約）|Aの本数差［95%区間］|Bの本数差［95%区間］|', '|---|---|---|']
    for d,label in [('promote','促進：実際−入れ替え'),('suppress','抑制：実際−入れ替え')]:
        rr={r['condition']:r for r in contrasts if r['known_pair']=='pooled' and r['horizon']==24 and r['direction']==d}
        text.append(f"|{label}|{ci(rr['A'],'')}|{ci(rr['B'],'')}|")
    text += ['', '集約の95%区間は探索的な平均の区間。個別ペアの主比較については、下記の36対比補正を用いる。', '',
             '### 個別ペアで方向が残るか', '',
             '主評価24ターンのBについて、4条件対中立＋実際対入れ替え2対比を6既知ペアで評価した36対比を、Bonferroni法で補正した。方向未確定は「差がないと証明した」という意味ではない。', '',
             '|既知ペア|実際促進−中立|実際抑制−中立|入替促進−中立|入替抑制−中立|促進：実際−入替|抑制：実際−入替|', '|---|---|---|---|---|---|---|']
    status=[]
    for p in pairs:
        entries=[direction(by[p,24,c,'bridge_count'],'difference_B_') for c in cases[1:]]
        entries += [direction(next(r for r in contrasts if r['known_pair']==p and r['horizon']==24 and r['condition']=='B' and r['direction']==d)) for d in ('promote','suppress')]
        status.extend(entries); text.append('|'+pair_name(p)+'|'+'|'.join(entries)+'|')
    text += ['',f"36対比中、補正区間が0を含まない増加は{status.count('増加')}、減少は{status.count('減少')}、方向未確定は{status.count('方向未確定')}。各ペアの平均・区間はsummary.csvとcontrasts.csvに保存した。", '',
             '## 早さと48ターンまでの変化', '',
             '|条件|12ターンB|24ターンB|48ターンB|24ターン時点の未成立累積ターンB|24ターンで全4本成立した割合B|', '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        vals=[by['pooled',h,c,'bridge_count']['B_mean'] for h in (12,24,48)]
        delay=by['pooled',24,c,'unformed_turns']['B_mean']; complete=by['pooled',24,c,'all_formed']['B_mean']
        text.append(f"|{LABELS[c]}|{vals[0]:.4f}|{vals[1]:.4f}|{vals[2]:.4f}|{delay:.4f}|{complete:.2%}|")
    text += ['', '未成立累積ターンは、各Bridgeが初めて成立する直前までのターン数を4本で平均した値。小さいほど早い。未成立なら評価時点で打ち切る。12・48ターンは副評価であり、別の独立試行ではない。', '',
             '## 操作は相手選択を変えたか', '',
             '以下はBの第7〜24ターンの平均。提供者とは実際に情報を受け取った相手を指し、入れ替え条件でも定義は同じ。', '',
             '|条件|情報提供者への会話割合|既知ペアへの割合|Bridge候補への割合|未成立Bridge候補への会話回数|選択分布の変化量|', '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        vals={m:by['pooled',24,c,m]['B_mean'] for m in ('source_contact_share','free_known_share','free_bridge_share','unformed_bridge_contacts','selection_total_variation')}
        text.append(f"|{LABELS[c]}|{vals['source_contact_share']:.2%}|{vals['free_known_share']:.2%}|{vals['free_bridge_share']:.2%}|{vals['unformed_bridge_contacts']:.4f}|{vals['selection_total_variation']:.4f}|")
    unchanged=by['pooled',24,'neutral','permutation_unchanged_fraction']['B_mean']
    transfers=by['pooled',24,'neutral','information_transfers']['B_mean']
    text += ['',f'課題中の情報移転は平均{transfers:.4f}件。提供者ラベルを入れ替えても受領量ベクトルが変わらない話し手は{unchanged:.2%}。この場合も除外せず集計した。', '',
             '選択分布の変化量は、同じ履歴における通常選択と倍率付き選択の全変動距離。提供者ラベルの入れ替えで倍率の集合は保たれるが、元の重みとの対応が変わるため、確率分布への作用の大きさまで完全に等しくはならない。', '',
             '## 何が分かり、何は分からないか', '',
             '実際の提供者を促進・抑制したときの中立との差、および実際の提供者と入れ替え条件の差は、24ターンのBで全6既知ペアに同じ方向で現れ、36対比補正後も0を含まなかった。一方、入れ替え条件と中立の差は全6ペアで方向未確定だった。',
             '実際の提供者を抑制するとBの形成数は増えたが、Aも増えた。A/B差の変化の95%区間は0を含むため、Island効果を明確に縮小したとは言えない。',
             '提供者への会話割合は約47.24%から、促進で50.23%、抑制で44.24%へ変わった。一方、Bridge候補全体への会話割合は各条件とも約60.7%だった。促進条件では、未成立Bridge候補への会話回数はむしろ増え、形成は遅れた。この指標は未成立状態が長く続けば増え得るため、会話回数だけで成果を判断できない。',
             'したがって、今回の減少を「Bridgeへの会話機会が全体として減ったから」とは説明できない。4本の中での接触配分・発話方向・話題と反応・その後の履歴のどの経路が効いたかは、別途切り分ける余地がある。実際の情報提供者は課題中の接触方向や順序とも結び付いており、提供者という心理的意味だけを抽出した対照ではない。', '',
             '- この結果は、早期に均等接触した後の「情報提供履歴と再接触の結び付き」という仮定に対する、既存モデルの感度を示す。感謝、新奇性、共同達成感を実測したものではない。',
             '- 前回と比べ、課題量・情報量・介入開始時点・seedが違う。前回の小さな差が天井効果だけで生じたとは結論しない。',
             '- 式は前回と同じ2 ** (theta × c / 3)だが、今回の課題は各ペア1会話なのでcは0か1。実際の倍率は約0.794〜1.260であり、前回と実効的な作用の幅は一致しない。',
             '- 同じseedを対応させているが、相手選択が分かれた後は話題や反応履歴も既存ルールに従って変化する。後半の好反応確率まで固定した実験ではない。',
             '- 区間が0を含む場合は、小さな効果の有無を含め方向が定まらない。実用的に同等とする許容差は事前に設定していない。',
             '- ①会話機会を増やす効果と②同じ会話機会での構造の効果について、モデル内の知見を追加した。③実際の滞在を延ばす要因と④その滞在延長を通じたBridge増加は、引き続き未確認。', '',
             '## 検証と成果物', '',
             f"保存後監査：{a['status']}。{a['condition_runs']:,}条件、{a['conversations']:,}会話、{a['seed_metric_rows']:,}指標行、{a['summary_rows']:,}集計行、{a['contrast_rows']:,}対比行を検算。",
             '既存分を含む69テストと120条件の小規模実行が通過。中立条件と既存モデルの一致、A内・B内それぞれで比較5条件の初期6会話の一致、各ペア1接触、課題終了時Bridge 0、情報授受なしでの全条件一致、保存データの破損検出を確認した。',
             '保存後監査は全会話の選択重み・選択乱数・話題・反応乱数・情報授受・関係状態と、seed CSV・集計JSON/CSVを再計算した。人物設定を含むソースと出力のハッシュも確認した。', '',
             f"- データ：`{out.relative_to(ROOT).as_posix() if out.is_absolute() else out.as_posix()}`",
             '- 実行方法・仮定・指標定義：`docs/EARLY_RECONTACT_METHOD.md`',
             '- 実験実装：`poc/ab_poc/early_recontact_experiment.py`',
             '- テスト：`tests/test_early_recontact.py`',
             '- 保存後監査：`validation/verify_early_recontact.py`',
             f"- 監査記録：`{Path(audit).as_posix()}`",
             f"- manifest SHA-256：`{a['manifest_sha256']}`", '',
             '通常UI・人物・関係形成閾値は変更していない。追加実験の無効化は新しい実験コマンドを使用しないだけでよい。外部公開・課金・人への実験は行っていない。', '',
             '## 次の一手', '',
             '次は計画の第2段階「同じ6接触を、序盤・中盤・終盤・分散のどこに置くか」の比較が候補。今回の再接触係数を混ぜず、同じ48会話の中で接触を配る時期だけを変える。今回実行した60,000条件には、このタイミング比較と人物設定の変更は含まれない。', '']
    Path(report_path).write_text('\n'.join(text),encoding='utf-8')
    print(report_path)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--audit',required=True); p.add_argument('--report',required=True)
    a=p.parse_args(); report(a.output,a.audit,a.report)
