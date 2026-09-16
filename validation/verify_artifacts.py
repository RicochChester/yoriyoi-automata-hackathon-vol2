"""Audit the final 100-seed artifacts against raw runs and the original archive."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poc.ab_poc.bridge_experiment import aggregate, analyze_ab


def main():
    output = ROOT / 'results/bridge-001-100'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    baseline = json.loads((ROOT / 'validation/baseline_run_hashes.json').read_text())
    raw = [json.loads(line) for line in (output / 'raw_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['root_seed'] for r in raw] == list(range(1, 101))
    pairs, summaries = [], []
    for run in raw:
        blob = json.dumps(run, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        assert hashlib.sha256(blob).hexdigest() == baseline[str(run['root_seed'])]
        rows, summary = analyze_ab(run)
        pairs.extend(rows)
        summaries.append(summary)
    assert len(pairs) == 1200
    assert pairs == json.loads((output / 'pairs.json').read_text(encoding='utf-8'))
    for name, records in (('pairs.csv', pairs), ('seed_summary.csv', summaries)):
        with (output / name).open(encoding='utf-8-sig', newline='') as handle:
            assert list(csv.DictReader(handle)) == [{k: str(v) for k, v in row.items()} for row in records]
    with (output / 'bridge_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        bridge = list(csv.DictReader(handle))
    assert len(bridge) == 100
    assert bridge == [{k: str(summary[k]) for k in bridge[0]} for summary in summaries]
    assert aggregate(summaries) == json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    original_count = 0
    with zipfile.ZipFile(ROOT.parent / 'upstream.zip') as archive:
        for member in archive.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            if not member.is_dir() and relative.parts[0] == 'poc':
                assert (ROOT / relative).read_bytes() == archive.read(member), str(relative)
                original_count += 1
    report = {
        'status': 'pass', 'seed_count': 100, 'condition_run_count': 200,
        'pair_row_count': 1200, 'original_poc_files_byte_identical': original_count,
        'all_raw_run_hashes_match_pre_edit_baseline': True,
        'CSV_JSON_raw_run_aggregation_consistent': True,
        'source_and_output_checksums_valid': True,
    }
    (ROOT / 'validation/artifact-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
