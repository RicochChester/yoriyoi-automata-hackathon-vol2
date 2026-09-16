"""Small, hash-verified records from completed experiments; no local results required."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT=Path(__file__).parent/'fixtures'
MANIFEST=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))

def records(dataset,pair,start=101,end=101):
    key='-'.join(pair) if not isinstance(pair,str) else pair
    relative=f'{dataset}/{key}.jsonl.gz'
    data=(ROOT/relative).read_bytes()
    assert hashlib.sha256(data).hexdigest()==MANIFEST[relative]['sha256']
    all_records=[json.loads(line) for line in gzip.decompress(data).decode('utf-8').splitlines()]
    assert len(all_records)==MANIFEST[relative]['records']
    selected=[r for r in all_records if start<=r['result']['root_seed']<=end]
    assert selected, 'fixture range not available'
    return selected

def inputs(key,start,end):
    saved=records('information',key,start,end)
    assert len(saved)%2==0
    for i in range(0,len(saved),2):
        familiar,novel=saved[i:i+2]
        assert familiar['profile']=='familiar' and novel['profile']=='novel'
        assert familiar['case_id']==novel['case_id']
        yield familiar['case_id'],familiar['result'],novel['result']
