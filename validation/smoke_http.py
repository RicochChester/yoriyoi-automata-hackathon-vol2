"""Start the original HTTP service on a free loopback port and stop it safely."""
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poc.server import POCRequestHandler, ThreadingHTTPServer
from poc.ab_poc.engine import run_ab


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), POCRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    opener = build_opener(ProxyHandler({}))
    results = {}
    try:
        for path in ('/api/health', '/ab-lab/', '/ab-lab/ab-lab.js', '/api/ab-poc/compare?seed=1&participant_provider=rule'):
            with opener.open(origin + path, timeout=10) as response:
                data = response.read()
                assert response.status == 200
                results[path] = {'status': response.status, 'bytes': len(data)}
                if path.startswith('/api/ab-poc/compare'):
                    assert json.loads(data) == run_ab(seed=1), 'HTTP output differs from Python runner'
                if path == '/api/health':
                    assert json.loads(data)['status'] == 'ok'
        report = {'status': 'pass', 'loopback_only': True, 'HTTP_equals_run_ab_seed_1': True, 'routes': results}
        (ROOT / 'validation/http-smoke.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == '__main__':
    main()
