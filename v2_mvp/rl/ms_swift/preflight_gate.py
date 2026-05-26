import json
import re
import hashlib
from pathlib import Path

TRAIN = Path('v2_mvp/rl/data/2n_rl_data_clean_v1.jsonl')
EVAL = Path('v2_mvp/rl/data/eval_set.jsonl')
DATASET_INFO = Path('v2_mvp/rl/ms_swift/dataset_info.json')
CONFIG = Path('v2_mvp/rl/ms_swift/grpo_config.yaml')
REWARD = Path('v2_mvp/rl/reward_manager.py')


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


def has_law(text: str) -> bool:
    t = str(text or '')
    return ('劳动合同法' in t) and bool(re.search(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*条', t))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    required = [TRAIN, EVAL, DATASET_INFO, CONFIG, REWARD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f'[FAIL] missing files: {missing}')

    train = load_jsonl(TRAIN)
    evals = load_jsonl(EVAL)

    if len(train) < 80:
        raise SystemExit(f'[FAIL] train rows too small: {len(train)}')

    law_cov = sum(1 for r in train if has_law(r.get('reasoning'))) / len(train)
    if law_cov < 0.8:
        raise SystemExit(f'[FAIL] law coverage too low: {law_cov:.4f}')

    outcomes = {'win': 0, 'lose': 0, 'other': 0}
    for r in train:
        o = str(r.get('outcome', '')).lower()
        if o.startswith('win') or '胜' in o:
            outcomes['win'] += 1
        elif o.startswith('lose') or '败' in o or '输' in o:
            outcomes['lose'] += 1
        else:
            outcomes['other'] += 1

    print('[PASS] preflight gate')
    print('train_rows=', len(train))
    print('eval_rows=', len(evals))
    print('law_coverage=', round(law_cov, 4))
    print('outcome_dist=', outcomes)
    print('train_sha256=', sha256_file(TRAIN))
    print('eval_sha256=', sha256_file(EVAL))


if __name__ == '__main__':
    main()
