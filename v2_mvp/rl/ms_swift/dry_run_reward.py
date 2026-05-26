import json
from statistics import mean
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2_mvp.rl.ms_swift.reward_swift_adapter import reward_fn

DATA = Path('v2_mvp/rl/data/2n_rl_data_clean_v1.jsonl')
LIMIT = 20


def main():
    rows = [json.loads(l) for l in DATA.read_text(encoding='utf-8').splitlines() if l.strip()][:LIMIT]
    completions = [str(r.get('reference_output') or r.get('reasoning') or '') for r in rows]
    rewards = reward_fn(completions, rows)

    if not rewards:
        raise SystemExit('[FAIL] no rewards produced')

    print('[PASS] dry-run reward')
    print('samples=', len(rewards))
    print('reward_min=', round(min(rewards), 4))
    print('reward_max=', round(max(rewards), 4))
    print('reward_mean=', round(mean(rewards), 4))

    # law signal sanity from existing reasoning text
    law_pos = 0
    for r in rows:
        t = str(r.get('reasoning') or '')
        if ('劳动合同法' in t) and ('条' in t):
            law_pos += 1
    print('law_text_signal_ratio=', round(law_pos / len(rows), 4))


if __name__ == '__main__':
    main()
