# Hash Map Group-By Patterns (list[dict])

Common, reliable ways to group a list of dictionaries in Python.

## 1) Group rows by a single key (collect into lists)

```python
from collections import defaultdict
from typing import Any, Dict, List

rows: List[Dict[str, Any]] = [
    {"team": "A", "user": "u1", "score": 5},
    {"team": "B", "user": "u2", "score": 3},
    {"team": "A", "user": "u3", "score": 7},
]

by_team: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
for row in rows:
    key = row.get("team")  # safe if key may be missing
    by_team[key].append(row)

# by_team["A"] → two dicts; by_team["B"] → one dict
```

## 2) Group and aggregate (sum/count/avg)

```python
from collections import defaultdict
from typing import Any, Dict, List

rows: List[Dict[str, Any]] = [
    {"team": "A", "score": 5},
    {"team": "B", "score": 3},
    {"team": "A", "score": 7},
]

sum_by_team: Dict[str, int] = defaultdict(int)
count_by_team: Dict[str, int] = defaultdict(int)

for row in rows:
    k = row.get("team")
    v = int(row.get("score", 0))
    sum_by_team[k] += v
    count_by_team[k] += 1

avg_by_team = {k: (sum_by_team[k] / count_by_team[k]) for k in sum_by_team}
```

- For min/max, initialize with `defaultdict(lambda: +float("inf"))` or `-float("inf")` and use `min()`/`max()` inside the loop.

Examples:
```python
from collections import defaultdict
from typing import Dict

# Min/Max score per team (numeric)
min_score_by_team: Dict[str, float] = defaultdict(lambda: float("inf"))
max_score_by_team: Dict[str, float] = defaultdict(lambda: float("-inf"))
for row in rows:
    k = row.get("team")
    v = float(row.get("score", 0))
    min_score_by_team[k] = min(min_score_by_team[k], v)
    max_score_by_team[k] = max(max_score_by_team[k], v)

# Optional: replace +inf with None for groups with no data beyond defaults
min_score_clean = {k: (None if v == float("inf") else v) for k, v in min_score_by_team.items()}

# Multi-key min using tuple key
from typing import Tuple
min_by_group: Dict[Tuple[str, str], float] = defaultdict(lambda: float("inf"))
for row in rows:
    key = (row.get("team"), row.get("region"))
    val = float(row.get("score", 0))
    min_by_group[key] = min(min_by_group[key], val)
```

## 3) Multi-key group (tuple keys)

```python
from collections import defaultdict
from typing import Any, Dict, List, Tuple

rows: List[Dict[str, Any]] = [
    {"team": "A", "region": "us", "score": 5},
    {"team": "A", "region": "eu", "score": 6},
    {"team": "A", "region": "us", "score": 7},
]

sum_by_group: Dict[Tuple[str, str], int] = defaultdict(int)
for row in rows:
    key = (row.get("team"), row.get("region"))
    sum_by_group[key] += int(row.get("score", 0))

# key ("A", "us") → 12
```

## 4) itertools.groupby (requires sorting)

```python
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, List

rows: List[Dict[str, Any]] = [
    {"team": "B", "score": 3},
    {"team": "A", "score": 5},
    {"team": "A", "score": 7},
]

rows_sorted = sorted(rows, key=itemgetter("team"))
by_team = {k: list(g) for k, g in groupby(rows_sorted, key=itemgetter("team"))}
```

- Use when input is already sorted or sorting cost is acceptable. Otherwise prefer `defaultdict`.

## 5) pandas one-liners (recommended for data work)

```python
import pandas as pd

# rows: list[dict]
df = pd.DataFrame(rows)

# Counts per team
counts = df.groupby("team").size().rename("count").reset_index()

# Sum and average score per team
agg = df.groupby("team")["score"].agg(sum_score="sum", avg_score="mean").reset_index()

# Multi-key group
agg2 = df.groupby(["team", "region"]).agg(sum_score=("score", "sum")).reset_index()
```

## Tips

- Normalize keys upfront (lowercase/strip) to avoid splitting groups by casing/whitespace.
- Use `tuple` keys for multi-column groups; avoid mutable keys.
- Prefer `pandas` for complex aggregations, joins, or when chaining multiple transforms.
- Validate missing keys with `.get()` and default values to keep the pipeline robust.
