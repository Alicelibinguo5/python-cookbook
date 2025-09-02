# Queue (Quick Reference)

A minimal set of Python queue patterns. Prefer `collections.deque` for O(1) FIFO operations.

## Basics with deque

```python
from collections import deque

q: deque[int] = deque()

# enqueue
q.append(10)
q.append(20)

# dequeue (raises IndexError if empty)
first = q.popleft()

# peek (safe check)
front = q[0] if q else None

# size / emptiness
n = len(q)
is_empty = not q
```

- `append`/`popleft` are O(1). Avoid `list.pop(0)` which is O(n).
- `appendleft`/`pop` support double-ended usage if needed.

## BFS (level-order) template

```python
from collections import deque
from typing import Dict, Iterable, List

def bfs_order(graph: Dict[int, Iterable[int]], start: int) -> List[int]:
    visited: set[int] = {start}
    queue: deque[int] = deque([start])
    order: List[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for nei in graph.get(node, []):
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
    return order
```

- For grid BFS, enqueue coordinates `(r, c)` and guard bounds/visited.

## Sliding window maximum (monotonic deque)

```python
from collections import deque
from typing import List

def sliding_window_max(nums: List[int], k: int) -> List[int]:
    if k <= 0:
        return []
    dq: deque[int] = deque()  # stores indices; values decreasing in dq
    out: List[int] = []

    for i, x in enumerate(nums):
        # remove out-of-window indices from front
        while dq and dq[0] <= i - k:
            dq.popleft()
        # maintain decreasing values
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
```

- Each index enters and leaves the deque at most once → O(n).

## Thread-safe queue (producer/consumer)

```python
from queue import Queue
from threading import Thread

q: Queue[int] = Queue()

def worker() -> None:
    while True:
        item = q.get()
        if item is None:  # sentinel to stop
            break
        # process item
        q.task_done()

# start workers
workers = [Thread(target=worker, daemon=True) for _ in range(2)]
for t in workers:
    t.start()

# produce
for x in range(5):
    q.put(x)

q.join()  # wait until all items processed
for _ in workers:
    q.put(None)  # stop
for t in workers:
    t.join()
```

- Use `queue.Queue` for threads; `deque` is not thread-safe for multi-producer/consumer.

## Tips

- Use `deque` for FIFO; use `heapq` for priority queues.
- For fixed-size windows, pop expired indices before using the front.
- Check `.get()`/`.task_done()` balance when using `queue.Queue`.
