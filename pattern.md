# Coding Pattern Cheatsheet (Python)

A compact reference of the most common interview patterns with minimal, readable templates.

## Pattern → When to use

| Keywords | Pattern |
|----------|---------|
| pair, sorted, complement | Two Pointers / HashMap |
| substring, subarray | Sliding Window |
| prefix, sum equals k | Prefix Sum + HashMap |
| next greater/smaller | Monotonic Stack/Deque |
| k window max | Monotonic Deque |
| search value/answer | Binary Search |
| tree traversal | DFS/BFS |
| graph reachability | DFS/BFS |
| connectivity/groups | Union-Find |
| all combinations/paths | Backtracking |
| intervals, scheduling | Greedy |
| top-k, k-th | Heap |

---

## Two Pointers

Sorted two-sum
```python
def two_sum_sorted(values, target):
    left, right = 0, len(values) - 1
    while left < right:
        s = values[left] + values[right]
        if s == target:
            return [left, right]
        if s < target:
            left += 1
        else:
            right -= 1
    return []
```

Container with most water
```python
def max_area(heights):
    left, right = 0, len(heights) - 1
    best = 0
    while left < right:
        area = (right - left) * min(heights[left], heights[right])
        best = max(best, area)
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return best
```

---

## Sliding Window

Fixed-size max-sum subarray
```python
def max_sum_k(values, k):
    if k <= 0 or k > len(values):
        return 0
    window_sum = sum(values[:k])
    best = window_sum
    for i in range(k, len(values)):
        window_sum += values[i] - values[i - k]
        best = max(best, window_sum)
    return best
```

Variable-size: longest substring with ≤ k distinct chars
```python
def longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    left = 0
    counts = {}
    best = 0
    for right, ch in enumerate(s):
        counts[ch] = counts.get(ch, 0) + 1
        while len(counts) > k:
            left_ch = s[left]
            counts[left_ch] -= 1
            if counts[left_ch] == 0:
                counts.pop(left_ch)
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

## Prefix Sum + HashMap
Subarray sum equals k
```python
def count_subarrays_sum_k(values, k):
    count = 0
    prefix = 0
    seen = {0: 1}
    for x in values:
        prefix += x
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
```

---

## Monotonic Stack / Deque

Next greater element (to the right)
```python
def next_greater(values):
    out = [-1] * len(values)
    stack = []  # indices; decreasing by value
    for i, x in enumerate(values):
        while stack and values[stack[-1]] < x:
            j = stack.pop()
            out[j] = x
        stack.append(i)
    return out
```

Sliding window maximum (deque)
```python
from collections import deque

def sliding_window_max(nums, k):
    if k <= 0:
        return []
    dq = deque()  # indices; values decreasing
    out = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
```

---

## Binary Search

On sorted array
```python
def binary_search(values, target):
    left, right = 0, len(values) - 1
    while left <= right:
        mid = (left + right) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

Binary search on answer (predicate is monotonic)
```python
def min_satisfying(low, high, ok):
    # find smallest x in [low, high] with ok(x) True; return high+1 if none
    while low <= high:
        mid = (low + high) // 2
        if ok(mid):
            high = mid - 1
        else:
            low = mid + 1
    return low
```

---

## Tree Traversal

BFS (level order)
```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
```

---

## Graph Traversal

DFS and BFS (adjacency list)
```python
from collections import deque

def dfs_graph(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for nei in graph.get(start, []):
        if nei not in visited:
            dfs_graph(graph, nei, visited)
    return visited

def bfs_graph(graph, start):
    seen, q = {start}, deque([start])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nei in graph.get(node, []):
            if nei not in seen:
                seen.add(nei)
                q.append(nei)
    return order
```

---

## Union-Find (Disjoint Set)
```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True
```

---

## Backtracking
General template
```python
def backtrack(choices, path, is_valid, add_solution):
    if is_valid(path):
        add_solution(path)
        return
    for choice in choices(path):
        path.append(choice)
        backtrack(choices, path, is_valid, add_solution)
        path.pop()
```

Subsets
```python
def subsets(nums):
    out = []
    def dfs(i, path):
        if i == len(nums):
            out.append(path[:])
            return
        dfs(i + 1, path)
        path.append(nums[i])
        dfs(i + 1, path)
        path.pop()
    dfs(0, [])
    return out
```

---

## Heap (Top-K)
```python
import heapq

def top_k_largest(nums, k):
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        else:
            heapq.heappushpop(heap, x)
    return sorted(heap, reverse=True)
```

---

## Greedy (Intervals)

Non-overlapping intervals to remove
```python
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda it: it[1])
    removed = 0
    prev_end = float('-inf')
    for start, end in intervals:
        if start < prev_end:
            removed += 1
        else:
            prev_end = end
    return removed
```

---

Tips
- Start with keywords → map to pattern.
- Write minimal template first, then fill logic.
- Track invariants (window validity, stack order, visited sets).
- Prefer O(n) or O(n log n) approaches where patterns apply.