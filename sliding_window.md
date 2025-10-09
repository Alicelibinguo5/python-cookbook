Sliding Window — Quick Pattern Recognition & Templates

1) When to think sliding window
- **contiguous** subarray/substring mentioned → sliding window
- **fixed size k** → fixed-size template
- **longest/shortest with constraint** ("at most", "at least", "no more than") → expand + shrink
- **at most K distinct** (or exactly K) → hashmap + shrink; exactly K = atMost(K) - atMost(K-1)
- **cover/contain all chars/pattern, anagram/permutation** → need/missing counts
- **window max/min** (e.g., max in every window) → monotonic deque
- **replace at most K** to make window valid (e.g., repeating chars) → window_len - most_frequent_count <= K
- **binary array with at most K flips** → count "bad" elements; shrink when bad_count > K

2) Fixed-size window (arrays)
```python
from typing import List

def max_sum_subarray_of_size_k(nums: List[int], k: int) -> int:
    window_sum = sum(nums[:k])
    best = window_sum
    for right in range(k, len(nums)):
        window_sum += nums[right] - nums[right - k]
        best = max(best, window_sum)
    return best
```

3) Generic variable-size with constraint (shrink while invalid)
```python
from collections import defaultdict
from typing import Iterable

def longest_substring_no_more_than_k_repeats(s: str, k: int) -> int:
    freq = defaultdict(int)
    left = 0
    best = 0
    for right, ch in enumerate(s):
        freq[ch] += 1
        # shrink while invalid (example condition shown)
        while freq[ch] > k:
            freq[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

4) At most K distinct (exactly K via difference)
```python
from collections import defaultdict

def at_most_k_distinct(s: str, k: int) -> int:
    count = defaultdict(int)
    left = 0
    total = 0
    distinct = 0
    for right, ch in enumerate(s):
        if count[ch] == 0:
            distinct += 1
        count[ch] += 1
        while distinct > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                distinct -= 1
            left += 1
        total += right - left + 1
    return total

def exactly_k_distinct(s: str, k: int) -> int:
    return at_most_k_distinct(s, k) - at_most_k_distinct(s, k - 1)
```

5) Minimum window covering all chars of t (missing counter)
```python
from collections import Counter

def min_window_cover(s: str, t: str) -> str:
    need = Counter(t)
    missing = len(t)
    left = 0
    best = (float('inf'), 0, 0)
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    if best[0] == float('inf'):
        return ""
    _, i, j = best
    return s[i:j+1]
```

6) Find all anagrams of p in s (fixed counts + sliding)
```python
from collections import Counter
from typing import List

def find_anagrams(s: str, p: str) -> List[int]:
    if len(p) > len(s):
        return []
    need = Counter(p)
    window = Counter(s[:len(p)])
    result = []
    if window == need:
        result.append(0)
    for right in range(len(p), len(s)):
        left = right - len(p)
        window[s[right]] += 1
        window[s[left]] -= 1
        if window[s[left]] == 0:
            del window[s[left]]
        if window == need:
            result.append(left + 1)
    return result
```

7) Sliding window maximum (monotonic deque)
```python
from collections import deque
from typing import List

def sliding_window_max(nums: List[int], k: int) -> List[int]:
    dq = deque()  # stores indices; nums[dq] is decreasing
    result = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

8) Longest repeating character replacement (classic trick)
```python
from collections import defaultdict

def character_replacement(s: str, k: int) -> int:
    freq = defaultdict(int)
    left = 0
    best = 0
    max_count_in_window = 0
    for right, ch in enumerate(s):
        freq[ch] += 1
        max_count_in_window = max(max_count_in_window, freq[ch])
        while (right - left + 1) - max_count_in_window > k:
            freq[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

9) Sum-constrained windows (non-negative arrays)
```python
from typing import List

def longest_subarray_sum_at_most_target(nums: List[int], target: int) -> int:
    left = 0
    window_sum = 0
    best = 0
    for right, x in enumerate(nums):
        window_sum += x
        while window_sum > target:
            window_sum -= nums[left]
            left += 1
        best = max(best, right - left + 1)
    return best
```

10) Two quick mappings (recognition → template)
- **"size k"** → fixed-size sum/counter template
- **"at most K distinct"** → hashmap size; shrink when size > K
- **"exactly K distinct"** → atMost(K) - atMost(K-1)
- **"cover all chars of t" / "min window"** → need Counter, missing
- **"find anagrams/permutation"** → equal Counters with size-k slide
- **"max/min in window"** → monotonic deque
- **"replace at most K"** → window_len - max_freq <= K
- **"binary flips at most K"** → count bads; shrink when bads > K



