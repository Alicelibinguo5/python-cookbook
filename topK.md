
3 fastest Pythonic ways to compute Top K (sorted, heapq.nlargest, Counter.most_common)
1. hashmap + sorting + slice
```python
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import collections
        freq_map = collections.defaultdict(int) #auto default int 0 
        for n in nums:
            freq_map[n] += 1
        topk = sorted(freq_map.items(), key = lambda x:x[1], reverse= True)[:k]
        return [k for k,v in topk]
```
❌ Downside:
	•	O(n log n) because it sorts the entire list, even if you only need the top k.
    - Python’s sorted is stable → it keeps the original insertion order for ties.
2. heapq.nlargest() + collections.Counter()
```python
import heapq
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. Build hash map: character and how often it appears
        # O(N) time
        count = Counter(nums)   
        return heapq.nlargest(k, freq_map.items(), key=lambda x: x[1])
 ```
- Efficient: O(n log k) instead of O(n log n)
- Under the hood, it keeps a min-heap of size k and discards smaller elements as it scans.
```python
min_heap = []  # will store tuples (freq, num)
for num, freq in freq_map.items():
    heapq.heappush(min_heap, (freq, num))   # push into heap
    if len(min_heap) > k:                   # if heap too big, pop the smallest
        heapq.heappop(min_heap)
```
nsmallest
```python
        from heapq import heappush, heappop
        Freqs = collections.Counter(words)
        return heapq.nsmallest(k, Freqs,
            key=lambda word:(~Freqs[word], word)
        )
```

3. collections.Counter.most_common

```python
from collections import Counter
freq_map = Counter(nums)
topk = freq_map.most_common(k)
```

🧭 Rule of thumb for interviews:
	•	Start with sorting (clear & correct, great when explaining).
	•	If interviewer hints about efficiency → mention heapq.nlargest.
	•	If the problem is literally “frequency count” → drop Counter.most_common for elegance.

If ties matter (like “if frequencies equal, return smaller number first”):
👉 You must add a secondary key to your sort/heap
```python
# Break ties by smaller number first
sorted(freq_map.items(), key=lambda x: (-x[1], x[0]))[:k]
# [(1, 2), (2, 2)] because 1 < 2
```