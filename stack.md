# Stack Patterns (Coding Interview)

Concise, reusable patterns for solving common stack problems in Python.

## Basics

- Use list as a stack: `append` to push, `pop` to pop (amortized O(1)).
- Prefer storing indices when you need distances or to write into an output array.
- Monotonic stacks maintain increasing/decreasing order and solve many O(n) "next" queries.

## Next Greater Element (right)

```python
from typing import List

def next_greater_elements(values: List[int]) -> List[int]:
    result = [-1] * len(values)
    stack: List[int] = []  # indices; stack keeps decreasing values
    for i, x in enumerate(values):
        while stack and values[stack[-1]] < x:
            j = stack.pop()
            result[j] = x
        stack.append(i)
    return result
```

- Variants:
  - Next greater to left: traverse left→right, compare with stack top for current `i`.
  - Next smaller: flip inequality.

## Monotonic Stack Template (indices)

```python
from typing import Callable, List

def resolve_next_by(values: List[int], condition: Callable[[int, int], bool]) -> List[int]:
    """Return index of next element i where condition(values[top], values[i]) is True.
    If none, returns -1. Example: next greater → lambda a, b: a < b
    """
    result = [-1] * len(values)
    stack: List[int] = []
    for i, v in enumerate(values):
        while stack and condition(values[stack[-1]], v):
            j = stack.pop()
            result[j] = i
        stack.append(i)
    return result
```

## Daily Temperatures (days to next warmer)

```python
from typing import List

def daily_temperatures(temps: List[int]) -> List[int]:
    ans = [0] * len(temps)
    stack: List[int] = []  # indices; decreasing by temperature
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            ans[j] = i - j
        stack.append(i)
    return ans
```

## Valid Parentheses

```python
from typing import Dict

def is_valid_parentheses(s: str) -> bool:
    pairs: Dict[str, str] = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        # ignore other characters if present
    return not stack
```

## Evaluate Reverse Polish Notation (RPN)

```python
from typing import Iterable

def eval_rpn(tokens: Iterable[str]) -> int:
    stack: list[int] = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),  # truncate toward zero
    }
    for tok in tokens:
        if tok in ops:
            b = stack.pop()
            a = stack.pop()
            stack.append(ops[tok](a, b))
        else:
            stack.append(int(tok))
    return stack[-1]
```

## Min Stack (O(1) get_min)

Store pairs `(value, current_min)`.

```python
class MinStack:
    def __init__(self) -> None:
        self._data: list[tuple[int, int]] = []

    def push(self, x: int) -> None:
        current_min = x if not self._data else min(x, self._data[-1][1])
        self._data.append((x, current_min))

    def pop(self) -> int:
        return self._data.pop()[0]

    def top(self) -> int:
        return self._data[-1][0]

    def get_min(self) -> int:
        return self._data[-1][1]
```

## Stock Span (consecutive days ≤ today)

```python
from typing import List

def stock_span(prices: List[int]) -> List[int]:
    span = [0] * len(prices)
    stack: List[int] = []  # indices; decreasing by price
    for i, p in enumerate(prices):
        while stack and prices[stack[-1]] <= p:
            stack.pop()
        span[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)
    return span
```

## Decode String (k[encoded])

```python
from typing import List

def decode_string(s: str) -> str:
    count_stack: List[int] = []
    string_stack: List[str] = []
    current: list[str] = []
    k = 0
    for ch in s:
        if ch.isdigit():
            k = k * 10 + int(ch)
        elif ch == '[':
            count_stack.append(k)
            string_stack.append(''.join(current))
            current = []
            k = 0
        elif ch == ']':
            repeat = count_stack.pop()
            prev = string_stack.pop()
            current = [prev + ''.join(current) * repeat]
        else:
            current.append(ch)
    return ''.join(current)
```

## Tips

- Prefer indices over values on the stack for distance calculations.
- Write the invariant: what order does the stack maintain? What does each element represent?
- Many problems are O(n) if each index is pushed and popped at most once.
