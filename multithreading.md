# Multi-threading Patterns (Python)

Concise, reusable patterns for Python multi-threading with practical examples and best practices.

## Basics

- Use [`threading.Thread`](https://docs.python.org/3/library/threading.html#threading.Thread) for I/O-bound tasks, not CPU-bound (GIL limitation).
- Always use [`threading.Lock`](https://docs.python.org/3/library/threading.html#threading.Lock) for shared mutable state.
- Prefer [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) for simpler thread management.
- Use [`queue.Queue`](https://docs.python.org/3/library/queue.html#queue.Queue) for thread-safe communication.

## Basic Thread Creation

```python
import threading
import time
from typing import List

def worker(name: str, delay: float) -> None:
    """Simple worker function that simulates work."""
    print(f"Worker {name} starting")
    time.sleep(delay)
    print(f"Worker {name} finished")

# Create and start threads
threads: List[threading.Thread] = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"T{i}", 1.0))
    t.start()
    threads.append(t)

# Wait for all threads to complete
for t in threads:
    t.join()
```

## Thread-Safe Counter with Lock

```python
import threading
from typing import List

class SafeCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> None:
        with self._lock:
            self._value += 1
    
    def get_value(self) -> int:
        with self._lock:
            return self._value

# Usage
counter = SafeCounter()

def increment_worker(counter: SafeCounter, times: int) -> None:
    for _ in range(times):
        counter.increment()

threads: List[threading.Thread] = []
for i in range(5):
    t = threading.Thread(target=increment_worker, args=(counter, 1000))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(f"Final count: {counter.get_value()}")  # Should be 5000
```

## Producer-Consumer Pattern

```python
import threading
import time
from queue import Queue
from typing import Optional

def producer(q: Queue[Optional[int]], items: int) -> None:
    """Produce items and put them in queue."""
    for i in range(items):
        item = i * i
        q.put(item)
        print(f"Produced: {item}")
        time.sleep(0.1)
    q.put(None)  # Sentinel to signal completion

def consumer(q: Queue[Optional[int]], name: str) -> None:
    """Consume items from queue until sentinel received."""
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        print(f"Consumer {name} processed: {item}")
        time.sleep(0.2)
        q.task_done()

# Setup
q: Queue[Optional[int]] = Queue(maxsize=5)

# Start threads
producer_thread = threading.Thread(target=producer, args=(q, 10))
consumer_thread = threading.Thread(target=consumer, args=(q, "A"))

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
```

## ThreadPoolExecutor (Recommended)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List

def fetch_data(url_id: int) -> dict:
    """Simulate fetching data from URL."""
    time.sleep(1)  # Simulate network delay
    return {"id": url_id, "data": f"content_{url_id}"}

# Process multiple URLs concurrently
url_ids = list(range(5))

with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all tasks
    future_to_id = {executor.submit(fetch_data, url_id): url_id 
                    for url_id in url_ids}
    
    # Process results as they complete
    results: List[dict] = []
    for future in as_completed(future_to_id):
        url_id = future_to_id[future]
        try:
            result = future.result()
            results.append(result)
            print(f"Completed URL {url_id}: {result}")
        except Exception as exc:
            print(f"URL {url_id} generated exception: {exc}")

print(f"Total results: {len(results)}")
```

## Thread-Safe Singleton Pattern

```python
import threading
from typing import Optional

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'DatabaseConnection':
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.connection_string = "db://localhost:5432"
                    self._initialized = True
    
    def query(self, sql: str) -> str:
        return f"Executing: {sql}"

# Usage - all threads get same instance
def worker() -> None:
    db = DatabaseConnection()
    print(f"Thread {threading.current_thread().name}: {id(db)}")

threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Event Coordination

```python
import threading
import time
from typing import List

def waiter(event: threading.Event, name: str) -> None:
    """Wait for event to be set."""
    print(f"{name} waiting for event")
    event.wait()
    print(f"{name} received event!")

def setter(event: threading.Event) -> None:
    """Set event after delay."""
    time.sleep(2)
    print("Setting event")
    event.set()

# Coordinate multiple threads with event
event = threading.Event()

# Start waiters
waiters: List[threading.Thread] = []
for i in range(3):
    t = threading.Thread(target=waiter, args=(event, f"Waiter-{i}"))
    t.start()
    waiters.append(t)

# Start setter
setter_thread = threading.Thread(target=setter, args=(event,))
setter_thread.start()

# Wait for completion
setter_thread.join()
for t in waiters:
    t.join()
```

## Barrier Synchronization

```python
import threading
import time
import random
from typing import List

def worker(barrier: threading.Barrier, worker_id: int) -> None:
    """Worker that synchronizes at barrier."""
    # Simulate different work times
    work_time = random.uniform(1, 3)
    print(f"Worker {worker_id} working for {work_time:.1f}s")
    time.sleep(work_time)
    
    print(f"Worker {worker_id} waiting at barrier")
    try:
        barrier.wait()
        print(f"Worker {worker_id} passed barrier")
    except threading.BrokenBarrierError:
        print(f"Worker {worker_id}: barrier broken")

# Synchronize 4 workers
num_workers = 4
barrier = threading.Barrier(num_workers)

workers: List[threading.Thread] = []
for i in range(num_workers):
    t = threading.Thread(target=worker, args=(barrier, i))
    t.start()
    workers.append(t)

for t in workers:
    t.join()
```

## Context Manager for Thread-Safe Resource

```python
import threading
from contextlib import contextmanager
from typing import Generator, Any

class SharedResource:
    def __init__(self) -> None:
        self._data = {"count": 0}
        self._lock = threading.Lock()
    
    @contextmanager
    def acquire(self) -> Generator[dict, None, None]:
        """Context manager for thread-safe access."""
        self._lock.acquire()
        try:
            yield self._data
        finally:
            self._lock.release()

def worker(resource: SharedResource, worker_id: int) -> None:
    """Worker that safely modifies shared resource."""
    with resource.acquire() as data:
        current = data["count"]
        # Simulate some processing
        import time
        time.sleep(0.01)
        data["count"] = current + 1
        print(f"Worker {worker_id}: count = {data['count']}")

# Usage
resource = SharedResource()
threads = [threading.Thread(target=worker, args=(resource, i)) 
           for i in range(10)]

for t in threads:
    t.start()
for t in threads:
    t.join()

with resource.acquire() as data:
    print(f"Final count: {data['count']}")
```

## Tips

- **Use ThreadPoolExecutor** for most cases - simpler than manual thread management.
- **Always protect shared state** with locks or use thread-safe data structures.
- **Avoid deadlocks** by acquiring locks in consistent order across threads.
- **Use daemon threads** for background tasks that should exit when main program ends.
- **Handle exceptions** in worker threads - they don't propagate to main thread.
- **Consider asyncio** for I/O-bound concurrency instead of threads.
- **Profile first** - threading adds overhead and complexity.

## Common Pitfalls

- **Race conditions**: Always use locks for shared mutable state.
- **Deadlocks**: Acquire multiple locks in same order everywhere.
- **GIL limitation**: Python threads don't help with CPU-bound tasks.
- **Exception handling**: Unhandled exceptions in threads are silent.
- **Resource leaks**: Always join threads or use context managers.