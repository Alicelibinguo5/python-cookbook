# Pattern of tree problem

# depth first search 
1. define base case
2. main logic w recursion 
```python
def dfs(root):
    if not root: return 0 
    return 1 + dfs(root.left) + dfs(root.right)
``` 
3. if not root: return 0 
4. return 1 + dfs(root.left) + dfs(root.right)  

# Iteration 
1. define stack with tuple (node, value)
2. iterate though stack if not empty 
```python
 while stack:                     
    node, value = stack.pop()
    if node.left:
        stack.append((node.left, value + node.left.val))
    if node.right:
        stack.append((node.right, value + node.right.val))
    if not node.left and not node.right:
        return value
 ```

# BFS 
1. define queue with tuple (node, value)
2. iterate though queue if not empty 
```python
 while queue:
    node, value = queue.pop(0)
    if node.left:
        queue.append((node.left, value + node.left.val))
    if node.right:
        queue.append((node.right, value + node.right.val))
```