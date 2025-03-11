# Speeding up pytorch code


### 1. Use torch.gather for batch indexing

```
# ❌ BAD: Makes many independent calls to PyTorch's C++ backend (slow)
torch.cat([mat[:, t] for t in range(T)], dim=-1)  # Shape: (B, T)

# ✅ GOOD: Single optimized operation (fast)
torch.gather(mat, T.unsqueeze(0).repeat(B, 1), dim=-1)  # Shape: (B, T)
```




# Common bugs

### 1. Loops and Lambda functions don't play well

```
# ❌ BAD: All lambdas capture final loop values
funcs_list = [
    lambda a, b: a/b
    for (a, b) in zip ([1, 1, 1], [2, 4, 8])
]
print([f() for f in funcs_list])
>>> [1/8, 1/8, 1/8]  # wrong

# ✅ GOOD: Immediate binding via default arguments
def create_fn(a, b):
    return lambda a, b: a/b

funcs_list = [ 
    create_fn(a, b)
    for (a, b) in zip ([1, 1, 1], [2, 4, 8])
]
print([f() for f in funcs_list])
>>> [1/2, 1/4, 1/8] # correct
```