import time

def is_valid(grid, r, c, k):
    for i in range(9):
        if grid[r][i] == k: return False
        if grid[i][c] == k: return False
    R, C = r//3*3, c//3*3
    for i in range(3):
        for j in range(3):
            if grid[R+i][C+j] == k: return False
    return True

def solve(grid, timeout=5):
    start_time = time.time()
    return _solve(grid, start_time, timeout)

def _solve(grid, start_time, timeout):
    if time.time() - start_time > timeout:
        return False
        
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for k in range(1, 10):
                    if is_valid(grid, r, c, k):
                        grid[r][c] = k
                        if _solve(grid, start_time, timeout): return True
                        grid[r][c] = 0
                return False
    return True
