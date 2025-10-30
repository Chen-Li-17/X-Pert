# Memoization table (global or passed)
memo = {}

# Helper function to calculate sum of digits in base 6
def sum_base_6(n):
    sum_val = 0
    if n == 0:
        return 0
    while n > 0:
        sum_val += n % 6
        n //= 6
    return sum_val

# Precompute powers of 9
powers_of_9 = []
p = 9
while p <= 10**12:
    powers_of_9.append(p)
    p *= 9
# Reversing (trying big notes first) can be a good heuristic
powers_of_9.reverse() 

# Main recursive function
def solve(n):
    if n == 0:
        return 0
    if n in memo:
        return memo[n]

    # Option 1: Pay entirely with powers of 6
    min_cost = sum_base_6(n)

    # Option 2: Try using one power-of-9 note
    for p in powers_of_9:
        if p <= n:
            min_cost = min(min_cost, 1 + solve(n - p))
        else:
            # Since powers_of_9 is sorted descending, we can stop
            continue 
            
    memo[n] = min_cost
    return min_cost

# Main execution
T = int(input())
for _ in range(T):
    X = int(input())
    # Clear memo for each test case
    memo.clear() 
    print(solve(X))