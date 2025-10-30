# import sys

# def digit_sum_base(n, b):
#     s = 0
#     while n:
#         s += n % b
#         n //= b
#     return s  # n==0 时返回0，表示不需要任何该系纸币

# def solve_one(X):
#     # 生成所有 <=X 的6的幂
#     p6 = [1]
#     while p6[-1] <= X // 6:
#         p6.append(p6[-1] * 6)
#     p6 = p6[::-1]  # 从大到小便于剪枝（可选）

#     best = digit_sum_base(X, 6)  # 只用6系的上界
#     best = min(best, digit_sum_base(X, 9))  # 只用9系的上界

#     # DFS：逐位决定用了几张该位（相当于该位的“6进制数字”）
#     def dfs(i, sum6, cnt6):
#         nonlocal best
#         # 剪枝
#         if cnt6 >= best or sum6 > X:
#             return
#         # 用当前组合评估：右边全部用9系
#         rest = X - sum6
#         best = min(best, cnt6 + digit_sum_base(rest, 9))
#         # 继续放更小的一位6^i
#         if i == len(p6):
#             return
#         coin = p6[i]
#         # 这一位最多还能放的张数（不能超过剩余金额 / coin）
#         max_take = min((X - sum6) // coin, 8)  # 超过8也没必要，多的可以由更低位替代
#         # 尝试 0..max_take
#         for t in range(1, max_take + 1):
#             dfs(i + 1, sum6 + t * coin, cnt6 + t)
#         dfs(i + 1, sum6, cnt6)  # 也要试不取这一位

#     dfs(0, 0, 0)
#     return best

# def main():
#     T = int(input())
#     out = []
#     for _ in range(T):
#         X = int(input())
#         out.append(str(solve_one(X)))
#     print("\n".join(out))

# if __name__ == "__main__":
#     main()


import sys

def digit_sum_base(n, b):
    # n==0 -> 0
    s = 0
    while n:
        n, r = divmod(n, b)
        s += r
    return s

def solve_one(X: int) -> int:
    # 预生成 6 的幂（<= X）
    p6 = [1]
    while p6[-1] <= X // 6:
        p6.append(p6[-1] * 6)
    p6.reverse()  # 大到小，优先放大面额

    # 初始上界：只用 6 系 or 只用 9 系
    best = min(digit_sum_base(X, 6), digit_sum_base(X, 9))

    # 记忆化：记录在 (idx, sum6) 下已用最少的张数，避免重复
    # 由于 sum6 会很大，改为记忆 (idx, rem)；其中 rem = X - sum6
    # 并且只在“cnt6 >= 已记录的更优解”时剪枝
    seen = {}

    def dfs(idx: int, rem: int, cnt6: int):
        nonlocal best
        # 剪枝 1：张数已不可能更优
        if cnt6 >= best:
            return
        # 剪枝 2：负数（超付）
        if rem < 0:
            return
        # 剪枝 3：记忆化
        key = (idx, rem)
        if key in seen and seen[key] <= cnt6:
            return
        seen[key] = cnt6

        # 以“右半全用 9 系”评估下界，更新答案
        best = min(best, cnt6 + digit_sum_base(rem, 9))

        # 到底：没有更小的 6^i 可放
        if idx == len(p6):
            return

        coin = p6[idx]

        # 当前面额最多可放多少张
        max_take = rem // coin
        if max_take > 8:  # ≥9 不划算（可进位成更高位，不会更优）
            max_take = 8

        # 关键加速：只枚举“最可能的少量 t”
        # 令 q = rem // coin 的“9 取模意义上的就近值”，
        # 经验上 S9(rem - t*coin) 在 t≈q 处最小；取 q±2 的窗口即可。
        q = rem // coin
        cand = set()
        for t in (q-2, q-1, q, q+1, q+2):
            if 0 <= t <= max_take:
                cand.add(t)
        # 也要确保 0 和 max_take 在候选里（有时极端最优在端点）
        cand.add(0)
        cand.add(max_take)

        # 为了更快，按“乐观估计的总代价”排序再深入
        cand = sorted(
            cand,
            key=lambda t: cnt6 + t + digit_sum_base(rem - t*coin, 9)
        )

        for t in cand:
            if cnt6 + t >= best:  # 轻剪枝
                continue
            dfs(idx + 1, rem - t * coin, cnt6 + t)

    dfs(0, X, 0)
    return best

def main():
    T = int(input())
    ans = []
    for _ in range(T):
        X = int(input())
        ans.append(str(solve_one(X)))
    print("\n".join(ans))

if __name__ == "__main__":
    main()
