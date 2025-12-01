import taichi as ti
import time
import numpy as np
import math
import matplotlib.pyplot as plt

ti.init(arch=ti.cuda)  # 若没有 GPU，改成 ti.cpu

# -----------------------
# problem size & fields
# -----------------------
n = 200  # 可以改成更大以观察差异（例如 500、1000）
A = ti.field(dtype=ti.float32, shape=(n, n))
x = ti.field(dtype=ti.float32, shape=n)       # 当前解
new_x = ti.field(dtype=ti.float32, shape=n)   # 用于 Jacobi 的新解缓冲
b = ti.field(dtype=ti.float32, shape=n)

# 临时向量，用于 CG
r = ti.field(dtype=ti.float32, shape=n)
p = ti.field(dtype=ti.float32, shape=n)
Ap = ti.field(dtype=ti.float32, shape=n)

# -----------------------
# kernels: init / matvec / dot / axpy / copy
# -----------------------
@ti.kernel
def init_system():
    # 三对角 SPD 矩阵：2 on diag, -1 on off-diags
    for i in range(n):
        for j in range(n):
            A[i, j] = 0.0
    for i in range(n):
        A[i, i] = 2.0
        if i - 1 >= 0:
            A[i, i - 1] = -1.0
        if i + 1 < n:
            A[i, i + 1] = -1.0

    for i in range(n):
        b[i] = float(i + 1)  # RHS
        x[i] = 0.0
        new_x[i] = 0.0
        r[i] = 0.0
        p[i] = 0.0
        Ap[i] = 0.0

@ti.kernel
def matvec(x_in: ti.template(), y_out: ti.template()):
    # y_out = A @ x_in
    for i in range(n):
        s = 0.0
        # simple dense matvec (for tri-diagonal A this is fine)
        for j in range(n):
            s += A[i, j] * x_in[j]
        y_out[i] = s

@ti.kernel
def dot(x_in: ti.template(), y_in: ti.template()) -> ti.f32:
    s = 0.0
    for i in range(n):
        s += x_in[i] * y_in[i]
    return s

@ti.kernel
def axpy(a: ti.f32, x_in: ti.template(), y_out: ti.template()):
    # y_out += a * x_in
    for i in range(n):
        y_out[i] = y_out[i] + a * x_in[i]

@ti.kernel
def x_plus_ax(x_in: ti.template(), a: ti.f32, y_in: ti.template(), out: ti.template()):
    # out = x_in + a * y_in
    for i in range(n):
        out[i] = x_in[i] + a * y_in[i]

@ti.kernel
def copy_vec(src: ti.template(), dst: ti.template()):
    for i in range(n):
        dst[i] = src[i]

@ti.kernel
def set_zero_vec(v: ti.template()):
    for i in range(n):
        v[i] = 0.0

@ti.kernel
def jacobi_one_iter():
    # new_x = D^{-1} (b - (A - D) x)
    for i in range(n):
        rloc = b[i]
        diag = A[i, i]
        # accumulate off-diagonal contributions using current x
        for j in range(n):
            if j != i:
                rloc -= A[i, j] * x[j]
        new_x[i] = rloc / diag
    # copy back
    for i in range(n):
        x[i] = new_x[i]

@ti.kernel
def residual_squared() -> ti.f32:
    s = 0.0
    for i in range(n):
        rloc = b[i]
        for j in range(n):
            rloc -= A[i, j] * x[j]
        s += rloc * rloc
    return s

# -----------------------
# Python implementations of solver loops (call kernels)
# -----------------------
def solve_jacobi(max_iters=50000, tol=1e-6, print_every=100, record_history=False):
    # x is global field; assume init_system() already set x=0
    history = [] if record_history else None
    start = time.time()
    initial_res = math.sqrt(float(residual_squared()))
    
    for k in range(1, max_iters + 1):
        iter_start = time.time()
        jacobi_one_iter()              # one Jacobi iteration (kernel)
        iter_time = time.time() - iter_start
        
        res = math.sqrt(float(residual_squared()))
        if record_history:
            history.append((k, res, iter_time))
        
        if (k % print_every) == 0 or k == 1:
            print(f"[Jacobi] iter {k:6d}, res L2 = {res:.6e}, iter_time = {iter_time*1000:.3f}ms")
            if res < tol:
                end = time.time()
                total_time = end - start
                return k, total_time, res, history, initial_res
    end = time.time()
    res = math.sqrt(float(residual_squared()))
    total_time = end - start
    return max_iters, total_time, res, history, initial_res

def solve_cg(max_iters=600, tol=1e-6, print_every=100, record_history=False):
    # standard CG implemented with Taichi kernels for ops
    # initialize: r = b - A x  (x initially zero), p = r
    history = [] if record_history else None
    matvec(x, Ap)                     # Ap = A @ x
    # compute r = b - Ap
    @ti.kernel
    def init_r_p():
        for i in range(n):
            r[i] = b[i] - Ap[i]
            p[i] = r[i]
    init_r_p()

    rr = float(dot(r, r))
    rr0 = rr if rr > 0 else 1.0
    initial_res = math.sqrt(rr0)
    start = time.time()

    for k in range(1, max_iters + 1):
        iter_start = time.time()
        matvec(p, Ap)                 # Ap = A @ p
        pAp = float(dot(p, Ap))
        if abs(pAp) < 1e-6:
            # breakdown
            print("[CG] tiny p^T A p -> terminate")
            break
        alpha = rr / pAp               # scalar

        # x = x + alpha * p
        axpy(alpha, p, x)

        # r = r - alpha * Ap
        @ti.kernel
        def update_r():
            for i in range(n):
                r[i] = r[i] - alpha * Ap[i]
        update_r()

        rr_new = float(dot(r, r))
        res = math.sqrt(rr_new)
        iter_time = time.time() - iter_start
        
        if record_history:
            history.append((k, res, iter_time))

        if (k % print_every) == 0 or k == 1:
            print(f"[CG] iter {k:6d}, res L2 = {res:.6e}, iter_time = {iter_time*1000:.3f}ms")
            if res < tol:
                end = time.time()
                total_time = end - start
                return k, total_time, res, history, initial_res

        beta = rr_new / (rr + 1e-30)
        # p = r + beta * p
        @ti.kernel
        def update_p(beta: ti.f32):
            for i in range(n):
                p[i] = r[i] + beta * p[i]
        update_p(beta)

        rr = rr_new

    end = time.time()
    res = math.sqrt(float(dot(r, r)))
    total_time = end - start
    return k, total_time, res, history, initial_res

# -----------------------
# helper: to numpy (for final check)
# -----------------------
def to_numpy():
    A_np = np.zeros((n, n), dtype=np.float32)
    b_np = np.zeros(n, dtype=np.float32)
    x_np = np.zeros(n, dtype=np.float32)
    for i in range(n):
        b_np[i] = b[i]
        x_np[i] = x[i]
        for j in range(n):
            A_np[i, j] = A[i, j]
    return A_np, b_np, x_np

# -----------------------
# Performance comparison functions
# -----------------------
def print_performance_table(jacobi_results, cg_results):
    """打印性能对比表格"""
    iters_j, time_j, res_j, history_j, init_res_j = jacobi_results
    iters_cg, time_cg, res_cg, history_cg, init_res_cg = cg_results
    
    print("\n" + "="*80)
    print("性能对比表 (Performance Comparison Table)")
    print("="*80)
    print(f"{'指标':<20} {'Jacobi':<30} {'CG':<30}")
    print("-"*80)
    print(f"{'迭代次数':<20} {iters_j:<30} {iters_cg:<30}")
    print(f"{'总时间 (s)':<20} {time_j:<30.6f} {time_cg:<30.6f}")
    print(f"{'平均每次迭代时间 (ms)':<20} {time_j/iters_j*1000:<30.3f} {time_cg/iters_cg*1000:<30.3f}")
    print(f"{'最终残差':<20} {res_j:<30.6e} {res_cg:<30.6e}")
    print(f"{'初始残差':<20} {init_res_j:<30.6e} {init_res_cg:<30.6e}")
    print(f"{'残差下降倍数':<20} {init_res_j/res_j:<30.2e} {init_res_cg/res_cg:<30.2e}")
    
    if time_j > 0:
        speedup = time_j / time_cg
        print(f"{'CG相对Jacobi加速比':<20} {speedup:<30.2f}x")
    
    if iters_j > 0:
        iter_ratio = iters_j / iters_cg
        print(f"{'CG迭代次数相对Jacobi':<20} {iter_ratio:<30.2f}x (更少)")
    
    print("="*80)

def plot_convergence_history(jacobi_results, cg_results, save_path=None):
    """绘制收敛曲线"""
    iters_j, time_j, res_j, history_j, init_res_j = jacobi_results
    iters_cg, time_cg, res_cg, history_cg, init_res_cg = cg_results
    
    if history_j is None or history_cg is None or len(history_j) == 0 or len(history_cg) == 0:
        print("警告: 未记录收敛历史或历史为空，跳过绘图")
        return
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 残差 vs 迭代次数
    plt.subplot(1, 2, 1)
    iter_j, res_j_hist, _ = zip(*history_j)
    iter_cg, res_cg_hist, _ = zip(*history_cg)
    
    plt.semilogy(iter_j, res_j_hist, 'b-', label='Jacobi', linewidth=2)
    plt.semilogy(iter_cg, res_cg_hist, 'r-', label='CG', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('残差 (L2范数)', fontsize=12)
    plt.title('收敛曲线: 残差 vs 迭代次数', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 子图2: 残差 vs 时间
    plt.subplot(1, 2, 2)
    _, _, time_j_hist = zip(*history_j)
    _, _, time_cg_hist = zip(*history_cg)
    
    cumtime_j = np.cumsum([0] + list(time_j_hist))
    cumtime_cg = np.cumsum([0] + list(time_cg_hist))
    
    # plt.semilogy(cumtime_j[:len(res_j_hist)], res_j_hist, 'b-', label='Jacobi', linewidth=2)
    # plt.semilogy(cumtime_cg[:len(res_cg_hist)], res_cg_hist, 'r-', label='CG', linewidth=2)
    # plt.xlabel('累计时间 (s)', fontsize=12)
    # plt.ylabel('残差 (L2范数)', fontsize=12)
    # plt.title('收敛曲线: 残差 vs 时间', fontsize=14)
    # plt.legend(fontsize=11)
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"\n收敛曲线已保存至: {save_path}")
    # else:
    #     plt.show()

# -----------------------
# main: run both solvers and compare
# -----------------------
if __name__ == "__main__":
    print("="*80)
    print("Jacobi 与 CG 迭代方法性能对比")
    print("="*80)
    print(f"问题规模: n = {n}")
    print("初始化系统...")
    init_system()

    max_iters = 50000
    tol = 1e-6
    print_every = 2000
    record_history = True  # 记录收敛历史用于绘图

    # --- Jacobi ---
    print("\n" + "="*80)
    print("=== Jacobi 迭代方法 ===")
    print("="*80)
    # reset x
    @ti.kernel
    def clear_x():
        for i in range(n):
            x[i] = 0.0
    clear_x()

    jacobi_results = solve_jacobi(max_iters=max_iters, tol=tol, 
                                   print_every=print_every, record_history=record_history)
    iters_j, time_j, res_j, history_j, init_res_j = jacobi_results
    print(f"\n[Jacobi] 完成: 迭代次数={iters_j}, 总时间={time_j:.6f}s, 最终残差={res_j:.6e}")

    A_np, b_np, x_j_np = to_numpy()
    x_ref = np.linalg.solve(A_np, b_np)
    err_j = np.linalg.norm(x_ref - x_j_np)
    print(f"[Jacobi] 与numpy解的L2误差 = {err_j:.6e}")

    # --- CG ---
    print("\n" + "="*80)
    print("=== 共轭梯度法 (CG) ===")
    print("="*80)
    # reset x
    init_system()   # also resets x,b,A
    @ti.kernel
    def clear_x2():
        for i in range(n):
            x[i] = 0.0
    clear_x2()

    cg_results = solve_cg(max_iters=500, tol=tol,
                         print_every=50, record_history=record_history)
    iters_cg, time_cg, res_cg, history_cg, init_res_cg = cg_results
    print(f"\n[CG] 完成: 迭代次数={iters_cg}, 总时间={time_cg:.6f}s, 最终残差={res_cg:.6e}")

    A_np, b_np, x_cg_np = to_numpy()
    err_cg = np.linalg.norm(x_ref - x_cg_np)
    print(f"[CG] 与numpy解的L2误差 = {err_cg:.6e}")

    # --- 性能对比分析 ---
    print_performance_table(jacobi_results, cg_results)
    
    # --- 绘制收敛曲线 ---
    try:
        plot_convergence_history(jacobi_results, cg_results, 
                                save_path="jacobi_cg_convergence.png")
    except Exception as e:
        print(f"\n绘图时出错 (可能缺少matplotlib): {e}")
        print("跳过绘图，但性能对比数据已完整输出")
    
    # --- 详细分析 ---
    print("\n" + "="*80)
    print("详细分析 (Detailed Analysis)")
    print("="*80)
    
    if history_j and history_cg:
        # 找到达到相同残差水平所需的迭代次数
        target_res = 1e-6
        iter_j_target = None
        iter_cg_target = None
        
        for iter_num, res, _ in history_j:
            if res < target_res and iter_j_target is None:
                iter_j_target = iter_num
                break
        
        for iter_num, res, _ in history_cg:
            if res < target_res and iter_cg_target is None:
                iter_cg_target = iter_num
                break
        
        if iter_j_target and iter_cg_target:
            print(f"\n达到残差 {target_res:.1e} 所需的迭代次数:")
            print(f"  Jacobi: {iter_j_target} 次")
            print(f"  CG:     {iter_cg_target} 次")
            print(f"  CG 比 Jacobi 快 {iter_j_target/iter_cg_target:.2f} 倍")
    
    print("\n结论:")
    if iters_cg < iters_j:
        print(f"  ✓ CG 方法收敛更快，迭代次数仅为 Jacobi 的 {iters_cg/iters_j*100:.1f}%")
    else:
        print(f"  - Jacobi 方法在此问题上迭代次数更少")
    
    if time_cg < time_j:
        print(f"  ✓ CG 方法总时间更短，比 Jacobi 快 {time_j/time_cg:.2f} 倍")
    else:
        print(f"  - Jacobi 方法在此问题上总时间更短")
    
    if res_cg < res_j:
        print(f"  ✓ CG 方法达到更低的残差")
    else:
        print(f"  - Jacobi 方法达到更低的残差")
    
    print("="*80)
