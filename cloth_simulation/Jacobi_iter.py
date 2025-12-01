import taichi as ti
import math
import numpy as np

ti.init(arch=ti.cuda)

n = 20

A = ti.field(dtype=ti.float32, shape=(n, n))

x = ti.field(dtype=ti.float32, shape=n)
new_x = ti.field(dtype=ti.float32, shape=n)

b = ti.field(dtype=ti.float32, shape=n)

@ti.kernel
def init():
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
        b[i] = float(i + 1)

    # 初始解 x 全 0
    for i in range(n):
        x[i] = 0.0
        new_x[i] = 0.0

# ---- jacobi ---- A = D + L + U 分解
@ti.kernel
def iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
        new_x[i] = r / A[i, i]

    for i in range(n):
        x[i] = new_x[i]

# @ti.kernel
# def iterate_CG(x : ti.template()):
#     r0 = b - A @ x
#     p0 = r0
#     r_f = r0
#     k = 0
#     for _ in range(50):
#         k += 1
#         alpha = r0.dot(r0) / (p0.dot(A @ p0))
#         x += alpha * p0
#         r1 = r0 - alpha * (A @ p0)
#
#         if r1.norm() / r_f.norm() < 1e-5 or k > 10:
#             break
#
#         beta = r1.dot(r1) / (r0.dot(r0))
#         p1 = r1 + beta * p0
#
#         p0 = p1
#         r0 = r1


@ti.kernel
def residual_sq() -> ti.f32:
    s: ti.f32 = 0.0
    for i in range(n):
        r = b[i]
        for j in range(n):
            r -= A[i, j] * x[j]
        s += r * r
    return s




def to_numpy():
    Ax = np.zeros((n, n), dtype=np.float32)
    bx = np.zeros(n, dtype=np.float32)
    xx = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bx[i] = b[i]
        xx[i] = x[i]
        for j in range(n):
            Ax[i, j] = A[i, j]
    return Ax, bx, xx




if __name__ == "__main__":
    init()

    max_iters = 5000
    tol = 1e-8
    print_every = 100

    print("开始 Jacobi 迭代求解 A x = b（n = {}）".format(n))

    for k in range(max_iters):
        iterate_CG(x)
        if (k + 1) % print_every == 0 or k == 0:
            res = math.sqrt(float(residual_sq()))
            print(f"iter {k+1:5d}, residual L2 = {res:.6e}")
            if res < tol:
                print("达到收敛阈值，退出迭代。")
                break

    A_np, b_np, x_np = to_numpy()

    x_ref = np.linalg.solve(A_np, b_np)

    err = np.linalg.norm(x_ref - x_np)
    print(f"最终与 numpy 直接解的 L2 误差 = {err:.6e}")

    print("部分解 x (Taichi):", np.round(x_np[:10], 6))
    print("部分解 x (numpy): ", np.round(x_ref[:10], 6))
