# ----- Projective Dynamics ----- #
import taichi as ti
from taichi import grouped

ti.init(arch=ti.cuda)  # Alternatively, ti.init(arch=ti.cpu)

# -------------------- Simulation parameters --------------------
n = 128
mass = 1.0
inv_m = 1.0 / mass
quad_size = 1.0 / n
dt = 4e-2 / n
inv_dt = 1 / dt
substeps = int(1 / 60 // dt)
gravity = ti.Vector([0, -9.8, 0])
spring_Y = 2e4
dashpot_damping = 1e4
drag_damping = 2
ball_radius = 0.3

# -------------------- Fields --------------------
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
x_pred = ti.Vector.field(3, dtype=float, shape=(n, n))
x_p = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
u = ti.Vector.field(3, dtype=float, shape=(n, n))
v_temp = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

b = ti.Vector.field(3, dtype=float, shape=(n, n))
x_vec = ti.Vector.field(3, dtype=float, shape=(n, n))
p_vec = ti.Vector.field(3, dtype=float, shape=(n, n))
r_vec = ti.Vector.field(3, dtype=float, shape=(n, n))
Ap_vec = ti.Vector.field(3, dtype=float, shape=(n, n))
r0 = ti.Vector.field(3, dtype=float, shape=(n, n))

# Scalars
re0 = ti.field(dtype=float, shape=(1,))
re1 = ti.field(dtype=float, shape=(1,))
row = spring_Y * dt * dt
primal_res = ti.Vector.field(1, dtype=float, shape=(1,))
primal_res_old = ti.Vector.field(1, dtype=float, shape=(1,))
Dual_res = ti.Vector.field(1, dtype=float, shape=(1,))

# -------------------- Initialization --------------------
@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0],
            0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        x_pred[i, j] = x[i, j]
        v[i, j] = ti.Vector([0.0,0.0,0.0])
        u[i, j] = ti.Vector([0.0,0.0,0.0])

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = i * (n - 1) + j
        # 1st triangle
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

# -------------------- Spring offsets --------------------
spring_offsets = []
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0):
            spring_offsets.append(ti.Vector([i, j]))

# -------------------- Update functions --------------------
@ti.kernel
def update_x_pred():
    for i in grouped(x):
        v[i] += gravity * dt
        x_pred[i] = x[i] + v[i] * dt

@ti.kernel
def update_v():
    for i in grouped(v):
        v[i] = (x_p[i] - x[i]) * inv_dt

@ti.kernel
def update_x():
    for i in grouped(x):
        x[i] = x_p[i]

# -------------------- Linear system helpers --------------------
@ti.kernel
def computeAp(p : ti.template()):
    # A = H + M / (dt*dt)
    # Ap = Mp + dt * dt * \sum_ (Ki * p)

    for i in grouped(Ap_vec):
        Ap_vec[i] = ti.Vector([0.0,0.0,0.0])
    I = ti.Matrix.identity(ti.f32,3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):

            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                Kii = I
                Kij = -I

                Ap_vec[i] += row * dt * dt * (Kii @ p[i] + Kij @ p[j])
                Ap_vec[j] += row * dt * dt * (Kij @ p[i] + Kii @ p[j])
        Ap_vec[i] += mass * p[i]

@ti.kernel
def compute_b(p : ti.template()):
    for i in ti.grouped(x):
        # x_pred[i] = x[i] + dt * v[i] + dt*dt * gravity
        b[i] = mass * x_pred[i]

    I = ti.Matrix.identity(ti.f32, 3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):
            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                Kii = I
                Kij = -I

                b[i] += row * dt*dt * (Kii @ (p[i]-u[i]) + Kij @ (p[j]-u[j]))
                b[j] += row * dt*dt * (Kij @ (p[i]-u[i]) + Kii @ (p[j]-u[j]))

@ti.kernel
def compute_r0(b: ti.template(), Ap: ti.template(), r: ti.template()):
    for I in ti.grouped(b):
        r[I] = b[I] - Ap[I]

@ti.kernel
def compute(pos : ti.template(),fro : ti.template()):
    for i in ti.grouped(fro):
        pos[i] = fro[i]


@ti.kernel
def dot_product(a: ti.template(), b: ti.template()) -> ti.f32:
    res = 0.0
    for I in ti.grouped(a):
        res += a[I].dot(b[I])
    return res

@ti.kernel
def axpy(x: ti.template(), alpha: ti.f32, p: ti.template()):
    for I in ti.grouped(x):
        x[I] += alpha * p[I]


@ti.kernel
def compute_norm(f: ti.template()) -> ti.f32:
    res = 0.0
    for I in ti.grouped(f):
        res += f[I].norm_sqr()  # 平方和
    return ti.sqrt(res)

@ti.kernel
def compute_add(pos : ti.template(),elem : ti.template()):
    for i in ti.grouped(pos):
        pos[i] += elem[i]


alpha = 0.1

# -------------------- Iteration --------------------
@ti.kernel
def iter_x_p():
    primal_res_old[0] = 0.0
    primal_res[0] = 0.0
    for i in grouped(x):
        temp = x_p[i]
        sum_n = ti.Vector([0.0,0.0,0.0])
        count = 0
        for off in ti.static(spring_offsets):
            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                # ADMM x-update: consider dual u
                sum_n += x_p[j] + u[i]
                count += 1
        x_p[i] = (x_pred[i] + alpha * sum_n) / (1.0 + alpha * count)
        # 更新残差平方和
        primal_res_old[0] += (x_p[i] - temp).norm_sqr()

@ti.kernel
def iter_u():
    Dual_res[0] = 0.0
    for i in ti.grouped(u):
        temp = u[i]
        for off in ti.static(spring_offsets):
            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                C_ij = x_p[i] - x_p[j]
                d = C_ij.normalized()
                proj_ij = d * quad_size * (i - j).norm()  # 投影后的长度向量
                u[i] += alpha * (C_ij - proj_ij)
        Dual_res[0] += (u[i] - temp).norm_sqr()


@ti.kernel
def test_u()->ti.f32:
    res = 0.0
    for i in ti.grouped(u):
        res += u[i].norm_sqr()
    return ti.sqrt(res)

@ti.kernel
def low_u(res : ti.f32):
    res_inv = 1.0 / res
    for i in ti.grouped(u):
        u[i] *= res_inv
# -------------------- Substep --------------------
def substep():
    update_x_pred()
    init_p()
    for _ in range(10):
        for _ in range(5):
            iter_x_p()  # 内部循环若需要，可多次
        iter_u()
        global_step()
        if test(): break
    res = test_u()
    if res > 10.0:
        low_u(res)
    coll_x()  # 先投影位置
    update_v()  # 再更新速度
    coll_v()  # 应对剩余穿透
    update_x()


# -------------------- Initialize x_p --------------------
@ti.kernel
def init_p():
    for i in ti.grouped(x):
        x_p[i] = x_pred[i]


# -------------------- Collision --------------------
@ti.kernel
def coll_v():
    damp = ti.exp(-drag_damping * dt)
    for i in ti.grouped(x):
        v[i] *= damp
        offset = x_p[i] - ball_center[0]
        off_norm = offset.norm()
        if off_norm <= ball_radius:
            normal = offset / off_norm
            v[i] -= min(v[i].dot(normal), 0) * normal


@ti.kernel
def coll_x():
    for i in ti.grouped(x):
        offset_to_center = x_p[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius :
            normal = offset_to_center.normalized()
            x_p[i] = normal * (ball_radius ) + ball_center[0]


# -------------------- Update vertices for rendering --------------------
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


# -------------------- Test convergence --------------------
@ti.kernel
def compute_res(f: ti.template()) -> ti.f32:
    res = 0.0
    for I in ti.grouped(f):
        res += f[I].norm_sqr()  # 平方和
    return res

def test():
    return (
        primal_res[0] / max(primal_res_old[0], 1e-12) < 1e-4
        and Dual_res[0] / max(compute_res(u), 1e-12) < 1e-4
    )


# -------------------- Global step (CG iteration) --------------------
def iterate_CG():
    compute_b(x_p)
    computeAp(x_p)
    compute_r0(b, Ap_vec, r0)
    compute(r_vec, r0)
    compute(p_vec, r_vec)
    re0 = dot_product(r_vec, r_vec)

    for _ in range(15):
        computeAp(p_vec)
        alpha = dot_product(r_vec, r_vec) / dot_product(Ap_vec, p_vec)
        axpy(x_p, alpha, p_vec)
        axpy(r_vec, -alpha, Ap_vec)
        if dot_product(r_vec, r_vec) / dot_product(r0, r0) < 1e-4:
            break
        re1 = dot_product(r_vec, r_vec)
        beta = re1 / re0
        re0 = re1
        axpy(p_vec, beta, p_vec)
        compute_add(p_vec, r_vec)


def global_step():
    iterate_CG()


# -------------------- Rendering loop --------------------
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 1.0:  # Reset
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()

        current_t += dt

    update_vertices()
    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
