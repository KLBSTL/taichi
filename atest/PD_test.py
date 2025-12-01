import math
import taichi as ti

ti.init(arch=ti.opengl)

# ---------------------
# Simulation parameters
# ---------------------
n = 128
quad_size = 1.0 / n
mass = 1.0
dt = 4e-2 / n
inv_dt = 1.0 / dt
substeps = max(1, int(1 / 60 // dt) // 3)
gravity = ti.Vector([0.0, -9.8, 0.0])
spring_Y = 3e3
drag_damping = 1.0
pd_iterations = 5
cg_max_iters = 200
cg_tolerance = 1e-6

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
ball_center[0] = [0.0, 0.0, 0.0]

# ---------------------
# Simulation state
# ---------------------
x = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
x_rest = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
v = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
y = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
x_pred = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
x_p = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
rhs = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
cg_r = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
cg_p = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
cg_Ap = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))

# ---------------------
# Rendering helpers
# ---------------------
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=ti.f32, shape=n * n)
colors = ti.Vector.field(3, dtype=ti.f32, shape=n * n)


# ---------------------
# Spring connectivity
# ---------------------
bending_springs = False
spring_offsets = []
if bending_springs:
    span = range(-1, 2)
else:
    span = range(-2, 3)

for di in span:
    for dj in span:
        if (di, dj) != (0, 0):
            if bending_springs or abs(di) + abs(dj) <= 2:
                spring_offsets.append((di, dj))

spring_pairs = []
for i in range(n):
    for j in range(n):
        for off in spring_offsets:
            ni, nj = i + off[0], j + off[1]
            if 0 <= ni < n and 0 <= nj < n:
                if ni > i or (ni == i and nj > j):
                    spring_pairs.append(((i, j), (ni, nj)))

num_springs = len(spring_pairs)
spring_i = ti.Vector.field(2, dtype=ti.i32, shape=num_springs)
spring_j = ti.Vector.field(2, dtype=ti.i32, shape=num_springs)
spring_rest = ti.field(dtype=ti.f32, shape=num_springs)

for idx, (a, b) in enumerate(spring_pairs):
    spring_i[idx] = ti.Vector([a[0], a[1]])
    spring_j[idx] = ti.Vector([b[0], b[1]])
    rest = quad_size * math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    spring_rest[idx] = rest


@ti.func
def is_fixed(i, j):
    return (i == 0 and (j == 0 or j == n - 1))


# ---------------------
# Initialization
# ---------------------
@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.05
    for i, j in x:
        pos = ti.Vector([
            i * quad_size - 0.5 + random_offset[0],
            0.6,
            j * quad_size - 0.5 + random_offset[1],
        ])
        x[i, j] = pos
        x_rest[i, j] = pos
        v[i, j] = ti.Vector([0.0, 0.0, 0.0])
        y[i, j] = pos
        x_pred[i, j] = pos


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = i * (n - 1) + j
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1.0, 0.334, 0.52)


# ---------------------
# PD local/global steps
# ---------------------
@ti.kernel
def compute_y():
    for i, j in ti.ndrange(n, n):
        v[i, j] += gravity * dt
        y[i, j] = x[i, j] + dt * v[i, j]
        x_pred[i, j] = y[i, j]


@ti.kernel
def local_step():
    for I in ti.grouped(x_p):
        x_p[I] = ti.Vector([0.0, 0.0, 0.0])

    for s in range(num_springs):
        ia = spring_i[s]
        ja = spring_j[s]
        i0, i1 = ia[0], ia[1]
        j0, j1 = ja[0], ja[1]
        xi = x_pred[i0, i1]
        xj = x_pred[j0, j1]
        rest = spring_rest[s]
        dir_vec = xi - xj
        length = dir_vec.norm()
        if length > 1e-6:
            dir_vec /= length
        else:
            dir_vec = ti.Vector([0.0, 0.0, 0.0])
        center = 0.5 * (xi + xj)
        pi = center + 0.5 * rest * dir_vec
        pj = center - 0.5 * rest * dir_vec
        x_p[i0, i1] += spring_Y * pi
        x_p[j0, j1] += spring_Y * pj


@ti.kernel
def build_rhs():
    for I in ti.grouped(rhs):
        rhs[I] = mass * y[I] + dt * dt * x_p[I]


@ti.kernel
def apply_A(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(dst):
        dst[I] = mass * src[I]

    for s in range(num_springs):
        ia = spring_i[s]
        ja = spring_j[s]
        i0, i1 = ia[0], ia[1]
        j0, j1 = ja[0], ja[1]
        diff = src[i0, i1] - src[j0, j1]
        dst[i0, i1] += dt * dt * spring_Y * diff
        dst[j0, j1] -= dt * dt * spring_Y * diff


@ti.kernel
def init_residual():
    for I in ti.grouped(cg_r):
        cg_r[I] = rhs[I] - cg_Ap[I]
        cg_p[I] = cg_r[I]


@ti.kernel
def cg_update(alpha: ti.f32):
    for I in ti.grouped(x_pred):
        x_pred[I] += alpha * cg_p[I]
        cg_r[I] -= alpha * cg_Ap[I]


@ti.kernel
def cg_update_direction(beta: ti.f32):
    for I in ti.grouped(cg_p):
        cg_p[I] = cg_r[I] + beta * cg_p[I]


@ti.kernel
def enforce_pins(field: ti.template()):
    for i, j in ti.ndrange(n, n):
        if is_fixed(i, j):
            field[i, j] = x_rest[i, j]


@ti.kernel
def collide_pred():
    for i, j in ti.ndrange(n, n):
        if not is_fixed(i, j):
            offset = x_pred[i, j] - ball_center[0]
            dist = offset.norm()
            if dist < ball_radius:
                if dist > 1e-6:
                    normal = offset / dist
                else:
                    normal = ti.Vector([0.0, 1.0, 0.0])
                x_pred[i, j] = ball_center[0] + normal * (ball_radius + 1e-3)


@ti.kernel
def update_states():
    decay = ti.exp(-drag_damping * dt)
    for i, j in ti.ndrange(n, n):
        if is_fixed(i, j):
            x[i, j] = x_rest[i, j]
            v[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            new_pos = x_pred[i, j]
            v[i, j] = (new_pos - x[i, j]) * inv_dt
            v[i, j] *= decay
            x[i, j] = new_pos


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


@ti.kernel
def dot_kernel(a: ti.template(), b: ti.template()) -> ti.f32:
    acc = 0.0
    for I in ti.grouped(a):
        acc += a[I].dot(b[I])
    return acc


def cg_solve():
    apply_A(x_pred, cg_Ap)
    init_residual()
    rTr = dot_kernel(cg_r, cg_r)
    if rTr < cg_tolerance:
        return
    for _ in range(cg_max_iters):
        apply_A(cg_p, cg_Ap)
        pAp = dot_kernel(cg_p, cg_Ap)
        if abs(pAp) < 1e-12:
            break
        alpha = rTr / pAp
        cg_update(alpha)
        new_rTr = dot_kernel(cg_r, cg_r)
        if new_rTr < cg_tolerance:
            break
        beta = new_rTr / (rTr + 1e-12)
        cg_update_direction(beta)
        rTr = new_rTr


def substep():
    compute_y()
    enforce_pins(x_pred)
    enforce_pins(y)
    for _ in range(pd_iterations):
        local_step()
        build_rhs()
        cg_solve()
        collide_pred()
        enforce_pins(x_pred)
    update_states()


# ---------------------
# Rendering loop
# ---------------------
initialize_mesh_indices()
initialize_mass_points()

window = ti.ui.Window("Projective Dynamics Cloth", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

while window.running:
    if current_t > 1.5:
        initialize_mass_points()
        current_t = 0.0

    for _ in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(
        vertices,
        indices=indices,
        per_vertex_color=colors,
        two_sided=True,
    )
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()


