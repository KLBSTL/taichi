# ----- PBD ----- #
import taichi as ti
from taichi import grouped

ti.init(arch=ti.cuda)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
mass = 1.0
inv_m = 1.0 / mass
quad_size = 1.0 / n
dt = 4e-2 / n
inv_dt = 1 / dt
substeps = int(1 / 60 // dt) // 10
gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e3
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
x_pred = ti.Vector.field(3,dtype=float,shape=(n,n))
x_p = ti.Vector.field(3,dtype=float,shape=(n,n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

b = ti.Vector.field(3,dtype=float,shape=(n,n))
x_vec = ti.Vector.field(3,dtype=float,shape=(n,n))
p_vec = ti.Vector.field(3,dtype=float,shape=(n,n))
r_vec = ti.Vector.field(3,dtype=float,shape=(n,n))
Ap_vec = ti.Vector.field(3,dtype=float,shape=(n,n))
r0 = ti.Vector.field(3,dtype=float,shape=(n,n))
re0 = ti.Vector.field(3,dtype=float,shape=(1))
re1 = ti.Vector.field(3,dtype=float,shape=(1))


v_temp = ti.Vector.field(3, dtype=float, shape=(n, n))

bending_springs = False

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        x_pred[i, j] = x[i, j]
        v[i, j] = [0, 0, 0]



@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))



@ti.kernel
def update_x_pred():
    for i in grouped(v):
        v[i] += gravity * dt
        v_temp[i] = ti.Vector([0.0,0.0,0.0])
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                force += -spring_Y * d * (current_dist / original_dist - 1)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v_temp[i] += force * dt
    for i in grouped(v):
        v[i] += v_temp[i]
        x_pred[i] = x[i] + v[i] * dt


@ti.kernel
def update_v():
    for i in grouped(v):
        v[i] = (x_p[i] - x[i]) * inv_dt

@ti.kernel
def update_x():
    for i in grouped(x):
        x[i] = x_p[i]


@ti.kernel
def computeAp(p : ti.template()):
    # A = H + M / (dt*dt)
    # Ap = Mp + dt * dt * sigma(Ki * p)

    for i in grouped(Ap_vec):
        Ap_vec[i] = ti.Vector([0.0,0.0,0.0])
    I = ti.Matrix.identity(ti.f32,3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):

            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                Kii = spring_Y * I
                Kij = -spring_Y * I

                # Apply to both points
                Ap_vec[i] += dt * dt * (Kii @ p[i] + Kij @ p[j])
                Ap_vec[j] += dt * dt * (Kij @ p[i] + Kii @ p[j])
        Ap_vec[i] += mass * p[i]

@ti.kernel
def compute_b(p : ti.template()):
    for i in ti.grouped(x):
        x_pred[i] = x[i] + dt * v[i] + dt*dt * gravity
        b[i] = mass * x_pred[i]

    I = ti.Matrix.identity(ti.f32, 3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):
            j = i + off
            if 0 <= j[0] < n and 0 <= j[1] < n:
                Kii = spring_Y * I
                Kij = -spring_Y * I

                # dt^2 * H*p 的贡献
                b[i] += dt*dt * (Kii @ p[i] + Kij @ p[j])
                b[j] += dt*dt * (Kij @ p[i] + Kii @ p[j])

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
        res += a[I].dot(b[I])  # a[I] 和 b[I] 是单个向量/矩阵，可以用 dot
    return res

@ti.kernel
def axpy(x: ti.template(), alpha: ti.f32, p: ti.template()):
    for I in ti.grouped(x):
        x[I] += alpha * p[I]


@ti.kernel
def compute_norm(f: ti.template()) -> ti.f32:
    res = 0.0
    for I in ti.grouped(f):
        res += f[I].norm_sqr()  # 每个向量/矩阵的平方和
    return ti.sqrt(res)

@ti.kernel
def compute_add(pos : ti.template(),elem : ti.template()):
    for i in ti.grouped(pos):
        pos[i] += elem[i]


@ti.kernel
def up_x():
    for i in ti.grouped(x):
        x[i] = x_p[i]

def iterate_CG():
    compute_b(x_p)
    computeAp(x_p)
    compute_r0(b,Ap_vec,r0)
    compute(r_vec,r0)
    compute(p_vec,r_vec)
    re0 = dot_product(r_vec,r_vec)
    k = 0
    for _ in range(10):
        k += 1
        computeAp(p_vec)
        alpha = dot_product(r_vec,r_vec) / (dot_product(Ap_vec,p_vec))
        axpy(x_p,alpha,p_vec)
        # computeAp(p_vec)
        axpy(r_vec,-alpha,Ap_vec)

        if dot_product(r_vec,r_vec) / dot_product(r0,r0) < 1e-5 or k > 10:
            break

        re1 = dot_product(r_vec,r_vec)

        beta = re1 / re0
        re0 = re1

        axpy(p_vec,beta,p_vec)
        compute_add(p_vec,r_vec)




def global_step():
    iterate_CG()


@ti.kernel
def init_p():
    for i in ti.grouped(x):
        x_p[i] = x_pred[i]


@ti.kernel
def iter():
    for i in grouped(x):
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x_pred[i] - x_pred[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()

                corre = 0.5 * (current_dist - original_dist) * d
                x_p[i] -= corre
                x_p[j] += corre

                x_pred[i] = x_p[i]
                x_pred[j] = x_p[j]

def substep():
    update_x_pred()
    init_p()
    for _ in range(5):
        iter()
    # coll_v()
    global_step()
    update_v()
    update_x()

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 1.5:
        # Reset
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
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()