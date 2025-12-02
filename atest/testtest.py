# ----- Projective Dynamics ----- #
import taichi as ti
from taichi import grouped
ti.init(arch=ti.cuda)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
mass = 1.0
inv_m = 1.0 / mass
quad_size = 1.0 / n
dt = 2e-2 / n
inv_dt = 1 / dt
substeps = int(1 / 60 // dt) // 40
gravity = ti.Vector([0, -9.8, 0])
spring_Y = 2e3
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
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0):
            spring_offsets.append(ti.Vector([i, j]))

rest_len_field = ti.field(dtype=ti.f32, shape=(n * n,))  # map flat_i*n + flat_j -> rest length




@ti.kernel
def update_x_pred():
    for i in grouped(x):
        v[i] += gravity * dt
        x_pred[i] = x[i] + v[i] * dt
        # v_temp[i] = ti.Vector([0.0,0.0,0.0])
    # for i in ti.grouped(x):
    #     force = ti.Vector([0.0, 0.0, 0.0])
    #     for spring_offset in ti.static(spring_offsets):
    #         j = i + spring_offset
    #         if 0 <= j[0] < n and 0 <= j[1] < n:
    #             x_ij = x[i] - x[j]
    #             v_ij = v[i] - v[j]
    #             d = x_ij.normalized()
    #             current_dist = x_ij.norm()
    #             original_dist = quad_size * float(i - j).norm()
    #             force += -spring_Y * d * (current_dist / original_dist - 1)
    #             force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        # v_temp[i] += force * dt
    # for i in grouped(v):
    #     v[i] += v_temp[i]
    #     x_pred[i] = x[i] + v[i] * dt


@ti.kernel
def update_v():
    for i in grouped(v):
        v[i] = (x_p[i] - x[i]) / dt

@ti.kernel
def update_x():
    for i in grouped(x):
        x[i] = x_p[i]

@ti.kernel
def computeAp(p : ti.template()):
    # A = H + M / (dt*dt)
    # Ap = Mp + dt * dt * \sum_ (Ki * p)

    for i in grouped(Ap_vec):
        Ap_vec[i] = ti.Vector([0.0,0.0,0.0])
    I = ti.Matrix.identity(ti.f32,3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):
            if off[0] < 0 or (off[0] == 0 and off[1] < 0):
                j = i + off
                if 0 <= j[0] < n and 0 <= j[1] < n:
                    Kii = spring_Y * I
                    Kij = -spring_Y * I

                    Ap_vec[i] += dt * dt * (Kii @ p[i] + Kij @ p[j])
                    Ap_vec[j] += dt * dt * (Kij @ p[i] + Kii @ p[j])
        Ap_vec[i] += mass * p[i]

@ti.kernel
def compute_b(p : ti.template()):
    for i in ti.grouped(b):
        # x_pred[i] = x[i] + dt * v[i] + dt*dt * gravity
        b[i] = mass * x_pred[i]

    I = ti.Matrix.identity(ti.f32, 3)
    for i in ti.grouped(x):
        for off in ti.static(spring_offsets):
            if off[0] < 0 or (off[0] == 0 and off[1] < 0):
                j = i + off
                if 0 <= j[0] < n and 0 <= j[1] < n:
                    Kii = spring_Y * I
                    Kij = -spring_Y * I

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
        res += a[I].dot(b[I])
    return res

@ti.kernel
def axpy(x: ti.template(), alpha: ti.f32, p: ti.template()):
    for I in ti.grouped(x):
        x[I] += alpha * p[I]

@ti.kernel
def axpy2(x: ti.template(), alpha: ti.f32, p: ti.template()):
    for I in ti.grouped(x):
        x[I] = alpha * p[I]

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

def iterate_CG():
    compute_b(x_p)
    computeAp(x_p)
    compute_r0(b,Ap_vec,r0)
    compute(r_vec,r0)
    compute(p_vec,r_vec)
    re0 = dot_product(r_vec,r_vec)
    it = 0
    for _ in range(100):
        it+=1
        computeAp(p_vec)
        denom = (dot_product(Ap_vec,p_vec))
        if denom < 1e-8:
            print("denom too small, stopping iter:", it, "denom=", denom)
            break
        alpha = dot_product(r_vec,r_vec) / denom
        axpy(x_p,alpha,p_vec)
        axpy(r_vec,-alpha,Ap_vec)

        if dot_product(r_vec,r_vec) / dot_product(r0,r0) < 1e-6:
            print("stopping iter:", it, "denom=", denom)
            break

        re1 = dot_product(r_vec,r_vec)

        beta = re1 / re0
        re0 = re1

        axpy2(p_vec,beta,p_vec)
        compute_add(p_vec,r_vec)

def global_step():
    iterate_CG()

@ti.kernel
def init_p():
    for i in ti.grouped(x):
        x_p[i] = x_pred[i]

@ti.kernel
def coll_v():
    damp = ti.exp(-drag_damping * dt)
    for i in ti.grouped(x):
        v[i] *= damp
        offset = x_p[i] - ball_center[0]
        off_norm = offset.norm()
        if off_norm <= ball_radius:
            normal = offset / off_norm
            # x_p[i] = normal * (ball_radius) + ball_center[0]
            v[i] -= min(v[i].dot(normal), 0) * normal
        # x_p[i] += dt * v[i]

@ti.kernel
def coll_x():
    for i in ti.grouped(x):
        offset_to_center = x_p[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius + 0.01:
            normal = offset_to_center.normalized()
            x_p[i] =  normal * (ball_radius + 0.01) + ball_center[0]

@ti.kernel
def iter():
    for i in grouped(x):
        for spring_offset in ti.static(spring_offsets):
            if spring_offset[0] < 0 or (spring_offset[0] == 0 and spring_offset[1] < 0):
                j = i + spring_offset
                if 0 <= j[0] < n and 0 <= j[1] < n:
                    x_ij = x_p[i] - x_p[j]
                    d = x_ij.normalized()
                    current_dist = x_ij.norm()
                    original_dist = quad_size * float(i - j).norm()

                    corre = 0.5 * (current_dist - original_dist) * d
                    x_p[i] -= corre
                    x_p[j] += corre

def substep():
    update_x_pred()
    init_p()
    # print("after init_p",sample())
    for _ in range(2):
        iter()
    print("after iter", compute_residual_norm(x_p))
    global_step()
    print("after global", compute_residual_norm(x_p))

    update_v()
    coll_v()

    # coll_x()
    # coll_x_and_fix_velocity()

    # update_v()


    update_x()

@ti.kernel
def compute_residual_norm(x_field: ti.template())->ti.f32:
    # compute Ap for current x_field into Ap_vec (reuse computeAp)
    # computeAp(x_field)      # writes Ap_vec
    s = 0.0
    for I in ti.grouped(b):
        r = b[I] - Ap_vec[I]
        s += r.dot(r)
    return s



@ti.kernel
def sample()->ti.f32:
    current_dist = 0.0
    for i in grouped(x):
        for spring_offset in ti.static(spring_offsets):
            # if spring_offset[0] < 0 or (spring_offset[0] == 0 and spring_offset[1] < 0):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x_p[i] - x_p[j]
                current_dist += x_ij.norm()
    return current_dist



@ti.kernel
def coll_x_and_fix_velocity():
    eps = 1e-9
    for I in ti.grouped(x):
        offset = x_p[I] - ball_center[0]
        dist = offset.norm()
        if dist <= ball_radius + 1e-6:
            # push to surface
            normal = ti.Vector([0.0, 1.0, 0.0])
            if dist > eps:
                normal = offset / dist

            x_p[I] = ball_center[0] + normal * (ball_radius + 1e-6)
            # set velocity consistent and remove inward component
            new_v = (x_p[I] - x[I]) * inv_dt
            vn = new_v.dot(normal)
            if vn < 0:
                new_v -= vn * normal
            v[I] = new_v


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
    if current_t > 0.7:
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