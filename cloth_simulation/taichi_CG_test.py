# ----- 隐式方法 ----- #

import taichi as ti

ti.init(arch=ti.opengl)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
quad_size = 1.0 / n
dt = 2.5e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
K = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

I = ti.Matrix.field(3,3,dtype=float,shape=())
HK = ti.Matrix.field(3,3,dtype=float,shape=(n,n))
A = ti.Matrix.field(3,3,dtype=float,shape=1)
b = ti.Vector.field(3,dtype=float,shape=1)

iter_dim = 3

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]
        HK[i, j] = ti.Matrix.zero(ti.f32, 3, 3)

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
def iter_H():
    for I in ti.grouped(HK):
        HK[I] = ti.Matrix.zero(ti.f32,3,3)
    I = ti.Matrix.identity(ti.f32, 3)
    for i_r,i_c in ti.ndrange(n,n):
        xi = x[i_r,i_c]
        for off in ti.static(spring_offsets):
            j_r = i_r + off[0]
            j_c = i_c + off[1]

            if 0 <= j_r < n and 0 <= j_c < n:
                xj = x[j_r,j_c]
                x_ij = xi - xj

                current_dist = x_ij.norm()
                d = x_ij / current_dist
                rest_length = quad_size * ti.sqrt((i_r - j_r) ** 2 + (i_c - j_c) ** 2)

                t1 = 1.0 - rest_length / current_dist
                t2 = rest_length / current_dist
                H_temp = K * (t1 * I + t2 * d.outer_product(d))


                HK[i_r,i_c] += H_temp
                HK[j_r,j_c] += H_temp
                HK[i_r,j_c] += -H_temp
                HK[j_r,i_c] += -H_temp


@ti.func
def iterate_CG(A,b,x):
    r0 = b - A @ x
    p0 = r0
    k = 0
    r_f = r0
    for _ in range(50):
        k+=1

        alpha = r0.dot(r0) / (p0.dot(A @ p0))
        x += alpha * p0
        r1 = r0 - alpha * (A @ p0)

        if r1.norm() / r_f.norm() < 1e-5:
            break

        beta = r1.dot(r1) / (r0.dot(r0))
        p1 = r1 + beta * p0
        p0 = p1
        r0 = r1

    return x,k

@ti.kernel
def substep()->ti.f32:
    k = 0
    I = ti.Matrix.identity(ti.f32,3)
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                rest_length = quad_size * float(i - j).norm()

                # Spring force
                force += -K * d * (current_dist / rest_length - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size


        A = I - dt * dt * HK[i]
        b = v[i] + dt * force
        # v[i] = A.inverse() @ b
        v[i],k = iterate_CG(A, b, v[i])
        # v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]

    return k


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

def update():
    iter_H()
    update_vertices()

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
        k = substep()
        print(k)
        current_t += dt
    update()

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