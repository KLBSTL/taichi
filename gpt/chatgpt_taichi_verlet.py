import taichi as ti
ti.init(arch=ti.opengl)  # 或 ti.cpu

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

mass = 1.0

gravity = ti.Vector([0.0, -9.8, 0.0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1.0

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))

# position, velocity (kept for compatibility), previous position for Verlet
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
x_last = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    # initialize ball center on device to avoid sync issues
    ball_center[0] = ti.Vector([0.0, 0.0, 0.0])

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        # set previous position so initial velocity is zero
        x_last[i, j] = x[i, j]
        v[i, j] = [0.0, 0.0, 0.0]


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
            colors[i * n + j] = (1.0, 0.334, 0.52)

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
def substep():
    # For each mass point: compute force using current positions and estimated velocities
    # then update positions with Verlet, update velocities as position differences
    damping = 0.995  # positional damping (keeps energy from blowing up)
    for I in ti.grouped(x):
        # compute spring + dashpot forces
        force = ti.Vector([0.0, 0.0, 0.0])
        # estimate current velocity from previous step
        vel_i_prev = (x[I] - x_last[I]) / dt

        for off in ti.static(spring_offsets):
            J = I + off
            if 0 <= J[0] < n and 0 <= J[1] < n:
                x_ij = x[I] - x[J]
                # estimate neighbor velocity from previous step
                vel_j_prev = (x[J] - x_last[J]) / dt
                v_ij_est = vel_i_prev - vel_j_prev

                # spring force
                if x_ij.norm() > 1e-12:
                    d = x_ij.normalized()
                    current_dist = x_ij.norm()
                    original_dist = quad_size * float(I - J).norm()
                    force += -spring_Y * d * (current_dist / original_dist - 1)
                    # dashpot using estimated relative velocity
                    force += -v_ij_est.dot(d) * d * dashpot_damping * quad_size

        # gravity (as force)
        force += gravity * mass

        # Verlet position update
        x_old = x[I]
        x_new = x[I] + (x[I] - x_last[I]) * damping + (force / mass) * dt * dt

        # collision with sphere: project out and correct velocity later
        offset_to_center = x_new - ball_center[0]
        dist = offset_to_center.norm()
        if dist <= ball_radius:
            # project to surface
            normal = offset_to_center / dist
            if dist > 1e-12:
                normal = offset_to_center / dist
            else:
                normal = ti.Vector([0.0, 1.0, 0.0])
            x_new = normal * ball_radius + ball_center[0]

        # set new position and estimate new velocity
        # new velocity estimated from x_new and x_old (consistent with Verlet)
        v_new = (x_new - x_old) / dt
        # apply drag to velocity
        v_new *= ti.exp(-drag_damping * dt)

        # if collided, remove inward normal component from velocity
        offset_to_center_after = x_new - ball_center[0]
        dist_after = offset_to_center_after.norm()
        if dist_after <= ball_radius + 1e-12:
            normal2 = ti.Vector([0.0,1.0,0.0])
            if dist_after > 1e-12:
                normal2 = offset_to_center_after / dist_after
            else:
                normal2 = ti.Vector([0.0, 1.0, 0.0])
            # remove inward component
            v_n = v_new.dot(normal2)
            if v_n < 0.0:
                v_new = v_new - v_n * normal2

        # finalize writes
        x_last[I] = x_old
        x[I] = x_new
        v[I] = v_new

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation (Verlet)", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 4.5:
        # Reset
        initialize_mass_points()
        current_t = 0.0

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
