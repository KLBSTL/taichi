# ----- MLP + PBD ----- #

import argparse
import math
import numpy as np
import os
import taichi as ti
from matplotlib import pyplot as plt
from taichi import grouped

real = ti.f32
ti.init(default_fp=real,arch=ti.cuda)


n = 128
mass = 1.0
inv_m = 1.0 / mass
quad_size = 1.0 / n
dt = 8e-2 / n
inv_dt = 1 / dt
substeps = int(1 / 60 // dt)
gravity = ti.Vector([0, -9.8, 0])
spring_Y = 1e3
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
x_pred = ti.Vector.field(3,dtype=float,shape=(n,n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
v_temp = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

iter_num = 400

max_steps = 500

scalar = lambda: ti.field(dtype=real)
vec = lambda : ti.Vector.field(2,dtype=real)

loss = scalar()
goal = vec()

weights1 = scalar()
bias1 = scalar()

n_hidden = 32
learning_rate = 0.8

weights2 = scalar()
bias2 = scalar()

hidden = scalar()
wind = scalar()

input_dim = 4

T = 5.5

wind_scale = 35.0

# target_wind_x =



def allocate_fields():
    ti.root.dense(ti.ij,(n_hidden,input_dim)).place(weights1)

    ti.root.dense(ti.i,n_hidden).place(bias1)

    ti.root.dense(ti.ij,(3,n_hidden)).place(weights2)

    ti.root.dense(ti.i,3).place(bias2)

    ti.root.dense(ti.ij,(max_steps,n_hidden)).place(hidden)

    ti.root.dense(ti.ij,(max_steps,3)).place(wind)

    ti.root.place(loss,goal)

    ti.root.lazy_grad()


@ti.kernel
def nn1(t: ti.i32,time: ti.f32):
    for i in range(n_hidden):
        actuation = 0.0

        actuation += weights1[i, 0] * ti.sin(2.0 * math.pi * time / T)

        actuation += weights1[i, 1] * ti.cos(2.0 * math.pi * time / T)

        actuation += weights1[i, 2] * ti.sin(4.0 * math.pi * time / T)

        actuation += weights1[i, 3] * ti.cos(4.0 * math.pi * time / T)

        actuation += bias1[i]
        actuation = ti.tanh(actuation)

        hidden[t, i] = actuation

@ti.kernel
def nn2(t: ti.i32):
    for i in range(3):
        actuation = 0.0

        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        wind[t, i] = wind_scale * actuation


@ti.kernel
def compute_loss():
    for t in range(max_steps):
        time = t / max_steps
        loss1 = wind[t, 0] - 8 * ti.sin(2 * math.pi * time / T)
        loss2 = wind[t, 1] - 0
        loss3 = wind[t, 2] - 4 * ti.cos(2 * math.pi * time / T)

        loss[None] += (loss1 * loss1 + loss2 * loss2 + loss3 * loss3) / max_steps


def forward():
    loss[None] = 0.0
    for t in range(max_steps):
        time = float(t) / max_steps
        nn1(t,time)
        nn2(t)


    compute_loss()


@ti.kernel
def clear():
    pass

def optimize(print_interval=1):
    global learning_rate
    for i in range(n_hidden):
        for j in range(input_dim):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + input_dim)) * 2

    for i in range(3):
        for j in range(n_hidden):
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + 3)) * 3

    losses = []
    for iter in range(iter_num):
        clear()
        # with ti.ad.Tape(loss) automatically clears all gradients
        with ti.ad.Tape(loss):
            forward()

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(input_dim):
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(3):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2

        grad_norm = total_norm_sqr**0.5
        should_print = (
            iter == 0 or
            iter == iter_num - 1 or
            (print_interval > 0 and (iter + 1) % print_interval == 0)
        )
        if should_print:
            print(
                f"Iter {iter + 1:03d}/{iter_num}: "
                f"loss={loss[None]:.6f}, grad_norm={grad_norm:.6f}",
                flush=True
            )

        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        gradient_clip = 0.2
        # scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        scale = learning_rate * min(1.0, gradient_clip / (total_norm_sqr**0.5 + 1e-6))
        for i in range(n_hidden):
            for j in range(input_dim):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(3):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

        if iter % 100 == 0:
            learning_rate *= 0.5

    for t in range(5):
        t *= 90
        time = t / max_steps
        print(f"pred {wind[t, 0]}, {wind[t, 1]}, {wind[t, 2]}", flush=True)
        print(
            f"target {8 * ti.sin(2 * math.pi * time / T)},"
            f"{0},{4 * ti.cos(2 * math.pi * time / T)}",
            flush=True
        )

    return losses


def save_weights(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    np.savez(
        path,
        weights1=weights1.to_numpy(),
        bias1=bias1.to_numpy(),
        weights2=weights2.to_numpy(),
        bias2=bias2.to_numpy()
    )
    print(f"Weights saved to {path}", flush=True)


def load_weights(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find weights file: {path}. Run train first or pass --weights."
        )

    with np.load(path) as data:
        weights1.from_numpy(data["weights1"])
        bias1.from_numpy(data["bias1"])
        weights2.from_numpy(data["weights2"])
        bias2.from_numpy(data["bias2"])
    print(f"Weights loaded from {path}", flush=True)


def plot_losses(losses, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses, color="tab:green", linewidth=2, label="training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Wind Network Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    print(
        f"Loss summary: first={losses[0]:.6f}, "
        f"last={losses[-1]:.6f}, min={min(losses):.6f}, "
        f"plot={path}",
        flush=True
    )


@ti.func
def is_fixed_point(i, j):
    return j == 0


@ti.func
def fixed_position(i):
    return ti.Vector([i * quad_size - 0.5, 1.2, -0.45])


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.02

    for i, j in x:
        if is_fixed_point(i, j):
            x[i, j] = fixed_position(i)
        else:
            x[i, j] = [
                i * quad_size - 0.5 + random_offset[0],
                1.2 - j * quad_size,
                -0.45 + random_offset[1]
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

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    spring_offsets.append(ti.Vector([1, 0]))
    spring_offsets.append(ti.Vector([0, 1]))
    spring_offsets.append(ti.Vector([1, 1]))
    spring_offsets.append(ti.Vector([1, -1]))
    spring_offsets.append(ti.Vector([2, 0]))
    spring_offsets.append(ti.Vector([0, 2]))

@ti.kernel
def update_x_pred(wind_id: ti.i32):
    for i in grouped(x):
        if is_fixed_point(i[0], i[1]):
            x[i] = fixed_position(i[0])
            x_pred[i] = fixed_position(i[0])
            v[i] = [0.0, 0.0, 0.0]
        else:
            wind_force = ti.Vector([
                wind[wind_id, 0],
                0.0,
                wind[wind_id, 2],
            ])
            v[i] += (gravity + wind_force) * dt
            x_pred[i] = x[i] + v[i] * dt


@ti.kernel
def enforce_fixed_points():
    for i, j in x:
        if is_fixed_point(i, j):
            x[i, j] = fixed_position(i)
            x_pred[i, j] = fixed_position(i)
            v[i, j] = [0.0, 0.0, 0.0]

@ti.kernel
def update_v():
    for i in grouped(v):
        v[i] = (x_pred[i] - x[i]) * inv_dt

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

                dx = (1.0 / 2) * (current_dist - original_dist) * d
                ti.atomic_add(x_pred[i][0], -dx[0])
                ti.atomic_add(x_pred[i][1], -dx[1])
                ti.atomic_add(x_pred[i][2], -dx[2])

                ti.atomic_add(x_pred[j][0], dx[0])
                ti.atomic_add(x_pred[j][1], dx[1])
                ti.atomic_add(x_pred[j][2], dx[2])



@ti.kernel
def coll_x():
    for i in ti.grouped(x):
        offset_to_center = x_pred[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            x_pred[i] =  normal * (ball_radius) + ball_center[0]


@ti.kernel
def coll_v():
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        # Ball velocity projection disabled for the fixed-flag scene.
        # offset_to_center = x_pred[i] - ball_center[0]
        # if offset_to_center.norm() <= ball_radius:
        #     normal = offset_to_center.normalized()
        #     if normal.norm() > 0.1:
        #         v[i] -= min(v[i].dot(normal), 0) * normal
        # x_pred[i] = x[i] + dt * v[i]

@ti.kernel
def update_x():
    for i in grouped(x):
        x[i] = x_pred[i]

def substep(wind_id: ti.i32):
    update_x_pred(wind_id)
    for _ in range(10):
        iter()
        enforce_fixed_points()
    # Sphere collision disabled for the fixed-flag scene.
    # coll_x()
    update_v()
    coll_v()
    update_x()

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

def run_visualization():
    window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                          vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    current_t = 0.0
    current_step = 0
    initialize_mass_points()

    while window.running:
        if current_t > 4.0:
            # Reset
            initialize_mass_points()
            current_t = 0
            current_step = 0

        for i in range(substeps):
            substep(current_step % max_steps)
            current_step += 1
            current_t += dt
        update_vertices()

        camera.position(1.85, 0.75, 1.9)
        camera.lookat(0.0, 0.45, -0.2)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.mesh(vertices,
                   indices=indices,
                   per_vertex_color=colors,
                   two_sided=True)

        # Ball display disabled for the fixed-flag scene.
        # scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)
        window.show()


def main():
    global iter_num

    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="train",
                        choices=["train", "test"], help="train/test")
    parser.add_argument("--iters", type=int, default=iter_num)
    parser.add_argument("--print-interval", type=int, default=1,
                        help="print training progress every N iterations")
    parser.add_argument("--weights", type=str,
                        default="taichi_pbd_wind_weights.npz")
    parser.add_argument("--loss-plot", type=str,
                        default="taichi_pbd_loss.png")
    parser.add_argument("--no-visualize", action="store_true")
    options = parser.parse_args()

    iter_num = options.iters
    allocate_fields()
    initialize_mesh_indices()

    if options.task == "train":
        print(
            f"Start training: iters={iter_num}, "
            f"print_interval={options.print_interval}",
            flush=True
        )
        losses = optimize(print_interval=options.print_interval)
        save_weights(options.weights)
        plot_losses(losses, options.loss_plot)
        forward()
    else:
        load_weights(options.weights)
        forward()
        print(f"Loaded wind loss={loss[None]:.6f}", flush=True)

    if not options.no_visualize:
        run_visualization()


if __name__ == "__main__":
    main()



"""
conda run -n DL --no-capture-output python taichi_PBD.py train
conda run -n DL python taichi_PBD.py test
"""






















