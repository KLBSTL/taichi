import taichi as ti

ti.init(arch=ti.opengl)  # 可改为 ti.opengl / ti.cuda

# ================= 参数 =================
n = 32                     # 网格分辨率 n x n
quad_size = 1.0 / n
dt = 1e-2
substeps = 5

gravity = ti.Vector([0.0, -9.8, 0.0])
K_spring = 1e4
dashpot_damping = 1e3
drag_damping = 1.0

ball_radius = 0.2
ball_center = ti.Vector.field(3, dtype=ti.f32, shape=1)
ball_center[0] = [0.0, 0.0, 0.0]

# ================= 数据字段 =================
x = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))  # 位置
v = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))  # 速度
HK = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(n, n))  # 节点块对角刚度

vertices = ti.Vector.field(3, dtype=ti.f32, shape=n * n)
indices = ti.field(int, shape=(n-1)*(n-1)*6)
colors = ti.Vector.field(3, dtype=ti.f32, shape=n*n)

# 邻域偏移（保留二阶邻域）
spring_offsets = []
for ii in range(-2, 3):
    for jj in range(-2, 3):
        if (ii, jj) != (0, 0) and abs(ii)+abs(jj) <= 2:
            spring_offsets.append(ti.Vector([ii,jj]))

# ================= 初始化 =================
@ti.kernel
def init_mass_points():
    for i, j in x:
        x[i,j] = ti.Vector([i*quad_size-0.5, 0.6, j*quad_size-0.5])
        v[i,j] = ti.Vector([0.0, 0.0, 0.0])
        HK[i,j] = ti.Matrix.zero(ti.f32,3,3)

@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(n-1,n-1):
        idx = (i*(n-1)+j)*6
        indices[idx+0] = i*n + j
        indices[idx+1] = (i+1)*n + j
        indices[idx+2] = i*n + (j+1)
        indices[idx+3] = (i+1)*n + j+1
        indices[idx+4] = i*n + (j+1)
        indices[idx+5] = (i+1)*n + j
    for i,j in ti.ndrange(n,n):
        if (i//4 + j//4) %2==0:
            colors[i*n+j] = [0.22,0.72,0.52]
        else:
            colors[i*n+j] = [1.0,0.334,0.52]

# ================= 刚度矩阵 =================
@ti.kernel
def compute_H():
    # 清零
    for I in ti.grouped(HK):
        HK[I] = ti.Matrix.zero(ti.f32,3,3)
    I3 = ti.Matrix.identity(ti.f32,3)
    for i_r,i_c in ti.ndrange(n,n):
        xi = x[i_r,i_c]
        for off in ti.static(spring_offsets):
            j_r = i_r + int(off[0])
            j_c = i_c + int(off[1])
            if 0<=j_r<n and 0<=j_c<n:
                xj = x[j_r,j_c]
                x_ij = xi - xj
                dist = x_ij.norm()
                d = x_ij / dist
                rest_length = quad_size * ti.sqrt((i_r-j_r)**2 + (i_c-j_c)**2)
                t1 = 1.0 - rest_length / dist
                t2 = rest_length / dist
                H_temp = K_spring * (t1*I3 + t2*d.outer_product(d))
                HK[i_r,i_c] += H_temp
                HK[j_r,j_c] += H_temp

# ================= 隐式子步 =================
@ti.kernel
def substep():
    # 外力
    for i,j in ti.ndrange(n,n):
        v[i,j] += gravity * dt
    # 计算弹簧力 & 隐式更新速度
    for i,j in ti.ndrange(n,n):
        force = ti.Vector([0.0,0.0,0.0])
        xi = x[i,j]
        vi = v[i,j]
        for off in ti.static(spring_offsets):
            jr = i+int(off[0])
            jc = j+int(off[1])
            if 0<=jr<n and 0<=jc<n:
                xj = x[jr,jc]
                vj = v[jr,jc]
                x_ij = xi - xj
                v_ij = vi - vj
                dist = x_ij.norm()
                d = x_ij / dist
                rest_len = quad_size * ti.sqrt((i-jr)**2 + (j-jc)**2)
                # 弹簧
                force += -K_spring*d*(dist/rest_len -1)
                # 阻尼
                force += -v_ij.dot(d)*d*dashpot_damping*quad_size
        # 隐式速度更新: v_new = inv(I - dt^2*HK) * (v + dt*force)
        A = ti.Matrix.identity(ti.f32,3) - dt*dt*HK[i,j]
        b_loc = v[i,j] + dt*force
        v[i,j] = A.inverse() @ b_loc

    # 阻尼 + 碰撞 + 更新位置
    for i,j in ti.ndrange(n,n):
        v[i,j] *= ti.exp(-drag_damping*dt)
        offset = x[i,j]-ball_center[0]
        if offset.norm() <= ball_radius:
            nrm = offset.normalized()
            v[i,j] -= min(v[i,j].dot(nrm),0.0)*nrm
        x[i,j] += dt*v[i,j]

# ================= 更新顶点 =================
@ti.kernel
def update_vertices():
    for i,j in ti.ndrange(n,n):
        vertices[i*n+j] = x[i,j]

# ================= 主循环 =================
init_mass_points()
init_mesh()

window = ti.ui.Window("Implicit Cloth (Taichi)", (800,800))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

current_t = 0.0
while window.running:
    for _ in range(substeps):
        compute_H()  # Python scope 调用 kernel
        substep()
        current_t += dt

    update_vertices()
    camera.position(0.0,1.0,2.5)
    camera.lookat(0.0,0.3,0.0)
    scene.set_camera(camera)
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)

    canvas.scene(scene)
    window.show()
