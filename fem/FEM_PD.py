import taichi as ti

ti.init(arch=ti.gpu)

N = 12
dt = 1e-4
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.31
damping = 14.5
mass = 1.0

pos = ti.Vector.field(2, float, NV)
y = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
R = ti.Matrix.field(2, 2, float, NF)
V = ti.field(float, NF)
area = ti.field(float,NF)
weight = ti.field(float,NF)

gravity = ti.Vector.field(2, float, ())
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())

b = ti.Vector.field(2,dtype=float,shape=(NV))
x_vec = ti.Vector.field(2,dtype=float,shape=(NV))
p_vec = ti.Vector.field(2,dtype=float,shape=(NV))
r_vec = ti.Vector.field(2,dtype=float,shape=(NV))
Ap_vec = ti.Vector.field(2,dtype=float,shape=(NV))
r0 = ti.Vector.field(2,dtype=float,shape=(NV))
re0 = ti.Vector.field(2,dtype=float,shape=(1))
re1 = ti.Vector.field(2,dtype=float,shape=(1))

T = ti.Matrix([  [-1,-1 ],
                 [ 1, 0 ],
                 [ 0, 1 ]  ])

fix = ti.field(ti.i32,NV)

@ti.kernel
def compute_F():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]


@ti.kernel
def init_pos():
    fix.fill(1)
    fix[12] = 0
    fix[NV-1] = 0
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        vel[k] = ti.Vector([0, 0])
        y[k] = pos[k]
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()
        area[i] = ti.abs(B_i_inv.determinant()) * 0.5
        weight[i] = area[i] * mu * 2


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


def paint_phi(gui):
    pos_ = pos.to_numpy()
    f2v_ = f2v.to_numpy()
    a, b, c = pos_[f2v_[:, 0]], pos_[f2v_[:, 1]], pos_[f2v_[:, 2]]
    gui.triangles(a, b, c, color=0x956333)

@ti.kernel
def predict():
    for i in ti.grouped(pos):
        vel[i] += gravity[None] * dt
        y[i] = pos[i] + vel[i] * dt


@ti.kernel
def compute_SVD():
    for temp in ti.grouped(F):
        S = F[temp].transpose() @ F[temp]

        s00 = S[0, 0]
        s01 = S[0, 1]
        s11 = S[1, 1]

        tau = s00 + s11
        delta = s00 - s11
        r = ti.sqrt(delta * delta + 4.0 * s01 * s01)

        lambda1 = 0.5 * (tau + r)
        lambda2 = 0.5 * (tau - r)

        sigma1 = ti.sqrt(lambda1)
        sigma2 = ti.sqrt(lambda2)

        v1 = ti.Vector([1.0, 0.0])
        if ti.abs(s01) > 1e-8:
            v1 = ti.Vector([lambda1 - s11, s01])
            v1 = v1.normalized()


        v2 = ti.Vector([-v1[1], v1[0]])

        V = ti.Matrix.cols([v1, v2])

        inv_sigma1 = 1.0 / sigma1 if sigma1 > 1e-8 else 0.0
        inv_sigma2 = 1.0 / sigma2 if sigma2 > 1e-8 else 0.0

        U = ti.Matrix.cols([
            F[temp] @ v1 * inv_sigma1,
            F[temp] @ v2 * inv_sigma2
        ])

        R[temp] = U @ V.transpose()

        if R[temp].determinant() < 0:
            U[0, 1] *= -1
            U[1, 1] *= -1

        R[temp] = U @ V.transpose()

def local_step_pd():
    compute_F()
    compute_SVD()

def global_step():
    init_x_vec()
    iterate_CG()

@ti.kernel
def init_x_vec():
    for i in ti.grouped(y):
        x_vec[i] = y[i]

@ti.kernel
def update_vx():
    for i in ti.grouped(vel):
        vel[i] = (x_vec[i] - pos[i]) / dt * ti.exp(-dt * damping)
        pos[i] = x_vec[i]
    vel[12] *= 0
    vel[NV-1] *= 0

def coll():
    for i in range(NV):
        # ball boundary condition:
        disp = pos[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius**2:
            NoV = vel[i].dot(disp)
            if NoV < 0:
                vel[i] -= NoV * disp / disp2
        cond = (pos[i] < 0) & (vel[i] < 0) | (pos[i] > 1) & (vel[i] > 0)
        # rect boundary condition:
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0

pd_iterations = 2

def substep():
    predict()
    for _ in range(pd_iterations):
        local_step_pd()
        project_fix()
        global_step()
    update_vx()
    # coll()

@ti.kernel
def project_fix():
    for i in ti.grouped(x_vec):
        if fix[i] == 0:
            x_vec[i] = pos[i]

@ti.kernel
def computeAp(p : ti.template()):
    for i in ti.grouped(Ap_vec):
        Ap_vec[i] = mass * p[i]
    for i in ti.ndrange(NF):
        ia,ib,ic = f2v[i]

        if fix[ia] and fix[ib] and fix[ic]:

            w = dt * dt * weight[i]

            pa,pb,pc = p[ia],p[ib],p[ic]

            Ds = ti.Matrix.cols([pa-pc,pb-pc])
            BTB = B[i] @ B[i].transpose()
            Res = Ds @ BTB

            f_a = ti.Vector([Res[0,0],Res[1,0]])
            f_b = ti.Vector([Res[0,1], Res[1, 1]])
            f_c = -f_a - f_b

            Ap_vec[ia] += w * f_a
            Ap_vec[ib] += w * f_b
            Ap_vec[ic] += w * f_c



@ti.kernel
def compute_b():
    for i in ti.grouped(b):
        b[i] = mass * y[i]

    for i in ti.ndrange(NF):
        ia,ib,ic = f2v[i]

        if fix[ia] and fix[ib] and fix[ic]:

            w = dt * dt * weight[i]

            G = R[i] @ B[i].transpose()
            g_a = ti.Vector([G[0,0],G[1,0]])
            g_b = ti.Vector([G[0,1], G[1,1]])
            g_c = -g_a - g_b

            b[ia] += w * g_a
            b[ib] += w * g_b
            b[ic] += w * g_c


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
    compute_b()
    computeAp(x_vec)
    compute_r0(b,Ap_vec,r0)
    r_init = dot_product(r0,r0)
    compute(r_vec,r0)
    compute(p_vec,r_vec)
    re0 = dot_product(r_vec,r_vec)
    for _ in range(100):
        computeAp(p_vec)
        denom = (dot_product(Ap_vec,p_vec))
        if denom < 1e-8:
            break
        alpha = dot_product(r_vec,r_vec) / denom
        axpy(x_vec,alpha,p_vec)
        axpy(r_vec,-alpha,Ap_vec)

        if dot_product(r_vec,r_vec) / r_init < 1e-6:
            break

        re1 = dot_product(r_vec,r_vec)

        beta = re1 / re0
        re0 = re1

        axpy2(p_vec,beta,p_vec)
        compute_add(p_vec,r_vec)


def main():
    init_mesh()
    init_pos()
    gravity[None] = [0, -1]

    gui = ti.GUI("FEM128")
    while gui.running:
        mouse_pos = gui.get_cursor_pos()
        attractor_pos[None] = mouse_pos
        attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB)
        for i in range(50):
            substep()
        paint_phi(gui)
        gui.circle(mouse_pos, radius=15, color=0x336699)
        gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
        gui.circles(pos.to_numpy(), radius=2, color=0xFFAA33)
        gui.show()


if __name__ == "__main__":
    main()