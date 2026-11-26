import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# ==============================================================================
# 1. SIP PROBLEM CLASS WITH EXACT SLSQP PROJECTION & CALL COUNTER
# ==============================================================================
class SIPProblem:
    def __init__(self, name, f, grad_f, g_vec, dg_dx_vec, t_grid, t_star, bounds):
        self.name = name
        self.f = f
        self.grad_f = grad_f
        self.g_vec = g_vec
        self.dg_dx_vec = dg_dx_vec
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.t_star = float(t_star)
        self.bounds = bounds
        self.low = np.array([b[0] for b in bounds], dtype=float)
        self.high = np.array([b[1] for b in bounds], dtype=float)
        self.proj_calls = 0

    def reset_counter(self):
        self.proj_calls = 0

    def t_mix(self, lam):
        return (1.0 - lam) * self.t_star + lam * self.t_grid

    def m_lambda(self, x, lam=1.0, custom_grid=None):
        grid = self.t_grid if custom_grid is None else custom_grid
        tm = (1.0 - lam) * self.t_star + lam * grid
        vals = self.g_vec(x, tm)
        idx = int(np.argmax(vals))
        return float(vals[idx]), float(tm[idx % len(tm)])

    def exact_project(self, y, lam=1.0):
        self.proj_calls += 1
        y_clip = np.clip(y, self.low, self.high)
        viol, _ = self.m_lambda(y_clip, lam)
        if viol <= 1e-10:
            return y_clip, True

        tm = self.t_mix(lam)
        res = minimize(
            fun=lambda z: 0.5 * np.sum((z - y_clip) ** 2),
            x0=y_clip,
            jac=lambda z: z - y_clip,
            method="SLSQP",
            bounds=self.bounds,
            constraints={
                "type": "ineq",
                "fun": lambda z: -self.g_vec(z, tm),
                "jac": lambda z: -self.dg_dx_vec(z, tm),
            },
            options={"ftol": 1e-12, "maxiter": 250, "disp": False},
        )
        z_proj = np.clip(res.x, self.low, self.high)
        final_viol, _ = self.m_lambda(z_proj, lam)
        return z_proj, (final_viol <= 1e-6)

    def kkt_residual(self, x, lam=1.0, alpha_ref=0.1):
        x_next, _ = self.exact_project(x - alpha_ref * self.grad_f(x), lam)
        return float(np.linalg.norm(x_next - x) / alpha_ref)


# ==============================================================================
# 2. LOCAL DESCENT & BUDGET-MATCHED BASELINES
# ==============================================================================
def armijo_pgd_step(prob, x, lam=1.0, alpha_init=0.5, c1=1e-4):
    g = prob.grad_f(x)
    fx = prob.f(x)
    alpha = alpha_init
    for _ in range(20):
        x_trial, ok = prob.exact_project(x - alpha * g, lam)
        if ok:
            s = x_trial - x
            if prob.f(x_trial) <= fx + c1 * np.dot(g, s) + 1e-12:
                return x_trial, True
        alpha *= 0.5
    return x.copy(), False


def run_single_pgd(prob, x0, max_proj_budget=500, kkt_tol=1e-7):
    prob.reset_counter()
    x, _ = prob.exact_project(x0, 1.0)
    traj = [x.copy()]
    f_hist = [prob.f(x)]
    while prob.proj_calls < max_proj_budget:
        if prob.kkt_residual(x, 1.0) < kkt_tol:
            break
        x, moved = armijo_pgd_step(prob, x, 1.0)
        traj.append(x.copy())
        f_hist.append(prob.f(x))
        if not moved:
            break
    return np.array(traj), np.array(f_hist), x, prob.proj_calls


def run_budget_multistart_pgd(prob, x0, proj_budget, rng, kkt_tol=1e-6):
    prob.reset_counter()
    x, _ = prob.exact_project(x0, 1.0)
    best_x = x.copy()
    best_f = prob.f(x)
    f_hist = [best_f]
    restarts = 1

    while prob.proj_calls < proj_budget:
        if prob.kkt_residual(x, 1.0) < kkt_tol:
            if prob.f(x) < best_f:
                best_f = prob.f(x)
                best_x = x.copy()
            x_rand = rng.uniform(prob.low, prob.high)
            x, _ = prob.exact_project(x_rand, 1.0)
            restarts += 1
        else:
            x, moved = armijo_pgd_step(prob, x, 1.0)
            if prob.f(x) < best_f:
                best_f = prob.f(x)
                best_x = x.copy()
            if not moved:
                x_rand = rng.uniform(prob.low, prob.high)
                x, _ = prob.exact_project(x_rand, 1.0)
                restarts += 1
        f_hist.append(best_f)
    return np.array(f_hist), best_x, prob.proj_calls, restarts


def run_budget_perturbed_pgd(prob, x0, proj_budget, rng, noise_r=0.35, kkt_tol=1e-6):
    prob.reset_counter()
    x, _ = prob.exact_project(x0, 1.0)
    best_x = x.copy()
    best_f = prob.f(x)
    f_hist = [best_f]

    while prob.proj_calls < proj_budget:
        if prob.kkt_residual(x, 1.0) < kkt_tol:
            if prob.f(x) < best_f:
                best_f = prob.f(x)
                best_x = x.copy()
            direction = rng.normal(size=len(x))
            direction /= np.linalg.norm(direction) + 1e-12
            radius = noise_r * (rng.uniform() ** (1.0 / len(x)))
            x, _ = prob.exact_project(x + radius * direction, 1.0)
        else:
            x, moved = armijo_pgd_step(prob, x, 1.0)
            if prob.f(x) < best_f:
                best_f = prob.f(x)
                best_x = x.copy()
            if not moved:
                x, _ = prob.exact_project(x + rng.normal(scale=noise_r, size=len(x)), 1.0)
        f_hist.append(best_f)
    return np.array(f_hist), best_x, prob.proj_calls


def run_budget_basin_hopping(prob, x0, proj_budget, rng, hop_scale=1.2, temp=0.5, kkt_tol=1e-5):
    prob.reset_counter()
    x, _ = prob.exact_project(x0, 1.0)
    curr_x = x.copy()
    curr_f = prob.f(x)
    best_x = curr_x.copy()
    best_f = curr_f
    f_hist = [best_f]

    while prob.proj_calls < proj_budget:
        for _ in range(30):
            if prob.proj_calls >= proj_budget or prob.kkt_residual(x, 1.0) < kkt_tol:
                break
            x, moved = armijo_pgd_step(prob, x, 1.0)
            f_hist.append(best_f)
            if not moved:
                break
        fx = prob.f(x)
        if fx < curr_f or rng.uniform() < np.exp(-(fx - curr_f) / temp):
            curr_x = x.copy()
            curr_f = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        f_hist.append(best_f)
        if prob.proj_calls >= proj_budget:
            break
        x, _ = prob.exact_project(curr_x + rng.uniform(-hop_scale, hop_scale, size=len(x)), 1.0)

    return np.array(f_hist), best_x, prob.proj_calls


# ==============================================================================
# 3. TWO-PHASE PROJECTED SEA-HOMOTOPY SOLVER
# ==============================================================================
def skew_rotation_vector(n, theta, plane_idx=0):
    v = np.zeros(n)
    i = plane_idx % max(1, n - 1)
    j = (i + 1) % n
    v[i] = np.cos(theta)
    v[j] = np.sin(theta)
    return v


def run_two_phase_sea_homotopy(
    prob,
    x0,
    M=5,
    stage_iters=80,
    kkt_tol=1e-4,
    c_s=0.14,
    delta_theta=np.pi / 10.0,
    max_spiral_angle=6.0 * np.pi,
    polish_tol=1e-7,
):
    prob.reset_counter()
    n = len(x0)
    lambdas = [1.0] if M == 1 else list(np.linspace(0.0, 1.0, M))

    x, _ = prob.exact_project(x0, lambdas[0])
    stage_best_x = x.copy()
    stage_best_f = prob.f(x)

    x_f1_init, _ = prob.exact_project(x, 1.0)
    f1_best_x = x_f1_init.copy()
    f1_best_f = prob.f(x_f1_init)

    traj = [x.copy()]
    f1_incumbent_hist = [f1_best_f]
    archive = []

    for lam in lambdas:
        x, _ = prob.exact_project(stage_best_x, lam)
        stage_best_x = x.copy()
        stage_best_f = prob.f(x)

        mode = "descent"
        center = None
        theta = 0.0
        plane_idx = 0

        for _ in range(stage_iters):
            if mode == "descent":
                res_kkt = prob.kkt_residual(x, lam)
                if res_kkt < kkt_tol:
                    fx = prob.f(x)
                    if fx < stage_best_f:
                        stage_best_f = fx
                        stage_best_x = x.copy()
                    if all(np.linalg.norm(x - a) > 1e-2 for a in archive):
                        archive.append(x.copy())
                    mode = "explore"
                    center = x.copy()
                    theta = 0.0
                    plane_idx += 1
                else:
                    x, moved = armijo_pgd_step(prob, x, lam)
                    fx = prob.f(x)
                    if fx < stage_best_f:
                        stage_best_f = fx
                        stage_best_x = x.copy()
                    if not moved:
                        mode = "explore"
                        center = x.copy()
                        theta = 0.0
            else:
                theta += delta_theta
                if theta > max_spiral_angle:
                    x = stage_best_x.copy()
                    break
                rho = c_s * theta
                v = skew_rotation_vector(n, theta, plane_idx)
                x_cand, ok = prob.exact_project(center + rho * v, lam)

                if ok and np.linalg.norm(x_cand - center) > 1e-2:
                    x_pgd, _ = prob.exact_project(x_cand - 0.1 * prob.grad_f(x_cand), lam)
                    d_proj = x_pgd - x_cand
                    if np.linalg.norm(d_proj) > 1e-3 and np.dot(d_proj, x_cand - center) > 0:
                        x = x_cand
                        mode = "descent"
                    else:
                        x = x_cand

            if prob.m_lambda(x, 1.0)[0] <= 1e-8 and prob.f(x) < f1_best_f:
                f1_best_f = prob.f(x)
                f1_best_x = x.copy()

            traj.append(x.copy())
            f1_incumbent_hist.append(f1_best_f)

        x = stage_best_x.copy()
        for _ in range(35):
            if prob.kkt_residual(x, lam) < polish_tol:
                break
            x, moved = armijo_pgd_step(prob, x, lam)
            if not moved:
                break
        stage_best_x = x.copy()
        stage_best_f = prob.f(x)

        if prob.m_lambda(stage_best_x, 1.0)[0] <= 1e-8 and stage_best_f < f1_best_f:
            f1_best_f = stage_best_f
            f1_best_x = stage_best_x.copy()

    x = stage_best_x.copy()
    for _ in range(40):
        if prob.kkt_residual(x, 1.0) < polish_tol:
            break
        x, moved = armijo_pgd_step(prob, x, 1.0)
        if not moved:
            break
    if prob.f(x) < f1_best_f and prob.m_lambda(x, 1.0)[0] <= 1e-7:
        f1_best_x = x.copy()
        f1_best_f = prob.f(x)

    return np.array(traj), np.array(f1_incumbent_hist), archive, f1_best_x, prob.proj_calls


# ==============================================================================
# 4. BENCHMARK PROBLEMS (WITH GENUINE DISCONNECTED POCKETS IN EXP 1)
# ==============================================================================
def build_exp1_disconnected_pockets(N=101):
    def f(x):
        x1, x2 = x
        well_left = -1.45 * np.exp(-2.2 * ((x1 + 1.05) ** 2 + (x2 + 0.10) ** 2))
        well_right = -0.65 * np.exp(-2.2 * ((x1 - 1.05) ** 2 + (x2 - 0.10) ** 2))
        quad = 0.10 * ((x1 + 0.2) ** 2 + x2 ** 2)
        return well_left + well_right + quad

    def grad_f(x):
        x1, x2 = x
        e_l = np.exp(-2.2 * ((x1 + 1.05) ** 2 + (x2 + 0.10) ** 2))
        e_r = np.exp(-2.2 * ((x1 - 1.05) ** 2 + (x2 - 0.10) ** 2))
        g1 = -1.45 * (-4.4 * (x1 + 1.05)) * e_l - 0.65 * (-4.4 * (x1 - 1.05)) * e_r + 0.20 * (x1 + 0.2)
        g2 = -1.45 * (-4.4 * (x2 + 0.10)) * e_l - 0.65 * (-4.4 * (x2 - 0.10)) * e_r + 0.20 * x2
        return np.array([g1, g2])

    def g_vec(x, t_arr):
        r2 = x[0] ** 2 + x[1] ** 2
        return r2 - 1.55 + 1.85 * np.exp(-3.2 * r2) + 0.65 * (1.0 - x[0] ** 2) * (t_arr ** 2)

    def dg_dx_vec(x, t_arr):
        r2 = x[0] ** 2 + x[1] ** 2
        dbar = 1.85 * (-6.4) * np.exp(-3.2 * r2)
        dg1 = 2.0 * x[0] + dbar * x[0] - 1.30 * x[0] * (t_arr ** 2)
        dg2 = np.full_like(t_arr, 2.0 * x[1] + dbar * x[1])
        return np.column_stack([dg1, dg2])

    return SIPProblem(
        name="Exp 1: Disconnected-Pocket Nonconvex SIP",
        f=f,
        grad_f=grad_f,
        g_vec=g_vec,
        dg_dx_vec=dg_dx_vec,
        t_grid=np.linspace(-1.0, 1.0, N),
        t_star=0.0,
        bounds=[(-2.5, 2.5), (-2.0, 2.0)],
    )


def build_exp2_compact_chebyshev(N=101):
    def f(x):
        return float(x[2])

    def grad_f(x):
        return np.array([0.0, 0.0, 1.0])

    def g_vec(x, t_arr):
        upper = np.exp(t_arr) - (x[0] + x[1] * t_arr) - x[2]
        lower = -np.exp(t_arr) + (x[0] + x[1] * t_arr) - x[2]
        return np.concatenate([upper, lower])

    def dg_dx_vec(x, t_arr):
        ones = np.ones_like(t_arr)
        j_up = np.column_stack([-ones, -t_arr, -ones])
        j_lo = np.column_stack([ones, t_arr, -ones])
        return np.vstack([j_up, j_lo])

    return SIPProblem(
        name="Exp 2: Compact Chebyshev Approximation of exp(t)",
        f=f,
        grad_f=grad_f,
        g_vec=g_vec,
        dg_dx_vec=dg_dx_vec,
        t_grid=np.linspace(0.0, 1.0, N),
        t_star=0.5,
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (0.0, 5.0)],
    )


def build_exp3_boundary_saddle(N=101):
    def f(x):
        return float(x[1] - 0.75 * x[0] ** 2)

    def grad_f(x):
        return np.array([-1.5 * x[0], 1.0])

    def g_vec(x, t_arr):
        return t_arr * x[0] - x[1] - 0.5 * t_arr ** 2

    def dg_dx_vec(x, t_arr):
        return np.column_stack([t_arr, -np.ones_like(t_arr)])

    return SIPProblem(
        name="Exp 3: Boundary-Induced Strict Saddle Trap",
        f=f,
        grad_f=grad_f,
        g_vec=g_vec,
        dg_dx_vec=dg_dx_vec,
        t_grid=np.linspace(-2.0, 2.0, N),
        t_star=0.0,
        bounds=[(-2.0, 2.0), (-1.0, 3.0)],
    )


def fmt_sci(val):
    if abs(val) < 1e-18:
        return "$0.00 \\times 10^{0}$"
    s = f"{val:.2e}"
    mant, exp = s.split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


# ==============================================================================
# 5. MAIN EXECUTION & AUTOMATIC LATEX TABLE GENERATOR
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    t_dense = np.linspace(-1.0, 1.0, 2001)

    # --- EXP 1 ---
    exp1 = build_exp1_disconnected_pockets(N=101)
    x0_1 = np.array([1.05, 0.00])

    traj_sea5, hist_sea5, arch1, x_sea5, budget1 = run_two_phase_sea_homotopy(
        exp1, x0_1, M=5, stage_iters=70, c_s=0.14
    )
    traj_sea1, hist_sea1, _, x_sea1, calls_sea1 = run_two_phase_sea_homotopy(
        exp1, x0_1, M=1, stage_iters=350, c_s=0.14
    )
    _, hist_pgd1, x_pgd1, calls_pgd1 = run_single_pgd(exp1, x0_1, max_proj_budget=budget1)
    hist_ms1, x_ms1, calls_ms1, n_restarts = run_budget_multistart_pgd(exp1, x0_1, budget1, rng)
    hist_pert1, x_pert1, calls_pert1 = run_budget_perturbed_pgd(exp1, x0_1, budget1, rng)
    hist_bh1, x_bh1, calls_bh1 = run_budget_basin_hopping(exp1, x0_1, budget1, rng)

    print("\n% ==================== LATEX TABLE 1 ====================")
    for name, x_sol, calls in [
        ("Single-Start PGD", x_pgd1, calls_pgd1),
        (f"Multi-Start PGD ({n_restarts}$\\times$)", x_ms1, calls_ms1),
        ("Perturbed PGD \\cite{Jin2017}", x_pert1, calls_pert1),
        ("Basin Hopping \\cite{WalesDoye1997}", x_bh1, calls_bh1),
        ("SEA without Homotopy ($M=1$)", x_sea1, calls_sea1),
        ("\\textbf{Two-Phase SEA-Homotopy ($M=5$)}", x_sea5, budget1),
    ]:
        vd, _ = exp1.m_lambda(x_sol, 1.0, custom_grid=t_dense)
        kr = exp1.kkt_residual(x_sol, 1.0)
        print(f"{name} & $({x_sol[0]:.3f}, {x_sol[1]:.3f})$ & ${exp1.f(x_sol):.6f}$ & {fmt_sci(vd)} & {fmt_sci(kr)} & ${calls}$ \\\\")

    # --- EXP 2 ---
    exp2 = build_exp2_compact_chebyshev(N=101)
    x0_2 = np.array([0.5, 0.5, 2.0])
    _, _, _, x_sea2, budget2 = run_two_phase_sea_homotopy(exp2, x0_2, M=5, stage_iters=60, c_s=0.05)
    _, _, x_pgd2, calls_pgd2 = run_single_pgd(exp2, x0_2, max_proj_budget=budget2)

    print("\n% ==================== LATEX TABLE 2 ====================")
    t_dense_01 = np.linspace(0.0, 1.0, 2001)
    for name, x_sol, calls in [
        ("Single-Start PGD", x_pgd2, calls_pgd2),
        ("\\textbf{Two-Phase SEA-Homotopy}", x_sea2, budget2),
    ]:
        vd, _ = exp2.m_lambda(x_sol, 1.0, custom_grid=t_dense_01)
        kr = exp2.kkt_residual(x_sol, 1.0)
        print(f"{name} & $({x_sol[0]:.3f}, {x_sol[1]:.3f}, {x_sol[2]:.3f})$ & ${exp2.f(x_sol):.6f}$ & {fmt_sci(vd)} & {fmt_sci(kr)} & ${calls}$ \\\\")

    # --- EXP 3 (stage_iters=80 so it reaches the exact corner (-2.0, 2.0) or (2.0, 2.0)) ---
    exp3 = build_exp3_boundary_saddle(N=101)
    x0_3 = np.array([0.0, 0.0])
    _, _, _, x_sea3, budget3 = run_two_phase_sea_homotopy(exp3, x0_3, M=3, stage_iters=80, c_s=0.08)
    _, _, x_pgd3_exact, _ = run_single_pgd(exp3, x0_3, max_proj_budget=budget3)
    _, _, x_pgd3_pert, _ = run_single_pgd(exp3, np.array([1e-4, 1e-4]), max_proj_budget=budget3)

    print("\n% ==================== LATEX TABLE 3 ====================")
    for name, x_sol in [
        ("PGD from exact saddle $(0,0)$", x_pgd3_exact),
        ("PGD from perturbed $(10^{-4}, 10^{-4})$", x_pgd3_pert),
        ("\\textbf{Two-Phase SEA from exact $(0,0)$}", x_sea3),
    ]:
        vd, _ = exp3.m_lambda(x_sol, 1.0, custom_grid=np.linspace(-2.0, 2.0, 2001))
        kr = exp3.kkt_residual(x_sol, 1.0)
        print(f"{name} & $({x_sol[0]:.3f}, {x_sol[1]:.3f})$ & ${exp3.f(x_sol):.6f}$ & {fmt_sci(vd)} & {fmt_sci(kr)} \\\\")

    # --- EXP 4 ---
    starts_20 = rng.uniform(low=[-1.5, -1.0], high=[1.5, 1.0], size=(20, 2))
    res_20 = {
        "Single-Start PGD": [],
        "Multi-Start PGD (Budget)": [],
        "Perturbed PGD (Budget)": [],
        "Basin Hopping (Budget)": [],
        "SEA without Homotopy ($M=1$)": [],
        "\\textbf{Two-Phase SEA-Homotopy ($M=5$)}": [],
    }
    for s in starts_20:
        _, _, xp, _ = run_single_pgd(exp1, s, max_proj_budget=budget1)
        _, xm, _, _ = run_budget_multistart_pgd(exp1, s, budget1, rng)
        _, xpert, _ = run_budget_perturbed_pgd(exp1, s, budget1, rng)
        _, xbh, _ = run_budget_basin_hopping(exp1, s, budget1, rng)
        _, _, _, xs1, _ = run_two_phase_sea_homotopy(exp1, s, M=1, stage_iters=350, c_s=0.14)
        _, _, _, xs5, _ = run_two_phase_sea_homotopy(exp1, s, M=5, stage_iters=70, c_s=0.14)

        res_20["Single-Start PGD"].append(exp1.f(xp))
        res_20["Multi-Start PGD (Budget)"].append(exp1.f(xm))
        res_20["Perturbed PGD (Budget)"].append(exp1.f(xpert))
        res_20["Basin Hopping (Budget)"].append(exp1.f(xbh))
        res_20["SEA without Homotopy ($M=1$)"].append(exp1.f(xs1))
        res_20["\\textbf{Two-Phase SEA-Homotopy ($M=5$)}"] .append(exp1.f(xs5))

    print("\n% ==================== LATEX TABLE 4 ====================")
    for m_name, vals in res_20.items():
        arr = np.array(vals)
        hit = 100.0 * np.mean(arr <= -1.20)
        print(f"{m_name} & ${np.mean(arr):.5f}$ & ${np.std(arr):.5f}$ & ${np.min(arr):.5f}$ & ${hit:.1f}\\%$ \\\\")

    # --- EXP 5 ---
    print("\n% ==================== LATEX TABLE 5 ====================")
    configs = [
        ("$c_s = 0.04$ ($M=5, N=101$)", dict(c_s=0.04, M=5, N=101)),
        ("$c_s = 0.14$ (default)", dict(c_s=0.14, M=5, N=101)),
        ("$c_s = 0.22$ ($M=5, N=101$)", dict(c_s=0.22, M=5, N=101)),
        ("$M = 1$ (No Homotopy)", dict(c_s=0.14, M=1, N=101)),
        ("$M = 3$", dict(c_s=0.14, M=3, N=101)),
        ("$M = 5$ (default)", dict(c_s=0.14, M=5, N=101)),
        ("$N = 25$", dict(c_s=0.14, M=5, N=25)),
        ("$N = 51$", dict(c_s=0.14, M=5, N=51)),
        ("$N = 101$ (default)", dict(c_s=0.14, M=5, N=101)),
        ("$N = 201$", dict(c_s=0.14, M=5, N=201)),
    ]
    for label, kw in configs:
        p_sub = build_exp1_disconnected_pockets(N=kw["N"])
        st_it = max(50, 350 // kw["M"])
        _, _, _, x_sub, calls_sub = run_two_phase_sea_homotopy(
            p_sub, x0_1, M=kw["M"], stage_iters=st_it, c_s=kw["c_s"]
        )
        vd, _ = p_sub.m_lambda(x_sub, 1.0, custom_grid=t_dense)
        kr = p_sub.kkt_residual(x_sub, 1.0)
        print(f"{label} & ${p_sub.f(x_sub):.6f}$ & {fmt_sci(vd)} & {fmt_sci(kr)} & ${calls_sub}$ \\\\")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    gx = np.linspace(-1.8, 1.8, 180)
    gy = np.linspace(-1.5, 1.5, 180)
    GX, GY = np.meshgrid(gx, gy)
    FZ = np.array([[exp1.f([xx, yy]) for xx in gx] for yy in gy])
    M0 = np.array([[exp1.m_lambda([xx, yy], 0.0)[0] for xx in gx] for yy in gy])
    M1 = np.array([[exp1.m_lambda([xx, yy], 1.0)[0] for xx in gx] for yy in gy])

    axes[0].contourf(GX, GY, FZ, levels=35, cmap="viridis", alpha=0.55)
    axes[0].contour(GX, GY, M0, levels=[0.0], colors="gray", linestyles="--", linewidths=1.5)
    axes[0].contour(GX, GY, M1, levels=[0.0], colors="black", linewidths=2.2)
    axes[0].plot(traj_sea1[:, 0], traj_sea1[:, 1], ":", color="tab:purple", lw=1.8, label="SEA ($M=1$, Trapped)")
    axes[0].plot(traj_sea5[:, 0], traj_sea5[:, 1], "-", color="tab:red", lw=1.8, label="SEA-Homotopy ($M=5$)")
    axes[0].scatter([x0_1[0]], [x0_1[1]], color="white", edgecolor="black", s=70, zorder=5, label="Start $x_0$")
    axes[0].scatter([x_sea5[0]], [x_sea5[1]], color="gold", edgecolor="black", marker="*", s=180, zorder=6, label="Global KKT")
    axes[0].set_title("(a) Disconnected Pockets $F(1)$ Bridged by $F(0)$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(hist_pgd1, "--", color="tab:orange", lw=1.8, label="Single-Start PGD")
    axes[1].plot(hist_ms1, "-.", color="tab:blue", lw=1.8, label="Multi-Start PGD (Budget)")
    axes[1].plot(hist_pert1, ":", color="tab:green", lw=1.8, label="Perturbed PGD (Budget)")
    axes[1].plot(hist_sea1, "--", color="tab:purple", lw=1.8, label="SEA ($M=1$, No Homotopy)")
    axes[1].plot(hist_sea5, "-", color="tab:red", lw=2.2, label="Two-Phase SEA-Homotopy ($M=5$)")
    axes[1].set_title("(b) $F(1)$-Feasible Incumbent Objective vs. Steps")
    axes[1].set_xlabel("Iteration / Step")
    axes[1].set_ylabel("$f(x_{\\mathrm{best}}^{(F(1))})$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("sea_sip_benchmarks.pdf")
    plt.savefig("sea_sip_benchmarks.png", dpi=300)
    plt.show()