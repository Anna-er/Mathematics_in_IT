import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad
from math import sqrt, sin, cos, pi

@dataclass
class ThomasAlgorithmSolver:
    def __init__(self, n, a, b, c, d):
        self.n = n
        self.a = a
        self.b = b
        self.c = c
        self.d = d

def run_thomas(solver, y):
    a = solver.a
    b = solver.b
    c = solver.c
    d = solver.d
    n = solver.n

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    y[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]
    return y

@dataclass
class Solution:
    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.n = n
        self.h = (x[n] - x[0]) / n

def phi(s, i, x):
    if i == 0:
        if s.x[0] <= x <= s.x[1]:
            return (s.x[1] - x) / s.h
        else:
            return 0
    elif i == s.n:
        if s.x[s.n - 1] <= x <= s.x[s.n]:
            return (x - s.x[s.n - 1]) / s.h
        else:
            return 0
    else:
        if s.x[i - 1] <= x <= s.x[i]:
            return (x - s.x[i - 1]) / s.h
        elif s.x[i] <= x <= s.x[i + 1]:
            return (s.x[i + 1] - x) / s.h
        else:
            return 0

def f(s, x):
    l, r = 0, s.n

    while r - l > 1:
        mid = (l + r) // 2
        if x > s.x[mid]:
            l = mid
        else:
            r = mid

    yl = s.y[l]
    pl = phi(s, l, x)
    yr = s.y[r]
    pr = phi(s, r, x)
    res = yl * pl + yr * pr

    return res

@dataclass
class FEMSolver:
    def __init__(self, lambda_, l, grid_size):
        self.lambda_ = lambda_
        self.N = grid_size - 1
        self.h = l / (grid_size - 1)
        if self.h > sqrt(6 / lambda_):
            print("Warning: h > sqrt(6/λ)")
        self.x = np.linspace(0, l, grid_size)

def matrix_filling(solv, i, j):
    if i > j:
        i, j = j, i

    h = solv.h
    lambda_ = solv.lambda_
    xim = solv.x[i - 1]
    xi = solv.x[i]
    xip = solv.x[i + 1]

    if i == j:
        return (lambda_ * (xi**2 * xip - xi * xip**2 + xip**3 / 3 - xi**2 * xim + xi * xim**2 - xim**3 / 3) + xip - xim) / h**2
    elif i + 1 == j:
        return ((-1 / 6.0) * (-6 + lambda_ * (xi - xip)**2) * (xi - xip)) / h**2
    else:
        return 0

def scalar_product_of_f_and_phi_j(solv, j):
    h = solv.h
    lambda_ = solv.lambda_
    xim = solv.x[j - 1]
    xi = solv.x[j]
    xip = solv.x[j + 1]

    return (2 * (-(xi - xip) * sqrt(lambda_) * cos(xi * sqrt(lambda_)) + sin(xi * sqrt(lambda_)) - sin(xip * sqrt(lambda_))) +
            2 * (-sqrt(lambda_) * (xi - xim) * cos(xi * sqrt(lambda_)) + sin(xi * sqrt(lambda_)) - sin(sqrt(lambda_) * xim))) / h

def run_fem_solver(s):
    ts = ThomasAlgorithmSolver(s.N - 1, np.zeros(s.N - 1), np.zeros(s.N - 1), np.zeros(s.N - 1), np.zeros(s.N - 1))

    for i in range(1, s.N):
        if i - 1 >= 1:
            ts.a[i - 1] = matrix_filling(s, i - 1, i)
        if i + 1 < s.N:
            ts.c[i - 1] = matrix_filling(s, i + 1, i)
        ts.b[i - 1] = matrix_filling(s, i, i)
        ts.d[i - 1] = scalar_product_of_f_and_phi_j(s, i)

    y = np.zeros(s.N + 1)
    run_thomas(ts, y[1:])
    y[0] = 0
    y[s.N] = 0

    return Solution(s.x, y, s.N)

def main():
    # import sys
    # if len(sys.argv) < 3:
    #     print(f"Using: {sys.argv[0]} <λ> <grid_size>")
    #     return

    # lambda_ = float(sys.argv[1])
    # n = int(sys.argv[2])

    lambd = [1, 10, 20]
    N = [10, 50, 100, 10000]
    for n in N:
        for lambda_ in lambd:
            l = 3 * pi / sqrt(lambda_)

            fs = FEMSolver(lambda_, l, n)
            s = run_fem_solver(fs)

            h = fs.h

            max_error = 0.0
            nn = n * 10
            nh = l / nn
            for i in range(nn):
                x = i * nh
                my_val = f(s, x)
                real_val = sin(sqrt(lambda_) * x)
                err = abs(my_val - real_val)
                if err > max_error:
                    max_error = err
            
            p, q, c1, p1, J_m = 1, lambda_, 1, 0, 0.1
            test_n = n * 100
            test_X = np.linspace(0.0, l, test_n)
            test_h = l / test_n

            y = np.array([sin(sqrt(lambda_) * x) for x in test_X])
            y_h = np.array([f(s, x) for x in test_X])
            norm_y_m_yh = sqrt(np.sum((y - y_h) ** 2))

            fun = lambda x: 2 * lambda_ * np.sin(np.sqrt(lambda_) * x)
            norm_f = sqrt(quad(lambda x: fun(x) ** 2, 0, l)[0])
            
            c = q * l ** 2 / 4 + 1
            c_ = J_m * sqrt(p + (q * l ** 2) / 4)

            print(f"N = {n} | λ = {lambda_} | max_error = {max_error:.6f} | error = {round(norm_y_m_yh, 2)} | estimation = {((c * c_)**2 * s.h**2 * norm_f)} | h^2 = {h**2:.6f}")

if __name__ == "__main__":
    main()
