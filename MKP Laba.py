import numpy as np
from scipy.integrate import odeint
from math import *
import matplotlib.pyplot as plt

MU = 398600
SIGMA = 3.5 * 23 / 1650 / 2
y0 = [0.0349, 1.2968, 7291.832, 0.1745, 0.0239]  # Начальные условия


def ctg(x):  # Функция котангенса
    if sin(x) == 0:
        return 0
    return cos(x) / sin(x)


def f(y, u):  # Функция для интегрирования
    big_omega, i, p, omega, e = y
    theta = u - omega
    v2 = (MU / p) * (1 + e**2 + 2 * e * cos(theta))
    r = p / (1 + e * cos(theta))
    ro0 = 1.225*10**(-8)
    hs = 8454
    ro = ro0 * exp((6371 - r) / hs)
    Fa = SIGMA * ro * v2
    f1 = -Fa * ((e * sin(theta)) / (1 + e**2 + 2 * e * cos(theta)) ** (1/2))
    f2 = -Fa * ((1 + e * cos(theta)) / (1 + e**2 + 2 * e * cos(theta)) ** (1/2))
    f3 = 0
    gamma = 1 - ((f3 * r**3) / (MU * p)) * sin(u) * ctg(i)
    dbig_omegadu = (r**3 * f3 * sin(u)) / (gamma * MU * p * sin(i))
    didu = (r**3 * f3 * cos(u)) / (gamma * MU * p)
    dpdu = (2 * f2 * r**3) / (gamma * MU)
    domegadu = (1 / gamma) * (r ** 2 / (MU * e)) * \
               (-f1 * cos(theta) + f2 * (1 + (r / p)) * sin(theta) - f3 * (r / p) * e * ctg(i) * sin(u))
    dedu = (1 / gamma) * (r ** 2 / MU) * (f1 * sin(theta) + f2 * ((1 + (r / p)) * cos(theta) + e * (r / p)))
    return [dbig_omegadu, didu, dpdu, domegadu, dedu]


def plot_showing(w, t):  # Отображение графиков
    var_names = ['Ω', 'i', 'p', 'ω', 'e']
    for j, name in enumerate(var_names):
        plt.plot(t, w[:, j], '-,', linewidth=1)
        plt.title(label=name, loc='center')
        plt.grid(True)
        plt.xlabel("u (rad)", fontsize=15, fontweight="bold")
        plt.ylabel(name, loc="top", rotation=0, labelpad=-35, fontsize=13, fontweight="bold")
        plt.show()


u = np.linspace(0, pi * 50, 10000)
w = odeint(f, y0, u)  # Интегрирование
plot_showing(w, u)
