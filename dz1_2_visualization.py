from pathlib import Path
import math
import os

import numpy as np


# Matplotlib создает служебный кэш шрифтов. Кладем его в папку build,
# чтобы рядом с работой не появлялись лишние системные файлы.
os.environ.setdefault("MPLCONFIGDIR", str(Path("build/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ДЗ1.2.
# Нужно изобразить исходную функцию и кубическую аппроксимацию
# в одной координатной плоскости для нескольких итераций алгоритма.

EPS = 1e-4
A = 1.5
B = 2.0
FIG_DIR = Path("figures")
BUILD_DIR = Path("build")


def f(x):
    """Исходная функция варианта 9."""
    return x**3 / 3 - 5 * x + x * np.log(x)


def df(x):
    """Первая производная исходной функции."""
    return x**2 + np.log(x) - 4


def cubic_approximation(a=A, b=B, eps=EPS):
    """Та же последовательность, что в ДЗ1.1, нужна для построения графиков."""
    # Здесь расчет повторяется специально, чтобы файл ДЗ1.2 был самостоятельным:
    # его можно запустить отдельно без импорта из ДЗ1.1.
    u = a
    v = b
    rows = []

    for k in range(1, 100):
        fu = float(f(u))
        fv = float(f(v))
        dfu = float(df(u))
        dfv = float(df(v))

        # Находим стационарную точку кубической модели.
        w = dfv + dfu - 3 * (fv - fu) / (v - u)
        z = math.sqrt(max(w * w - dfu * dfv, 0.0))
        denominator = dfv - dfu + 2 * z

        if abs(denominator) < 1e-15:
            x_new = (u + v) / 2
        else:
            x_new = v - (v - u) * (dfv + z - w) / denominator

        x_new = min(max(x_new, u), v)
        row = {
            "k": k,
            "u": u,
            "v": v,
            "f_u": fu,
            "f_v": fv,
            "df_u": dfu,
            "df_v": dfv,
            "x_new": x_new,
            "f_new": float(f(x_new)),
            "df_new": float(df(x_new)),
        }
        rows.append(row)

        if abs(row["df_new"]) < eps:
            break
        if row["df_new"] > 0:
            v = x_new
        else:
            u = x_new

    return rows


def hermite_cubic(row, xs):
    """Кубический многочлен Эрмита на текущем отрезке [u, v]."""
    u = row["u"]
    v = row["v"]
    h = v - u
    t = (xs - u) / h

    # Формула использует значения функции и производной на двух концах отрезка.
    # Именно поэтому аппроксимация касается исходной функции на границах отрезка.
    return (
        row["f_u"] * (2 * t**3 - 3 * t**2 + 1)
        + row["df_u"] * h * (t**3 - 2 * t**2 + t)
        + row["f_v"] * (-2 * t**3 + 3 * t**2)
        + row["df_v"] * h * (t**3 - t**2)
    )


def plot_iterations(rows):
    """Для каждой итерации рисуем исходную функцию и свою кубическую модель."""
    # Берем чуть более широкий интервал, чем [1.5, 2],
    # чтобы на рисунке было видно поведение функции рядом с границами.
    xs = np.linspace(1.48, 2.02, 500)
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.5), squeeze=False)

    for ax, row in zip(axes[0], rows):
        # Кубическую модель корректно рисовать на текущем отрезке [u, v].
        local_xs = np.linspace(row["u"], row["v"], 300)
        ax.plot(xs, f(xs), linewidth=2.0, label="исходная функция f(x)")
        ax.plot(local_xs, hermite_cubic(row, local_xs), "--", linewidth=2.0, label="кубическая аппроксимация")
        ax.scatter([row["u"], row["v"]], [row["f_u"], row["f_v"]], color="green", zorder=3, label="концы отрезка")
        ax.scatter([row["x_new"]], [row["f_new"]], marker="*", s=150, color="orange", zorder=4, label="новая точка")
        ax.set_title(f"Итерация {row['k']}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("ДЗ1.2. Исходная функция и кубические аппроксимации", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dz1_cubic_iterations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_in_one(rows):
    """Дополнительный общий график: все кубические модели в одной плоскости."""
    # Этот рисунок прямо отвечает формулировке ДЗ1.2:
    # исходная и аппроксимирующие функции показаны в одной координатной плоскости.
    xs = np.linspace(1.48, 2.02, 500)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(xs, f(xs), color="black", linewidth=2.5, label="исходная функция f(x)")
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for i, row in enumerate(rows):
        local_xs = np.linspace(row["u"], row["v"], 300)
        ax.plot(
            local_xs,
            hermite_cubic(row, local_xs),
            "--",
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=f"аппроксимация, итерация {row['k']}",
        )
        ax.scatter([row["x_new"]], [row["f_new"]], marker="*", s=130, color=colors[i % len(colors)])

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Все кубические аппроксимации в одной координатной плоскости")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dz1_cubic_all_approximations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # Создаем папки, куда будут сохранены рисунки и кэш matplotlib.
    FIG_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    rows = cubic_approximation()
    plot_iterations(rows)
    plot_all_in_one(rows)

    print("ДЗ1.2. Построены графики:")
    print("  figures/dz1_cubic_iterations.png")
    print("  figures/dz1_cubic_all_approximations.png")


if __name__ == "__main__":
    main()
