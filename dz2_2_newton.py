from pathlib import Path
import json
import os

import numpy as np


# Кэш matplotlib кладем внутрь проекта.
os.environ.setdefault("MPLCONFIGDIR", str(Path("build/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ДЗ2.2.
# Для функции варианта 9 нужно:
#   1) найти стационарные точки аналитически;
#   2) определить тип каждой точки;
#   3) найти минимум численно методом Ньютона.

EPS = 1e-4
X0 = np.array([2.5, 4.5])
FIG_DIR = Path("figures")
BUILD_DIR = Path("build")


def z_value(x):
    """Функция z(x, y) из варианта 9."""
    x1, x2 = x
    return float(x1**3 - 6 * x1**2 + 9 * x1 + x2**3 - 11 * x2**2 + 39 * x2 - 49)


def grad_z(x):
    """Градиент функции z."""
    x1, x2 = x
    return np.array([3 * x1**2 - 12 * x1 + 9, 3 * x2**2 - 22 * x2 + 39], dtype=float)


def hessian_z(x):
    """Матрица Гессе функции z."""
    x1, x2 = x
    return np.array([[6 * x1 - 12, 0.0], [0.0, 6 * x2 - 22]], dtype=float)


def stationary_points():
    """Аналитические стационарные точки и их классификация."""
    # Уравнения grad z = 0 распадаются на два независимых квадратных уравнения:
    #   3x^2 - 12x + 9 = 0  ->  x = 1 или x = 3,
    #   3y^2 - 22y + 39 = 0 ->  y = 3 или y = 13/3.
    # Поэтому всего получается 4 стационарные точки.
    points = [
        np.array([1.0, 3.0]),
        np.array([1.0, 13.0 / 3.0]),
        np.array([3.0, 3.0]),
        np.array([3.0, 13.0 / 3.0]),
    ]
    rows = []

    for point in points:
        # Тип точки определяем по собственным значениям Гессиана:
        # оба > 0  -> минимум,
        # оба < 0  -> максимум,
        # разные знаки -> седловая точка.
        eig = np.linalg.eigvals(hessian_z(point))
        if np.all(eig > 0):
            kind = "локальный минимум"
        elif np.all(eig < 0):
            kind = "локальный максимум"
        else:
            kind = "седловая точка"

        rows.append(
            {
                "point": point.tolist(),
                "value": z_value(point),
                "hessian_eigenvalues": eig.tolist(),
                "type": kind,
            }
        )

    return rows


def newton_method():
    """Метод Ньютона: x_{k+1} = x_k - H^{-1}(x_k) grad z(x_k)."""
    x = X0.copy()
    rows = []

    for k in range(50):
        # Сначала сохраняем текущее состояние, потом проверяем критерий остановки.
        g = grad_z(x)
        H = hessian_z(x)
        grad_norm = float(np.linalg.norm(g))

        rows.append(
            {
                "k": k,
                "point": x.tolist(),
                "value": z_value(x),
                "gradient": g.tolist(),
                "grad_norm": grad_norm,
                "hessian": H.tolist(),
            }
        )

        if grad_norm < EPS:
            break

        # solve(H, g) вычисляет H^{-1}g без явного обращения матрицы.
        # Так численно устойчивее, чем писать np.linalg.inv(H) @ g.
        x = x - np.linalg.solve(H, g)

    return rows


def plot_newton(rows, stationary):
    """Рисуем линии уровня, путь Ньютона и все стационарные точки."""
    # path — последовательность точек метода Ньютона.
    path = np.array([row["point"] for row in rows])

    # points — все найденные аналитически стационарные точки.
    points = np.array([row["point"] for row in stationary])

    # Сетка выбрана так, чтобы были видны начальная точка, путь и все стационарные точки.
    xs = np.linspace(0.5, 3.6, 350)
    ys = np.linspace(2.6, 4.8, 350)
    xx, yy = np.meshgrid(xs, ys)
    zz = xx**3 - 6 * xx**2 + 9 * xx + yy**3 - 11 * yy**2 + 39 * yy - 49

    fig, ax = plt.subplots(figsize=(7, 5.5))
    contour = ax.contour(xx, yy, zz, levels=22, cmap="plasma", linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.2g")
    ax.plot(path[:, 0], path[:, 1], "o-", color="tab:blue", label="траектория Ньютона")

    for i, point in enumerate(path):
        ax.annotate(f"N{i}", point, textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.scatter(points[:, 0], points[:, 1], marker="x", s=90, color="tab:red", label="стационарные точки")
    ax.scatter([3], [13 / 3], marker="*", s=170, color="orange", label="локальный минимум")
    ax.set_title("ДЗ2.2. Метод Ньютона, вариант 9")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dz2_newton_path.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # figures — графики для отчета, build — численные результаты.
    FIG_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    stationary = stationary_points()
    rows = newton_method()
    plot_newton(rows, stationary)

    # Сохраняем и аналитическую часть, и итерации Ньютона.
    result = {
        "task": "ДЗ2.2",
        "eps": EPS,
        "stationary_points": stationary,
        "newton_iterations": rows,
        "answer": rows[-1],
    }
    (BUILD_DIR / "dz2_2_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ДЗ2.2. Метод Ньютона")
    for row in rows:
        x, y = row["point"]
        print(
            f"k={row['k']}: x=({x:.8f}, {y:.8f}), "
            f"z={row['value']:.9f}, ||grad z||={row['grad_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
