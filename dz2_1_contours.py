from pathlib import Path
import json
import os

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path("build/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ДЗ2.1.
# Нужно построить линии уровня функции из ЛР4 и ломаные траекторий
# для трех методов: покоординатного, градиентного и наискорейшего спуска.
#
# В задании разрешено воспользоваться готовым кодом из Contours and solutions.ipynb.
# Оттуда взята общая идея функции plot_history:
#   1) построить сетку через np.meshgrid;
#   2) посчитать функцию на этой сетке;
#   3) нарисовать линии уровня ax.contour;
#   4) поверх линий уровня нарисовать ломаную последовательности решений.
#
# Главное отличие от исходного notebook:
#   там было levels=50, то есть просто 50 линий уровня;
#   здесь levels берутся из значений F в вершинах ломаной,
#   потому что этого прямо требует ДЗ2.1.

EPS = 1e-4
X0 = np.array([-3.0, -3.0])
H = np.array([[6.0, 2.0], [2.0, 6.0]])
FIG_DIR = Path("figures")
BUILD_DIR = Path("build")


def F(x):
    """Квадратичная функция варианта 9."""
    x1, x2 = x
    return float(3 * x1**2 + 2 * x1 * x2 + 3 * x2**2 - 4 * x1 - 12 * x2 + 12)


def grad_F(x):
    """Градиент функции F."""
    x1, x2 = x
    return np.array([6 * x1 + 2 * x2 - 4, 2 * x1 + 6 * x2 - 12], dtype=float)


def make_result(name, path, steps):
    """Собираем результаты метода в один словарь."""
    # Значение функции и норма градиента нужны и для отчета, и для проверки
    # критерия остановки, и для построения линий уровня через вершины пути.
    values = [F(point) for point in path]
    grad_norms = [float(np.linalg.norm(grad_F(point))) for point in path]
    return {
        "method": name,
        "path": [point.tolist() for point in path],
        "values": values,
        "grad_norms": grad_norms,
        "steps": steps,
        "iterations": len(path) - 1,
        "final_point": path[-1].tolist(),
        "final_value": values[-1],
        "final_grad_norm": grad_norms[-1],
    }


def coordinate_descent():
    """Покоординатный спуск: по очереди точно минимизируем по x1 и x2."""
    x = X0.copy()
    path = [x.copy()]
    steps = []

    for _ in range(10000):
        old_x = x.copy()

        # Минимизация по x1 при фиксированном x2:
        # x1_new = x1 - (dF/dx1) / H11.
        g = grad_F(x)
        step = -g[0] / H[0, 0]
        x[0] += step
        path.append(x.copy())
        steps.append(float(step))

        # Минимизация по x2 при фиксированном x1:
        # x2_new = x2 - (dF/dx2) / H22.
        g = grad_F(x)
        step = -g[1] / H[1, 1]
        x[1] += step
        path.append(x.copy())
        steps.append(float(step))

        if np.linalg.norm(grad_F(x)) < EPS:
            break

        # Иногда норма градиента уже почти мала, но из-за округлений еще чуть выше EPS.
        # Тогда дополнительно проверяем, что за полный цикл точка почти не изменилась.
        if np.linalg.norm(x - old_x) < EPS:
            break

    return make_result("Покоординатный спуск", path, steps)


def gradient_descent():
    """Градиентный спуск с простым дроблением шага."""
    x = X0.copy()
    path = [x.copy()]
    steps = []

    for _ in range(10000):
        g = grad_F(x)
        if np.linalg.norm(g) < EPS:
            break

        direction = -g
        alpha = 0.1

        # Если с выбранным шагом функция не убывает, уменьшаем шаг в 2 раза.
        # Это простой вариант backtracking line search.
        while F(x + alpha * direction) > F(x):
            alpha /= 2

        x = x + alpha * direction
        path.append(x.copy())
        steps.append(float(alpha))

    return make_result("Градиентный спуск", path, steps)


def steepest_descent():
    """Наискорейший спуск: шаг точно минимизирует F вдоль антиградиента."""
    x = X0.copy()
    path = [x.copy()]
    steps = []

    for _ in range(10000):
        g = grad_F(x)
        if np.linalg.norm(g) < EPS:
            break

        # Для квадратичной функции точный шаг имеет явную формулу.
        alpha = float((g @ g) / (g @ H @ g))
        x = x - alpha * g
        path.append(x.copy())
        steps.append(alpha)

    return make_result("Наискорейший спуск", path, steps)


def contour_levels(values):
    """Уровни берутся из значений F в вершинах ломаной."""
    levels = sorted(float(value) for value in values)

    # contour требует строго возрастающий список уровней.
    # Если из-за округления два значения совпали, слегка разводим их.
    for i in range(1, len(levels)):
        if levels[i] <= levels[i - 1]:
            levels[i] = levels[i - 1] + 1e-10
    return levels


def plot_result(result, filename):
    """Строим линии уровня и ломаную последовательных приближений."""
    path = np.array(result["path"])

    # Границы графика подбираем по траектории, чтобы весь путь был виден.
    margin = 0.8
    x_min, x_max = path[:, 0].min() - margin, path[:, 0].max() + margin
    y_min, y_max = path[:, 1].min() - margin, path[:, 1].max() + margin

    xs = np.linspace(x_min, x_max, 350)
    ys = np.linspace(y_min, y_max, 350)

    # Та же логика, что в Contours and solutions.ipynb:
    # xx и yy задают прямоугольную сетку, а zz хранит значение функции в узлах.
    xx, yy = np.meshgrid(xs, ys)
    zz = 3 * xx**2 + 2 * xx * yy + 3 * yy**2 - 4 * xx - 12 * yy + 12

    colorlist = ["darkblue", "blue", "aqua", "lawngreen", "gold", "darkorange", "brown"]
    colormap = LinearSegmentedColormap.from_list("optimization_contours", colors=colorlist, N=256)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contour(xx, yy, zz, levels=contour_levels(result["values"]), cmap=colormap, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.3g")
    ax.plot(path[:, 0], path[:, 1], "o-", color="tab:red", linewidth=1.8, markersize=4)

    # Чтобы подписи не слипались около минимума, подписываем начало и конец.
    shown = set(range(min(6, len(path))))
    shown.add(len(path) - 1)
    for i in sorted(shown):
        ax.annotate(f"M{i}", path[i], textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.scatter([0], [2], marker="*", s=160, color="orange", label="минимум (0, 2)")
    ax.set_title(f"{result['method']}: уровни проходят через вершины ломаной")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    results = [
        coordinate_descent(),
        gradient_descent(),
        steepest_descent(),
    ]
    filenames = [
        "dz2_coordinate_descent.png",
        "dz2_gradient_descent.png",
        "dz2_steepest_descent.png",
    ]

    for result, filename in zip(results, filenames):
        plot_result(result, filename)

    (BUILD_DIR / "dz2_1_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ДЗ2.1. Результаты трех методов")
    for result in results:
        x = result["final_point"]
        print(
            f"{result['method']}: шагов={result['iterations']}, "
            f"x=({x[0]:.8f}, {x[1]:.8f}), "
            f"F={result['final_value']:.3e}, "
            f"||grad F||={result['final_grad_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
