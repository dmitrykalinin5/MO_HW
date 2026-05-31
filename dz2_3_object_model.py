from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np


# Matplotlib при первом запуске создает кэш шрифтов.
# Чтобы он не лез в домашнюю папку пользователя, кладем его в build/mplconfig.
os.environ.setdefault("MPLCONFIGDIR", str(Path("build/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ДЗ2.3.
# Задание просит построить объектную модель трех методов оптимизации.
# Поэтому здесь есть классы, но без лишнего усложнения:
#   1) абстрактная функция Objective2D;
#   2) конкретная функция варианта 9;
#   3) базовый оптимизатор Optimizer2D;
#   4) три метода оптимизации.

EPS = 1e-4
X0 = np.array([-3.0, -3.0])
FIG_DIR = Path("figures")
BUILD_DIR = Path("build")


class Objective2D(ABC):
    """Абстрактная целевая функция двух переменных.

    Оптимизаторам не важно, какая именно функция передана.
    Им достаточно уметь спросить у функции значение, градиент и Гессиан.
    """

    @abstractmethod
    def value(self, x):
        """Вернуть значение функции в точке x = [x1, x2]."""

    @abstractmethod
    def gradient(self, x):
        """Вернуть градиент функции в точке x."""

    @abstractmethod
    def hessian(self, x):
        """Вернуть матрицу Гессе в точке x."""


class Variant9Quadratic(Objective2D):
    """Квадратичная функция варианта 9 из ЛР4."""

    def value(self, x):
        x1, x2 = x
        return float(3 * x1**2 + 2 * x1 * x2 + 3 * x2**2 - 4 * x1 - 12 * x2 + 12)

    def gradient(self, x):
        x1, x2 = x
        return np.array([6 * x1 + 2 * x2 - 4, 2 * x1 + 6 * x2 - 12], dtype=float)

    def hessian(self, x):
        # Для квадратичной функции Гессиан постоянный.
        return np.array([[6.0, 2.0], [2.0, 6.0]])


@dataclass
class OptimizationResult:
    """Результат работы любого метода.

    Все методы возвращают одинаковую структуру, поэтому потом проще строить
    графики, таблицы и JSON: не нужно проверять, какой именно метод запускался.
    """

    method: str
    path: list
    values: list
    grad_norms: list
    steps: list


class Optimizer2D(ABC):
    """Базовый класс для методов оптимизации."""

    def __init__(self, objective, eps=EPS):
        # Это композиция: оптимизатор хранит внутри себя целевую функцию.
        self.objective = objective
        self.eps = eps

    @abstractmethod
    def run(self, x0):
        """Запустить метод из начальной точки x0."""

    def make_result(self, method, path, steps):
        """Собрать общий результат по накопленной траектории."""
        values = [self.objective.value(point) for point in path]
        grad_norms = [float(np.linalg.norm(self.objective.gradient(point))) for point in path]
        return OptimizationResult(method, path, values, grad_norms, steps)


class CoordinateDescent(Optimizer2D):
    """Покоординатный спуск.

    На каждом цикле сначала точно минимизируем по x1 при фиксированном x2,
    затем точно минимизируем по x2 при фиксированном x1.
    """

    def run(self, x0):
        x = x0.copy()
        path = [x.copy()]
        steps = []

        for _ in range(10000):
            old_x = x.copy()
            H = self.objective.hessian(x)

            # axis = 0 означает координату x1, axis = 1 означает координату x2.
            for axis in (0, 1):
                g = self.objective.gradient(x)

                # Для квадратичной функции минимум по одной координате находится точно:
                # x_i := x_i - (dF/dx_i) / H_ii.
                step = -g[axis] / H[axis, axis]
                x[axis] += step

                path.append(x.copy())
                steps.append(float(step))

            # Останавливаемся, когда градиент уже достаточно мал.
            if np.linalg.norm(self.objective.gradient(x)) < self.eps:
                break

            # Дополнительная защита от бесконечного цикла: если точка почти не меняется.
            if np.linalg.norm(x - old_x) < self.eps:
                break

        return self.make_result("Покоординатный спуск", path, steps)


class GradientDescent(Optimizer2D):
    """Градиентный спуск.

    Направление движения — антиградиент. Шаг сначала берем равным 0.1.
    Если функция не уменьшается, делим шаг пополам.
    """

    def run(self, x0):
        x = x0.copy()
        path = [x.copy()]
        steps = []

        for _ in range(10000):
            g = self.objective.gradient(x)
            if np.linalg.norm(g) < self.eps:
                break

            direction = -g
            alpha = 0.1

            # Дробление шага: новый шаг должен уменьшать значение функции.
            while self.objective.value(x + alpha * direction) > self.objective.value(x):
                alpha /= 2

            x = x + alpha * direction
            path.append(x.copy())
            steps.append(float(alpha))

        return self.make_result("Градиентный спуск", path, steps)


class SteepestDescent(Optimizer2D):
    """Наискорейший спуск.

    Направление такое же, как в градиентном спуске: -grad F.
    Отличие только в шаге: для квадратичной функции он считается точно.
    """

    def run(self, x0):
        x = x0.copy()
        path = [x.copy()]
        steps = []

        for _ in range(10000):
            g = self.objective.gradient(x)
            if np.linalg.norm(g) < self.eps:
                break

            H = self.objective.hessian(x)

            # Точный шаг вдоль антиградиента для квадратичной функции:
            # alpha = (g^T g) / (g^T H g).
            alpha = float((g @ g) / (g @ H @ g))

            x = x - alpha * g
            path.append(x.copy())
            steps.append(alpha)

        return self.make_result("Наискорейший спуск", path, steps)


def run_oop_methods():
    """Создать функцию, создать методы и запустить все три оптимизатора."""
    objective = Variant9Quadratic()
    methods = [
        CoordinateDescent(objective),
        GradientDescent(objective),
        SteepestDescent(objective),
    ]
    return [method.run(X0) for method in methods]


def add_box(ax, x, y, w, h, title, lines, color):
    """Нарисовать прямоугольник класса на схеме."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03",
        linewidth=1.3,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=8.8)


def add_arrow(ax, start, end, style="-", curve=0.0, color="#333333"):
    """Нарисовать стрелку между блоками схемы."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=1.2,
            linestyle=style,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        ),
    )


def draw_object_model():
    """Построить простой рисунок объектной модели."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Верхний ряд: два абстрактных класса.
    add_box(ax, 1.0, 6.6, 3.3, 1.6, "Objective2D", ["<<abstract>>", "value(x)", "gradient(x)", "hessian(x)"], "#eef5ff")
    add_box(ax, 8.7, 6.6, 3.3, 1.6, "Optimizer2D", ["<<abstract>>", "objective, eps", "run(x0)", "make_result(...)"], "#eef5ff")

    # Средний ряд: конкретная функция и три метода.
    add_box(ax, 1.0, 4.0, 3.3, 1.5, "Variant9Quadratic", ["F(x1, x2)", "grad F", "H = [[6,2],[2,6]]"], "#f4fff0")
    add_box(ax, 5.4, 3.8, 2.5, 1.6, "CoordinateDescent", ["по x1", "по x2"], "#fff8e8")
    add_box(ax, 8.5, 3.8, 2.5, 1.6, "GradientDescent", ["direction = -grad", "alpha = 0.1 / 2"], "#fff8e8")
    add_box(ax, 11.4, 3.8, 2.5, 1.6, "SteepestDescent", ["direction = -grad", "точный alpha"], "#fff8e8")

    # Нижний ряд: общий результат.
    add_box(ax, 7.7, 1.2, 3.7, 1.5, "OptimizationResult", ["method", "path", "values", "grad_norms", "steps"], "#f7f7f7")

    # Наследование показываем сплошными синими стрелками.
    add_arrow(ax, (2.65, 5.5), (2.65, 6.6), color="#2f4f6f")
    add_arrow(ax, (6.65, 5.4), (9.35, 6.6), curve=0.05, color="#2f4f6f")
    add_arrow(ax, (9.75, 5.4), (10.35, 6.6), color="#2f4f6f")
    add_arrow(ax, (12.65, 5.4), (11.35, 6.6), curve=-0.05, color="#2f4f6f")

    # Композицию и создание результата показываем серым пунктиром.
    add_arrow(ax, (8.7, 7.4), (4.3, 7.4), style="--", color="#606060")
    add_arrow(ax, (9.75, 3.8), (9.55, 2.7), style="--", color="#606060")

    ax.text(
        0.8,
        0.45,
        "Обозначения: сплошная синяя стрелка — наследование; пунктирная серая стрелка — композиция или создание результата.",
        fontsize=9,
    )
    ax.set_title("Упрощенная объектная модель реализации трех методов оптимизации", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dz2_object_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results(results):
    """Сохранить результаты ООП-реализации в JSON."""
    data = []
    for result in results:
        data.append(
            {
                "method": result.method,
                "path": [point.tolist() for point in result.path],
                "values": result.values,
                "grad_norms": result.grad_norms,
                "steps": result.steps,
                "iterations": len(result.path) - 1,
            }
        )

    (BUILD_DIR / "dz2_3_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    FIG_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    results = run_oop_methods()
    draw_object_model()
    save_results(results)

    print("ДЗ2.3. Упрощенная объектная модель построена: figures/dz2_object_model.png")
    for result in results:
        x = result.path[-1]
        print(
            f"{result.method}: шагов={len(result.path) - 1}, "
            f"x=({x[0]:.8f}, {x[1]:.8f}), "
            f"||grad F||={result.grad_norms[-1]:.3e}"
        )


if __name__ == "__main__":
    main()
