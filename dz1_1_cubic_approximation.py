from pathlib import Path
import json
import math

import numpy as np


# ДЗ1.1.
# Вариант 9:
#   f(x) = x^3 / 3 - 5x + x ln(x), [a, b] = [1.5, 2].
# Нужно найти минимум методом кубической аппроксимации.

EPS = 1e-4
A = 1.5
B = 2.0
BUILD_DIR = Path("build")


def f(x):
    """Исходная функция варианта 9."""
    return x**3 / 3 - 5 * x + x * np.log(x)


def df(x):
    """Первая производная: f'(x) = x^2 + ln(x) - 4."""
    return x**2 + np.log(x) - 4


def cubic_approximation(a=A, b=B, eps=EPS):
    """Строим последовательность приближений методом кубической аппроксимации."""
    # u и v — текущие границы отрезка, внутри которого находится минимум.
    # В начале это исходный отрезок [1.5, 2].
    u = a
    v = b

    # rows — история итераций. Ее потом удобно сохранить в JSON и перенести в отчет.
    rows = []

    for k in range(1, 100):
        # На каждой итерации методу нужны значения функции и производной
        # на двух концах текущего отрезка.
        fu = float(f(u))
        fv = float(f(v))
        dfu = float(df(u))
        dfv = float(df(v))

        # Эти величины входят в стандартную формулу минимума кубической модели.
        # Кубическая модель строится так, чтобы совпадать с f(x) и f'(x)
        # в точках u и v.
        w = dfv + dfu - 3 * (fv - fu) / (v - u)
        z = math.sqrt(max(w * w - dfu * dfv, 0.0))
        denominator = dfv - dfu + 2 * z

        # Если знаменатель почти нулевой, берем середину отрезка как безопасную точку.
        if abs(denominator) < 1e-15:
            x_new = (u + v) / 2
        else:
            x_new = v - (v - u) * (dfv + z - w) / denominator

        # Из-за округлений точка не должна выходить за текущий отрезок.
        x_new = min(max(x_new, u), v)
        f_new = float(f(x_new))
        df_new = float(df(x_new))

        # Сохраняем не только ответ, но и промежуточные величины.
        # Это полезно для таблицы в отчете и для устной защиты.
        rows.append(
            {
                "k": k,
                "u": u,
                "v": v,
                "f_u": fu,
                "f_v": fv,
                "df_u": dfu,
                "df_v": dfv,
                "w": w,
                "z": z,
                "x_new": x_new,
                "f_new": f_new,
                "df_new": df_new,
                "interval_length": v - u,
            }
        )

        # Условие остановки из задания: точность eps = 0.0001.
        if abs(df_new) < eps:
            break

        # Если производная в новой точке положительна, минимум левее этой точки.
        # Если производная отрицательна, минимум правее этой точки.
        if df_new > 0:
            v = x_new
        else:
            u = x_new

    return rows


def main():
    # Если папки build еще нет, создаем ее.
    # В ней лежат машинные результаты, а не сам отчет.
    BUILD_DIR.mkdir(exist_ok=True)

    rows = cubic_approximation()
    answer = rows[-1]

    # JSON нужен, чтобы результаты можно было проверить независимо от PDF.
    result = {
        "task": "ДЗ1.1",
        "eps": EPS,
        "function": "f(x)=x^3/3-5x+x ln(x)",
        "segment": [A, B],
        "iterations": rows,
        "answer": {
            "x_min": answer["x_new"],
            "f_min": answer["f_new"],
            "abs_df": abs(answer["df_new"]),
        },
    }

    (BUILD_DIR / "dz1_1_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ДЗ1.1. Метод кубической аппроксимации")
    print("k | u | v | x_new | f(x_new) | f'(x_new)")
    for row in rows:
        print(
            f"{row['k']:2d} | {row['u']:.6f} | {row['v']:.6f} | "
            f"{row['x_new']:.9f} | {row['f_new']:.9f} | {row['df_new']:.3e}"
        )
    print(f"Ответ: x_min = {answer['x_new']:.9f}, f_min = {answer['f_new']:.9f}")


if __name__ == "__main__":
    main()
