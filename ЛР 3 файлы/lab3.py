import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Вариант 15: f(x) = x^2 + e^(-x), [a, b] = [0, 1]

def f(x):
    return x**2 + np.exp(-x)

def df(x):
    """Первая производная: f'(x) = 2x - e^(-x)"""
    return 2*x - np.exp(-x)

def d2f(x):
    """Вторая производная: f''(x) = 2 + e^(-x)"""
    return 2 + np.exp(-x)

# ─────────────────────────────────────────────
# МЕТОД КВАДРАТИЧНОЙ АППРОКСИМАЦИИ
# ─────────────────────────────────────────────

def quadratic_approximation(a, b, eps=1e-4, max_iter=100):
    """
    Метод квадратичной аппроксимации (метод парабол).
    На каждой итерации берём три точки x1, x2, x3,
    строим интерполяционный полином Лагранжа 2-й степени
    и находим его минимум x*.
    """
    # Начальные три точки
    x1 = a
    x3 = b
    x2 = (a + b) / 2.0

    history = []  # для графиков

    print("=" * 60)
    print("МЕТОД КВАДРАТИЧНОЙ АППРОКСИМАЦИИ")
    print(f"f(x) = x² + e^(-x),  [a, b] = [{a}, {b}],  ε = {eps}")
    print("=" * 60)

    for iteration in range(1, max_iter + 1):
        f1, f2, f3 = f(x1), f(x2), f(x3)

        # Коэффициенты интерполяционного полинома Лагранжа
        # P(x) = a0 + a1*(x-x1) + a2*(x-x1)*(x-x2)
        # Минимум параболы: x* = 0.5 * (x1+x2 - a1/a2)
        # Через формулу Лагранжа:
        denom = (f1*(x2 - x3) + f2*(x3 - x1) + f3*(x1 - x2))

        if abs(denom) < 1e-15:
            print(f"  Итерация {iteration}: знаменатель ≈ 0, останов.")
            break

        x_star = 0.5 * (
            (x1**2 - x2**2) * f3 +
            (x3**2 - x1**2) * f2 +
            (x2**2 - x3**2) * f1
        ) / denom

        f_star = f(x_star)

        # Коэффициенты параболы для визуализации
        # P(x) = A*x^2 + B*x + C через три точки
        A_mat = np.array([
            [x1**2, x1, 1],
            [x2**2, x2, 1],
            [x3**2, x3, 1]
        ])
        b_vec = np.array([f1, f2, f3])
        try:
            coeffs = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            coeffs = None

        history.append({
            'iter': iteration,
            'x1': x1, 'x2': x2, 'x3': x3,
            'f1': f1, 'f2': f2, 'f3': f3,
            'x_star': x_star, 'f_star': f_star,
            'coeffs': coeffs
        })

        print(f"\n  Итерация {iteration}:")
        print(f"    x1={x1:.6f}, x2={x2:.6f}, x3={x3:.6f}")
        print(f"    f1={f1:.6f}, f2={f2:.6f}, f3={f3:.6f}")
        print(f"    x* = {x_star:.6f},  f(x*) = {f_star:.6f}")

        # Критерий останова
        if abs(x_star - x2) < eps:
            print(f"\n  Останов: |x* - x2| = {abs(x_star - x2):.2e} < ε = {eps}")
            print(f"  Результат: x_min ≈ {x_star:.6f},  f(x_min) ≈ {f_star:.6f}")
            break

        # Обновление точек: выбираем тройку вокруг x*
        # Определяем, в каком подотрезке лежит x*
        points = sorted([(x1, f1), (x2, f2), (x3, f3), (x_star, f_star)],
                        key=lambda p: p[0])
        # Берём три точки с наименьшими значениями f
        points_sorted_by_f = sorted(points, key=lambda p: p[1])
        best3 = sorted(points_sorted_by_f[:3], key=lambda p: p[0])
        x1, x2, x3 = best3[0][0], best3[1][0], best3[2][0]

    else:
        print(f"\n  Достигнуто максимальное число итераций ({max_iter}).")
        print(f"  Результат: x_min ≈ {x_star:.6f},  f(x_min) ≈ {f_star:.6f}")

    return x_star, f_star, history


# ─────────────────────────────────────────────
# МЕТОД КУБИЧЕСКОЙ АППРОКСИМАЦИИ
# ─────────────────────────────────────────────

def cubic_approximation(a, b, eps=1e-4, max_iter=100):
    """
    Метод кубической аппроксимации (метод Пауэлла/кубической интерполяции).
    Использует значения функции и производной в двух точках.
    """
    print("\n" + "=" * 60)
    print("МЕТОД КУБИЧЕСКОЙ АППРОКСИМАЦИИ")
    print(f"f(x) = x² + e^(-x),  [a, b] = [{a}, {b}],  ε = {eps}")
    print("=" * 60)

    history_cubic = []

    u = a
    v = b

    for iteration in range(1, max_iter + 1):
        fu, fv = f(u), f(v)
        dfu, dfv = df(u), df(v)

        # Формула кубической интерполяции (метод Пауэлла)
        w = dfv + dfu - 3 * (fv - fu) / (v - u)
        discriminant = w**2 - dfu * dfv

        if discriminant < 0:
            # Нет минимума на отрезке — берём середину
            x_star = (u + v) / 2.0
        else:
            z = np.sqrt(discriminant)
            x_star = v - (v - u) * (dfv + z - w) / (dfv - dfu + 2 * z)
            # Ограничиваем x* отрезком [u, v]
            x_star = np.clip(x_star, u, v)

        f_star = f(x_star)
        df_star = df(x_star)

        # Коэффициенты кубического полинома через u, v для визуализации
        # Используем кубический сплайн Эрмита
        h = v - u
        coeffs_cubic = None
        if abs(h) > 1e-15:
            # P(t) = fu*(2t^3-3t^2+1) + dfu*h*(t^3-2t^2+t)
            #      + fv*(-2t^3+3t^2) + dfv*h*(t^3-t^2),  t=(x-u)/h
            coeffs_cubic = (u, v, fu, fv, dfu, dfv)

        history_cubic.append({
            'iter': iteration,
            'u': u, 'v': v,
            'fu': fu, 'fv': fv,
            'x_star': x_star, 'f_star': f_star,
            'coeffs': coeffs_cubic
        })

        print(f"\n  Итерация {iteration}:")
        print(f"    u={u:.6f}, v={v:.6f}")
        print(f"    f(u)={fu:.6f}, f(v)={fv:.6f}")
        print(f"    f'(u)={dfu:.6f}, f'(v)={dfv:.6f}")
        print(f"    x* = {x_star:.6f},  f(x*) = {f_star:.6f},  f'(x*) = {df_star:.6f}")

        # Критерий останова
        if abs(df_star) < eps:
            print(f"\n  Останов: |f'(x*)| = {abs(df_star):.2e} < ε = {eps}")
            print(f"  Результат: x_min ≈ {x_star:.6f},  f(x_min) ≈ {f_star:.6f}")
            break

        # Обновление отрезка
        if df_star > 0:
            v = x_star
        else:
            u = x_star

    else:
        print(f"\n  Достигнуто максимальное число итераций ({max_iter}).")
        print(f"  Результат: x_min ≈ {x_star:.6f},  f(x_min) ≈ {f_star:.6f}")

    return x_star, f_star, history_cubic


# ─────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────

def plot_results(a, b, history_quad, history_cubic, x_min_q, x_min_c):
    x_plot = np.linspace(a - 0.05, b + 0.05, 500)
    y_plot = f(x_plot)

    # ── Рисунок 1: несколько итераций квадратичной аппроксимации ──
    n_show = min(4, len(history_quad))
    fig1, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    fig1.suptitle("Метод квадратичной аппроксимации\n"
                  r"$f(x) = x^2 + e^{-x}$, $[a,b]=[0,1]$, вариант 15",
                  fontsize=13)

    for idx, ax in enumerate(axes):
        h = history_quad[idx]
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=r'$f(x)$')

        # Парабола
        if h['coeffs'] is not None:
            A, B, C = h['coeffs']
            y_par = A * x_plot**2 + B * x_plot + C
            ax.plot(x_plot, y_par, 'r--', linewidth=1.5, label='Парабола')

        # Три точки
        for xi, fi, lbl in [(h['x1'], h['f1'], '$x_1$'),
                             (h['x2'], h['f2'], '$x_2$'),
                             (h['x3'], h['f3'], '$x_3$')]:
            ax.plot(xi, fi, 'go', markersize=7)
            ax.annotate(lbl, (xi, fi), textcoords="offset points",
                        xytext=(4, 6), fontsize=9, color='green')

        # x*
        ax.plot(h['x_star'], h['f_star'], 'r*', markersize=12, label=f"$x^*={h['x_star']:.4f}$")
        ax.axvline(h['x_star'], color='r', linestyle=':', alpha=0.5)

        ax.set_title(f"Итерация {h['iter']}", fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(fontsize=8)
        ax.set_xlim(a - 0.05, b + 0.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quad_iterations.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранён: quad_iterations.png")

    # ── Рисунок 2: несколько итераций кубической аппроксимации ──
    n_show_c = min(4, len(history_cubic))
    fig2, axes2 = plt.subplots(1, n_show_c, figsize=(5 * n_show_c, 5))
    if n_show_c == 1:
        axes2 = [axes2]

    fig2.suptitle("Метод кубической аппроксимации\n"
                  r"$f(x) = x^2 + e^{-x}$, $[a,b]=[0,1]$, вариант 15",
                  fontsize=13)

    for idx, ax in enumerate(axes2):
        h = history_cubic[idx]
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=r'$f(x)$')

        # Кубический полином Эрмита
        if h['coeffs'] is not None:
            u0, v0, fu0, fv0, dfu0, dfv0 = h['coeffs']
            t_arr = np.linspace(0, 1, 300)
            x_cub = u0 + t_arr * (v0 - u0)
            h_len = v0 - u0
            P = (fu0 * (2*t_arr**3 - 3*t_arr**2 + 1) +
                 dfu0 * h_len * (t_arr**3 - 2*t_arr**2 + t_arr) +
                 fv0 * (-2*t_arr**3 + 3*t_arr**2) +
                 dfv0 * h_len * (t_arr**3 - t_arr**2))
            ax.plot(x_cub, P, 'r--', linewidth=1.5, label='Кубика')

        # Точки u, v
        ax.plot(h['u'], h['fu'], 'go', markersize=8)
        ax.annotate('u', (h['u'], h['fu']), textcoords="offset points",
                    xytext=(4, 6), fontsize=9, color='green')
        ax.plot(h['v'], h['fv'], 'gs', markersize=8)
        ax.annotate('v', (h['v'], h['fv']), textcoords="offset points",
                    xytext=(4, 6), fontsize=9, color='green')

        # x*
        ax.plot(h['x_star'], h['f_star'], 'r*', markersize=12,
                label=f"$x^*={h['x_star']:.4f}$")
        ax.axvline(h['x_star'], color='r', linestyle=':', alpha=0.5)

        ax.set_title(f"Итерация {h['iter']}", fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(fontsize=8)
        ax.set_xlim(a - 0.05, b + 0.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cubic_iterations.png', dpi=150, bbox_inches='tight')
    print("График сохранён: cubic_iterations.png")

    # ── Рисунок 3: итоговое сравнение ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(x_plot, y_plot, 'b-', linewidth=2.5, label=r'$f(x) = x^2 + e^{-x}$')
    ax3.axvline(x_min_q, color='red', linestyle='--', linewidth=1.5,
                label=f'Квадр. апп.: $x^*={x_min_q:.5f}$')
    ax3.axvline(x_min_c, color='orange', linestyle='-.', linewidth=1.5,
                label=f'Кубич. апп.: $x^*={x_min_c:.5f}$')
    ax3.plot(x_min_q, f(x_min_q), 'r*', markersize=14)
    ax3.plot(x_min_c, f(x_min_c), 'o', color='orange', markersize=10)

    # Аналитический минимум: f'(x) = 2x - e^(-x) = 0
    from scipy.optimize import brentq
    x_exact = brentq(df, a, b)
    ax3.axvline(x_exact, color='green', linestyle=':', linewidth=1.5,
                label=f'Точный минимум: $x^*={x_exact:.5f}$')
    ax3.plot(x_exact, f(x_exact), 'g^', markersize=12)

    ax3.set_title("Сравнение методов аппроксимации\n"
                  r"$f(x) = x^2 + e^{-x}$, вариант 15", fontsize=13)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('f(x)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print("График сохранён: comparison.png")

    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    a, b = 0.0, 1.0
    eps = 1e-4

    x_min_q, f_min_q, hist_q = quadratic_approximation(a, b, eps)
    x_min_c, f_min_c, hist_c = cubic_approximation(a, b, eps)

    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 60)
    print(f"  Квадратичная аппроксимация: x* = {x_min_q:.6f},  f(x*) = {f_min_q:.6f}")
    print(f"  Кубическая аппроксимация:   x* = {x_min_c:.6f},  f(x*) = {f_min_c:.6f}")

    from scipy.optimize import brentq
    x_exact = brentq(df, a, b)
    print(f"  Точный минимум (scipy):     x* = {x_exact:.6f},  f(x*) = {f(x_exact):.6f}")

    plot_results(a, b, hist_q, hist_c, x_min_q, x_min_c)
