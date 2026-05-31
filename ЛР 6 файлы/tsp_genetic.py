import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Матрица расстояний, Вариант 9 ────────────────────────────────────────────
#           Г1  Г2  Г3  Г4  Г5
dist = [
    [ 0,  6, 11,  5,  2],  # Город 1
    [ 6,  0,  3,  6,  2],  # Город 2
    [11,  3,  0,  9,  1],  # Город 3
    [ 5,  6,  9,  0,  5],  # Город 4
    [ 2,  2,  1,  5,  0],  # Город 5
]

N_CITIES = 5
CITY_NAMES = ['Г1', 'Г2', 'Г3', 'Г4', 'Г5']

# ── Параметры алгоритма ───────────────────────────────────────────────────────
POP_SIZE       = 4      # размер популяции N = 4
N_GENERATIONS  = 200    # число поколений
CROSSOVER_PROB = 0.9    # вероятность скрещивания
MUTATION_PROB  = 0.01   # вероятность мутации

# ── Целевая функция: длина маршрута (возврат в начало включён) ────────────────
def route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += dist[route[i]][route[i + 1]]
    length += dist[route[-1]][route[0]]
    return length

# ── Приспособленность: f_i = 1 / L_i ─────────────────────────────────────────
def fitness(route):
    return 1.0 / route_length(route)

# ── Отбор пар по убыванию вероятности рулетки ────────────────────────────────
# Пара 1: особь с наибольшей P + вторая по P
# Пара 2: третья + четвёртая
def select_pairs(population):
    f_values = [fitness(r) for r in population]
    F = sum(f_values)
    probs = [f / F for f in f_values]

    # Сортируем индексы по убыванию вероятности
    sorted_indices = sorted(range(len(population)), key=lambda i: probs[i], reverse=True)

    pair1 = (population[sorted_indices[0]], population[sorted_indices[1]])
    pair2 = (population[sorted_indices[2]], population[sorted_indices[3]])

    return pair1, pair2, probs, sorted_indices

# ── Скрещивание (двухточечный кроссовер из лекции) ───────────────────────────
# Структура: X | X X X | X  (1 снаружи | 3 в центре | 1 снаружи)
# Потомок A: центр от Р2, остаток из Р1 начиная со 2-го числа центра Р1
# Потомок B: центр от Р1, остаток из Р2 начиная со 2-го числа центра Р2
def crossover(parent1, parent2):
    # Две точки разрыва выбираются случайно, делят геном на три части:
    #   левая | центр | правая
    #
    # Город 1 фиксирован (всегда первый), в геноме только 4 промежуточных города.
    # Полный маршрут: 1 - [геном] - 1
    #
    # Примеры возможных разбиений генома из 4 элементов:
    #   pt1=0, pt2=2  →  | x x | x x
    #   pt1=0, pt2=3  →  | x x x | x
    #   pt1=1, pt2=3  →  x | x x | x
    #   pt1=1, pt2=4  →  x | x x x |
    #   pt1=2, pt2=4  →  x x | x x |
    #
    # Потомок A: центр от Р2, остаток из Р1 начиная со 2-го числа центра Р1
    # Потомок B: центр от Р1, остаток из Р2 начиная со 2-го числа центра Р2

    p1 = parent1[1:]   # убираем первый фиксированный город
    p2 = parent2[1:]
    n  = len(p1)       # = 4

    # Случайно выбираем две точки разрыва
    pt1 = np.random.randint(0, n - 1)        # от 0 до n-2
    pt2 = np.random.randint(pt1 + 1, n + 1)  # от pt1+1 до n

    def show_split(genome, label):
        # Наглядно показываем разбиение с палками
        left   = [str(c+1) for c in genome[:pt1]]
        middle = [str(c+1) for c in genome[pt1:pt2]]
        right  = [str(c+1) for c in genome[pt2:]]
        parts = []
        if left:
            parts.append(' '.join(left))
        parts.append('| ' + ' '.join(middle) + ' |')
        if right:
            parts.append(' '.join(right))
        print(f"    {label}: 1 - {' - '.join(parts)} - 1")

    show_split(p1, 'Р1')
    show_split(p2, 'Р2')

    def make_child(main, donor):
        # Центральный фрагмент берём из donor
        middle = donor[pt1:pt2]

        # Начинаем со 2-го числа центра main (индекс pt1+1), идём по кругу
        start = pt1 + 1
        order = []
        for i in range(n):
            order.append(main[(start + i) % n])

        # пропускаем уже взятые города
        remaining = []
        for city in order:
            if city not in middle:
                remaining.append(city)

        # Собираем: левая + центр + правая
        child_middle = remaining[:pt1] + middle + remaining[pt1:]
        return [parent1[0]] + child_middle

    child_a = make_child(p1, p2)
    child_b = make_child(p2, p1)

    return child_a, child_b, pt1, pt2


# ── Мутация: случайная перестановка двух городов ─────────────────────────────
def mutate(route):
    if np.random.random() < MUTATION_PROB:
        i, j = np.random.choice(range(1, len(route)), size=2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route

# ── Инициализация начальной популяции ────────────────────────────────────────
np.random.seed(42)
population = []
for _ in range(POP_SIZE):
    middle = list(np.random.permutation(range(1, N_CITIES)))
    route = [0] + middle
    population.append(route)

# ── Вывод начальной популяции ─────────────────────────────────────────────────
print("=" * 60)
print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ — Задача коммивояжера, Вариант 9")
print("=" * 60)
print(f"\nПараметры: N={POP_SIZE}, мутация={MUTATION_PROB}, скрещивание={CROSSOVER_PROB}")
print("\n── Шаг 1: Начальная популяция (Поколение 0) ──────────────")
for k, route in enumerate(population):
    names = [str(c + 1) for c in route] + ['1']
    L = route_length(route)
    print(f"  Особь P{k+1}: {'-'.join(names)}   L = {L}")

# ── Главный цикл ──────────────────────────────────────────────────────────────
best_route   = None
best_length  = float('inf')
best_per_gen = []

for generation in range(N_GENERATIONS):

    lengths  = [route_length(r) for r in population]
    f_values = [1.0 / L for L in lengths]
    F        = sum(f_values)
    probs    = [f / F for f in f_values]

    for i, route in enumerate(population):
        if lengths[i] < best_length:
            best_length = lengths[i]
            best_route  = route[:]

    best_per_gen.append(best_length)

    # Подробный лог только для первого поколения
    if generation == 0:
        print("\n── Шаг 2: Целевая функция и приспособленность ────────────")
        for k in range(POP_SIZE):
            print(f"  L(P{k+1}) = {lengths[k]}   "
                  f"f{k+1} = 1/{lengths[k]} = {f_values[k]:.4f}")
        print(f"  Суммарная приспособленность F = {F:.4f}")

        print("\n── Шаг 3: Вероятности рулетки (P_i = f_i / F) ───────────")
        for k in range(POP_SIZE):
            marker = "  ← лучшая" if lengths[k] == min(lengths) else ""
            print(f"  P{k+1} = {probs[k]*100:.1f}%{marker}")

        (p1, p2), (p3, p4), rprobs, sidx = select_pairs(population)
        print("\n── Шаг 4: Формирование пар (по убыванию вероятности) ─────")
        print(f"  Пара 1: P{sidx[0]+1} ({rprobs[sidx[0]]*100:.1f}%) + P{sidx[1]+1} ({rprobs[sidx[1]]*100:.1f}%)")
        print(f"  Пара 2: P{sidx[2]+1} ({rprobs[sidx[2]]*100:.1f}%) + P{sidx[3]+1} ({rprobs[sidx[3]]*100:.1f}%)")

        print("\n── Шаг 5: Скрещивание (пара 1) ───────────────────────────")
        demo_a, demo_b, pt1, pt2 = crossover(p1, p2)
        print(f"  Точки разрыва: pt1={pt1}, pt2={pt2}  →  центр = позиции [{pt1}:{pt2}]")
        print(f"  Потомок A:  {'-'.join([str(c+1) for c in demo_a])}-1  — центр от Р2, остаток из Р1")
        print(f"  Потомок B:  {'-'.join([str(c+1) for c in demo_b])}-1  — центр от Р1, остаток из Р2")

        print("\n── Шаг 5: Скрещивание (пара 2) ───────────────────────────")
        demo_c, demo_d, pt1, pt2 = crossover(p3, p4)
        print(f"  Точки разрыва: pt1={pt1}, pt2={pt2}  →  центр = позиции [{pt1}:{pt2}]")
        print(f"  Потомок C:  {'-'.join([str(c+1) for c in demo_c])}-1  — центр от Р2, остаток из Р1")
        print(f"  Потомок D:  {'-'.join([str(c+1) for c in demo_d])}-1  — центр от Р1, остаток из Р2")

    # ── Скрещивание — расширяем популяцию (родители + потомки) ──────────────
    extended = population[:]

    (p1, p2), (p3, p4), _, _ = select_pairs(population)

    for parent1, parent2 in [(p1, p2), (p3, p4)]:
        if np.random.random() < CROSSOVER_PROB:
            child_a, child_b, _, _ = crossover(parent1, parent2)
        else:
            child_a, child_b = parent1[:], parent2[:]

        child_a = mutate(child_a)
        child_b = mutate(child_b)

        extended.append(child_a)
        extended.append(child_b)

    # ── Редукция — удаляем худших, оставляем POP_SIZE лучших ─────────────────
    extended.sort(key=route_length)
    population = extended[:POP_SIZE]

# ── Итог ──────────────────────────────────────────────────────────────────────
best_names = [str(c + 1) for c in best_route] + ['1']
print("\n── Итог ──────────────────────────────────────────────────")
print(f"  Лучший маршрут: {' -> '.join(best_names)}")
print(f"  Длина маршрута: {best_length}")
print("=" * 60)

# ── Визуализация 1: маршрут на графе ─────────────────────────────────────────
angles   = [2 * np.pi * i / N_CITIES for i in range(N_CITIES)]
city_pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N_CITIES)}

fig1, ax = plt.subplots(figsize=(9, 9))
fig1.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_title('Задача коммивояжера — лучший маршрут (Вариант 9)',
             fontsize=14, color='white', pad=15)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

for i in range(N_CITIES):
    for j in range(i + 1, N_CITIES):
        xi, yi = city_pos[i]
        xj, yj = city_pos[j]
        ax.plot([xi, xj], [yi, yj], color='#778ca3', linewidth=1, alpha=0.25, zorder=1)
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        ax.text(mx, my, str(dist[i][j]), fontsize=8, color='#aab4c4',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#1a1a2e', ec='none'))

for step in range(N_CITIES):
    u = best_route[step]
    v = best_route[(step + 1) % N_CITIES]
    xu, yu = city_pos[u]
    xv, yv = city_pos[v]
    ax.annotate('', xy=(xv, yv), xytext=(xu, yu),
                arrowprops=dict(arrowstyle='->', color='#ff4757',
                                lw=2.5, mutation_scale=20), zorder=3)

NODE_RADIUS = 0.13
for idx in range(N_CITIES):
    x, y = city_pos[idx]
    fill_color = '#2ecc71' if idx == best_route[0] else '#3d5a80'
    edge_color = '#27ae60' if idx == best_route[0] else '#2c3e6b'

    ax.add_patch(plt.Circle((x + 0.01, y - 0.01), NODE_RADIUS,
                              color='black', alpha=0.3, zorder=4))
    ax.add_patch(plt.Circle((x, y), NODE_RADIUS + 0.015,
                              color=edge_color, zorder=5))
    ax.add_patch(plt.Circle((x, y), NODE_RADIUS, color=fill_color, zorder=6))
    ax.text(x, y, str(idx + 1), ha='center', va='center',
            fontsize=15, fontweight='bold', color='white', zorder=7)

legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Начальный город'),
    mpatches.Patch(color='#3d5a80', label='Остальные города'),
    plt.Line2D([0],[0], color='#ff4757', linewidth=2.5,
               label=f'Маршрут (длина = {best_length})'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
          facecolor='#2d3561', edgecolor='none', labelcolor='white')

plt.tight_layout()
plt.savefig('graph_tsp.png', dpi=150, bbox_inches='tight',
            facecolor=fig1.get_facecolor())
plt.show()

# ── Визуализация 2: сходимость ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')
ax2.set_title('Сходимость: лучшая длина маршрута по поколениям',
              fontsize=14, color='white', pad=15)
ax2.plot(best_per_gen, color='#3498db', linewidth=2.5)
ax2.axhline(y=best_length, color='#ff4757', linestyle='--', linewidth=1.8,
            label=f'Оптимум = {best_length}')
ax2.set_xlabel('Поколение', color='white', fontsize=12)
ax2.set_ylabel('Длина лучшего маршрута', color='white', fontsize=12)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#778ca3')
ax2.spines['left'].set_color('#778ca3')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=11, facecolor='#2d3561', edgecolor='none', labelcolor='white')
ax2.grid(True, alpha=0.2, color='white')

plt.tight_layout()
plt.savefig('convergence_tsp.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.show()
