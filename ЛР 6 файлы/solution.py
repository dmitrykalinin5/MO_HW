import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Граф (матрица расстояний) ─────────────────────────────────────────────────
#        A    B    C    D    E    F    G
INF = 9999  # нет прямого пути между узлами
dist = [
    [INF,  4,  14, INF,  4, INF, INF],  # A
    [  4, INF, INF, INF, INF, INF,  32],  # B
    [ 14, INF, INF, INF, INF, INF, INF],  # C
    [INF, INF, INF, INF,   7,   6,  18],  # D
    [  4, INF, INF,   7, INF, INF, INF],  # E
    [INF, INF, INF,   6, INF, INF,   6],  # F
    [INF,  32, INF,  18, INF,   6, INF],  # G
]

NODE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
N_NODES = 7
START = 0  # A
END   = 6  # G

# Координаты узлов для рисования графа
pos = {
    0: (0.0,  0.5),   # A
    1: (0.5,  1.1),   # B
    2: (0.5, -0.1),   # C
    3: (0.9,  0.5),   # D
    4: (0.3,  0.5),   # E
    5: (1.3,  0.5),   # F
    6: (1.8,  0.5),   # G
}

# ── Параметры алгоритма ───────────────────────────────────────────────────────
N_ANTS         = 20    # сколько муравьёв за одну итерацию
N_ITERATIONS   = 100   # сколько итераций всего
ALPHA          = 1.0   # насколько важен феромон при выборе пути
BETA           = 2.0   # насколько важна длина ребра при выборе пути
RHO            = 0.3   # скорость испарения феромона (0..1)
Q              = 10.0  # сколько феромона оставляет муравей
INIT_PHEROMONE = 0.1   # начальное количество феромона на всех рёбрах

# ── Инициализация феромона ────────────────────────────────────────────────────
# Создаём таблицу N_NODES x N_NODES, заполненную начальным феромоном
pheromone = []
for i in range(N_NODES):
    row = []
    for j in range(N_NODES):
        row.append(INIT_PHEROMONE)
    pheromone.append(row)

# ── Один шаг муравья ──────────────────────────────────────────────────────────
def ant_walk():
    path    = [START]   # путь муравья, начинаем из A
    visited = [START]   # уже посещённые узлы
    current = START

    while current != END:
        # Собираем список соседей, куда можно пойти
        neighbors = []
        for j in range(N_NODES):
            if dist[current][j] != INF and j not in visited:
                neighbors.append(j)

        # Если некуда идти — тупик, путь не засчитывается
        if len(neighbors) == 0:
            return None

        # Считаем "привлекательность" каждого соседа:
        # чем больше феромона и чем короче ребро — тем лучше
        scores = []
        for j in neighbors:
            ph   = pheromone[current][j]          # феромон на ребре
            edge = dist[current][j]               # длина ребра
            attractiveness = (ph ** ALPHA) * ((1.0 / edge) ** BETA)
            scores.append(attractiveness)

        # Считаем сумму для нормировки
        total = 0
        for s in scores:
            total += s

        # Вероятность выбора каждого соседа
        probs = []
        for s in scores:
            probs.append(s / total)

        # Выбираем следующий узел случайно по вероятностям
        next_node = np.random.choice(neighbors, p=probs)

        path.append(next_node)
        visited.append(next_node)
        current = next_node

    # Считаем длину пути
    length = 0
    for i in range(len(path) - 1):
        length += dist[path[i]][path[i+1]]

    return path, length

# ── Главный цикл ACO ──────────────────────────────────────────────────────────
best_path     = None
best_length   = INF
best_per_iter = []  # для графика сходимости

for iteration in range(N_ITERATIONS):
    all_paths   = []
    all_lengths = []

    # Запускаем всех муравьёв
    for ant in range(N_ANTS):
        result = ant_walk()
        if result is not None:
            path, length = result
            all_paths.append(path)
            all_lengths.append(length)
            # Запоминаем лучший путь
            if length < best_length:
                best_length = length
                best_path   = path

    # Испарение феромона на всех рёбрах
    for i in range(N_NODES):
        for j in range(N_NODES):
            pheromone[i][j] = pheromone[i][j] * (1 - RHO)

    # Добавляем феромон на рёбра, по которым прошли муравьи
    # Чем короче путь — тем больше феромона оставляет муравей
    for k in range(len(all_paths)):
        path   = all_paths[k]
        length = all_lengths[k]
        delta  = Q / length  # вклад феромона
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            pheromone[u][v] += delta
            pheromone[v][u] += delta  # граф неориентированный

    best_per_iter.append(best_length)

# Вывод результата
best_path_names = []
for n in best_path:
    best_path_names.append(NODE_NAMES[n])

print("Лучший путь:", ' -> '.join(best_path_names))
print("Длина пути: ", best_length)

# ── Визуализация 1: граф ──────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 8))
fig1.patch.set_facecolor('#1a1a2e')

ax.set_facecolor('#1a1a2e')
ax.set_title('Граф: феромон на рёбрах и лучший путь', fontsize=15, color='white', pad=15)
ax.set_xlim(-0.3, 2.1)
ax.set_ylim(-0.4, 1.5)
ax.axis('off')

best_edges = []
for i in range(len(best_path) - 1):
    best_edges.append((best_path[i], best_path[i+1]))

max_ph = 0
for i in range(N_NODES):
    for j in range(N_NODES):
        if pheromone[i][j] > max_ph:
            max_ph = pheromone[i][j]

# Рисуем рёбра
for i in range(N_NODES):
    for j in range(i + 1, N_NODES):
        if dist[i][j] == INF:
            continue

        xi, yi = pos[i]
        xj, yj = pos[j]
        ph = pheromone[i][j]
        is_best = (i, j) in best_edges or (j, i) in best_edges

        if is_best:
            # Лучший путь: яркая красная линия с тенью
            ax.plot([xi, xj], [yi, yj], color='#ff6b6b', linewidth=8,
                    alpha=0.3, zorder=2, solid_capstyle='round')
            ax.plot([xi, xj], [yi, yj], color='#ff4757', linewidth=4,
                    zorder=3, solid_capstyle='round')
        else:
            # Остальные рёбра: толщина зависит от феромона
            width = 1.5 + 4 * (ph / max_ph)
            alpha = 0.3 + 0.5 * (ph / max_ph)
            ax.plot([xi, xj], [yi, yj], color='#778ca3', linewidth=width,
                    alpha=alpha, zorder=1, solid_capstyle='round')

        # Подпись веса ребра (смещаем чуть в сторону от линии)
        mx = (xi + xj) / 2
        my = (yi + yj) / 2
        dx = yj - yi
        dy = -(xj - xi)
        norm = (dx**2 + dy**2) ** 0.5 + 1e-9
        offset = 0.045
        mx += dx / norm * offset
        my += dy / norm * offset
        ax.text(mx, my, str(int(dist[i][j])), fontsize=13, fontweight='bold',
                ha='center', va='center', color='white',
                bbox=dict(boxstyle='round,pad=0.3', fc='#2d3561', ec='none', alpha=0.85))

# Рисуем узлы
NODE_RADIUS = 0.09
for idx in range(N_NODES):
    x, y = pos[idx]

    if idx == START:
        fill_color = '#2ecc71'
        edge_color = '#27ae60'
    elif idx == END:
        fill_color = '#ff4757'
        edge_color = '#c0392b'
    elif idx in best_path:
        fill_color = '#f39c12'
        edge_color = '#e67e22'
    else:
        fill_color = '#3d5a80'
        edge_color = '#2c3e6b'

    # Тень узла
    shadow = plt.Circle((x + 0.008, y - 0.008), NODE_RADIUS,
                         color='black', alpha=0.3, zorder=4)
    ax.add_patch(shadow)
    # Обводка
    border = plt.Circle((x, y), NODE_RADIUS + 0.012,
                         color=edge_color, zorder=5)
    ax.add_patch(border)
    # Заливка
    circle = plt.Circle((x, y), NODE_RADIUS, color=fill_color, zorder=6)
    ax.add_patch(circle)
    # Буква
    ax.text(x, y, NODE_NAMES[idx], ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', zorder=7)

# Легенда
legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Старт (A)'),
    mpatches.Patch(color='#ff4757', label='Финиш (G)'),
    mpatches.Patch(color='#f39c12', label='Узел на лучшем пути'),
    mpatches.Patch(color='#3d5a80', label='Остальные узлы'),
    plt.Line2D([0],[0], color='#ff4757', linewidth=3, label='Лучший путь'),
    plt.Line2D([0],[0], color='#778ca3', linewidth=2, label='Остальные рёбра (толщина ~ феромон)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
          facecolor='#2d3561', edgecolor='none', labelcolor='white')

plt.tight_layout(pad=2.0)
plt.savefig('graph_aco.png', dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show()

# ── Визуализация 2: сходимость ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_title('Сходимость: лучшая длина пути по итерациям', fontsize=15, color='white', pad=15)
ax2.plot(best_per_iter, color='#3498db', linewidth=2.5)
ax2.axhline(y=best_length, color='#ff4757', linestyle='--', linewidth=1.8,
            label=f'Оптимум = {int(best_length)}')
ax2.set_xlabel('Итерация', color='white', fontsize=12)
ax2.set_ylabel('Длина лучшего пути', color='white', fontsize=12)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#778ca3')
ax2.spines['left'].set_color('#778ca3')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=11, facecolor='#2d3561', edgecolor='none', labelcolor='white')
ax2.grid(True, alpha=0.2, color='white')

plt.tight_layout(pad=2.0)
plt.savefig('convergence_aco.png', dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.show()
