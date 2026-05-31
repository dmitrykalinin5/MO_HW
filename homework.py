from pathlib import Path
import json
import os
import runpy


# Этот файл нужен только для удобного общего запуска.
# Каждое домашнее задание лежит в отдельной программе:
#   dz1_1_cubic_approximation.py
#   dz1_2_visualization.py
#   dz2_1_contours.py
#   dz2_2_newton.py
#   dz2_3_object_model.py

SCRIPTS = [
    "dz1_1_cubic_approximation.py",
    "dz1_2_visualization.py",
    "dz2_1_contours.py",
    "dz2_2_newton.py",
    "dz2_3_object_model.py",
]

RESULT_FILES = [
    ("dz1_1", "build/dz1_1_results.json"),
    ("dz2_1", "build/dz2_1_results.json"),
    ("dz2_2", "build/dz2_2_results.json"),
    ("dz2_3", "build/dz2_3_results.json"),
]


def main():
    base_dir = Path(__file__).resolve().parent

    # На всякий случай переходим в папку проекта.
    # Тогда скрипт можно запускать из любого текущего каталога.
    os.chdir(base_dir)

    for script in SCRIPTS:
        # runpy запускает файл так, будто мы написали:
        # python3 <имя_файла.py>
        # Это позволяет оставить каждую домашку самостоятельной программой.
        print(f"\n=== Запуск {script} ===")
        runpy.run_path(str(base_dir / script), run_name="__main__")

    # Для отчета удобно иметь одну сводку, поэтому объединяем JSON-файлы.
    summary = {}
    for key, filename in RESULT_FILES:
        path = base_dir / filename
        if path.exists():
            summary[key] = json.loads(path.read_text(encoding="utf-8"))

    output = base_dir / "build" / "results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОбщая сводка сохранена: {output}")


if __name__ == "__main__":
    main()
