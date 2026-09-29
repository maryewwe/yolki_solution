# Мониторинг строительной техники с RF-DETR

Система компьютерного зрения для детекции строительной техники на широкоугольных кадрах со строительных площадок и CCTV-камер.

Финальная модель — **RF-DETR Medium @640**, дополнительно адаптированная к целевому домену на 80 размеченных изображениях организаторов.

Репозиторий содержит:

- пайплайн подготовки внешнего датасета;
- метаданные, фиксирующие итоговый состав и split датасета;
- три ноутбука, образующие финальную цепочку обучения модели;
- отдельный эксперимент, показывающий эффект target-domain fine-tuning;
- код инференса;
- метрики обучения и итоговой оценки модели.

---

## Финальная модель

**Архитектура:** RF-DETR Medium  
**Разрешение:** 640×640  
**Режим инференса:** full-frame  
**Порог confidence по умолчанию:** 0.25  
**Количество классов:** 13

Финальные веса опубликованы в разделе **GitHub Releases** в составе:

```text
final80_deployment_bundle.zip
```

Содержимое deployment bundle:

```text
final_model.pth
inference_example.py
requirements.txt
taxonomy.yaml
deployment_meta.json
final80_manifest.csv
README_DEPLOY.txt
```

Большой файл `final_model.pth` не хранится напрямую в Git-истории.

---

## Классы

Модель использует единую 13-классовую taxonomy:

```text
0  excavator
1  dump_truck
2  truck
3  loader
4  bulldozer
5  motor_grader
6  roller
7  concrete_mixer
8  telehandler
9  piling_machine
10 crane_manipulator
11 mobile_crane
12 tower_crane
```

---

# Данные

## Исходные датасеты

Для обучения не объединялись все найденные открытые датасеты подряд. Были отобраны четыре источника, которые лучше всего соответствовали целевой taxonomy и визуальному домену задачи.

Основные критерии отбора:

- совместимость с итоговой 13-классовой taxonomy;
- наличие реальных сцен строительных площадок;
- близость к fixed-camera / wide-view изображениям;
- наличие редких классов техники;
- наличие small-object и high-angle примеров.

Использованные источники:

### 1. Construction Equipment

https://www.kaggle.com/datasets/xyzyxzzxy/construction-equipment

### 2. Construction Vehicle Detection

https://universe.roboflow.com/capstone-lkzgq/construction-vehicle-detection-pxc7c

### 3. Heavy Equipment

https://universe.roboflow.com/jspark-pulqw/230620_heavyequipment-mvbju

### 4. APOCE — Aerial Photographs for Object Detection of Construction Equipment

https://universe.roboflow.com/research-s0une/apoce-aerial-photographs-for-object-detection-of-construction-equipment

---

## Итоговый внешний датасет

Подготовленная версия доступна на Kaggle:

https://www.kaggle.com/datasets/grigoriyyy/lct-base-combined-dataset

Итоговый датасет `construction_v1`:

| Split | Количество изображений |
|---|---:|
| Train | 19 391 |
| Validation | 4 609 |
| Test | 1 102 |
| **Всего** | **25 102** |

Всего размечено:

```text
45 493 bounding boxes
13 classes
```

---

## Пайплайн подготовки датасета

Исходные датасеты не просто объединялись в одну папку.

Пайплайн включал:

1. приведение исходных классов к единой 13-классовой taxonomy;
2. удаление неоднозначных и несовместимых категорий;
3. нормализацию формата разметки;
4. проверку и clipping bounding boxes;
5. удаление некорректных аннотаций;
6. поиск точных дубликатов;
7. group-aware обработку split там, где структура исходного датасета позволяла восстановить связанные кадры;
8. аудит near-duplicate изображений между train / val / test;
9. сохранение и применение remediation-решений;
10. независимую финальную проверку собранного датасета.

Точный состав итогового split зафиксирован в:

```text
dataset/metadata/manifest.csv
```

Скрипты сборки и аудита находятся в:

```text
dataset/scripts/
```

Метаданные финальной версии находятся в:

```text
dataset/metadata/
```

---

# Обучение модели

Финальная модель получена последовательностью из трёх этапов:

```text
construction_v1
      ↓
RF-DETR Medium @576
      ↓
fine-tuning @640
      ↓
target-domain fine-tuning на 80 изображениях организаторов
      ↓
FINAL RF-DETR Medium @640
```

---

## Этап 1 — базовое обучение RF-DETR Medium @576

Ноутбук:

```text
notebooks/01_lct-rfdetr-medium-576.ipynb
```

На этом этапе RF-DETR Medium обучается на `construction_v1` при разрешении 576×576.

Метрики обучения:

```text
metrics/metrics_rfdetr_medium_576.csv
```

---

## Этап 2 — resolution fine-tuning @640

Ноутбук:

```text
notebooks/02_lct-rfdetr-medium-640-finetune.ipynb
```

Лучший checkpoint предыдущего этапа дообучается при разрешении 640×640.

Переход `576 → 640` использовался из-за большого количества небольшой и удалённой техники на широких CCTV-кадрах. Fine-tuning на большем разрешении позволяет сохранить уже выученные признаки и одновременно дать модели больше пространственной информации для локализации небольших объектов.

Лучший результат на внешней validation-выборке:

```text
mAP50-95 ≈ 0.6984
```

Метрики:

```text
metrics/metrics_rfdetr_medium_640_finetune.csv
```

---

## Этап 3 — target-domain fine-tuning

Ноутбук:

```text
notebooks/03_lct-final80-rfdetr-medium640.ipynb
```

После обучения на внешнем датасете модель дополнительно адаптируется к изображениям организаторов.

Для финального запуска были объединены:

```text
65 бывших TRAIN
+15 бывших VAL
=80 target-domain изображений
```

Финальная модель обучается заново от external Medium@640 checkpoint на всех 80 доступных development-изображениях организаторов.

Результирующий EMA checkpoint:

```text
final_model.pth
```

Поскольку прежние 15 validation-изображений входят в FINAL80 training, у этого финального запуска нет отдельной held-out validation-метрики.

---

# Влияние target-domain fine-tuning

Для оценки пользы дообучения на изображениях организаторов был проведён отдельный эксперимент.

Ноутбук:

```text
notebooks/04_lct-target-finetune-impact.ipynb
```

Сравнивались две модели с одинаковой архитектурой и одинаковым разрешением:

```text
BEFORE TARGET FT
RF-DETR Medium @640
обучен только на construction_v1

AFTER TARGET FT
тот же RF-DETR Medium @640
+ fine-tuning на 80 изображениях организаторов
```

Таким образом, изменение качества связано именно с адаптацией к целевому домену, а не со сменой архитектуры или разрешения.

Файл с итоговым сравнением:

```text
metrics/TARGET_FT_COMPARISON.csv
```

Результаты на фиксированном TEST split из 20 изображений:

| Метрика | До target FT | FINAL80 | Изменение |
|---|---:|---:|---:|
| **mAP50-95** | 0.4047 | **0.5050** | **+0.1002** |
| mAP50 | 0.6101 | **0.7441** | **+0.1339** |
| mAP75 | 0.4079 | **0.5237** | **+0.1158** |
| Precision @0.25 | **0.655** | 0.619 | -0.036 |
| **Recall @0.25** | 0.479 | **0.697** | **+0.218** |
| **F1 @0.25** | 0.553 | **0.656** | **+0.103** |
| TP | 57 | **83** | **+26** |
| FP | **30** | 51 | +21 |
| FN | 62 | **36** | **-26** |

Ключевой эффект target-domain fine-tuning:

```text
mAP50-95: 0.405 → 0.505
Recall:    47.9% → 69.7%
F1:        55.3% → 65.6%
```

Дообучение на небольшом количестве изображений целевого домена значительно увеличило recall и число корректно найденных объектов.

TEST split содержит только 20 изображений, поэтому эти значения рассматриваются как итоговая project-level оценка, а не как крупный внешний benchmark.

---

# Итоговая оценка FINAL80

Отдельно сравнивалась предыдущая target-domain модель, обученная на 65 organizer TRAIN изображениях, и финальная модель, обученная на всех 80 доступных development-изображениях.

При `confidence = 0.25`:

| Метрика | Train=65 | FINAL80 |
|---|---:|---:|
| mAP50-95 | 0.4997 | **0.5050** |
| Precision | 0.597 | **0.619** |
| Recall | 0.647 | **0.697** |
| F1 | 0.621 | **0.656** |
| TP | 77 | **83** |
| FP | 52 | **51** |
| FN | 42 | **36** |

Финальное обучение на всех 80 target-domain изображениях увеличило recall и F1 без роста количества false positives.

Файлы итоговой оценки находятся в:

```text
evaluation/
```

---

# Инференс

Установка зависимостей:

```bash
pip install -r requirements.txt
```

После скачивания `final80_deployment_bundle.zip` из GitHub Releases необходимо извлечь `final_model.pth`.

Запуск:

```bash
python inference.py path/to/image.jpg
```

Пример ответа:

```json
[
  {
    "class": "excavator",
    "confidence": 0.87,
    "bbox_xyxy": [120.4, 81.2, 411.8, 352.5]
  }
]
```

Параметры финального deployment:

```text
architecture: RF-DETR Medium
resolution:   640
threshold:    0.25
mode:         full-frame
```

Экспериментальные crop / tiled inference ветки в финальный production pipeline не входят.

---

# Скорость инференса

На GPU, использованном при финальной оценке в Kaggle:

```text
среднее model.predict(): ~40.9 ms / frame
медиана:                 ~39.2 ms / frame
throughput:              ~24–25 FPS
```

Здесь измеряется только время `model.predict()`.

Чтение изображения, передача данных по сети и application-level post-processing в это время не входят.

---

# Зависимости

Основные зависимости проекта:

```text
rfdetr[train]==1.10.1
torch==2.10.0
numpy
pandas
PyYAML
Pillow
```

Полный список:

```text
requirements.txt
```

Для обучения и быстрого инференса требуется CUDA-capable GPU.

---

# Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── inference.py
│
├── notebooks/
│   ├── 01_lct-rfdetr-medium-576.ipynb
│   ├── 02_lct-rfdetr-medium-640-finetune.ipynb
│   ├── 03_lct-final80-rfdetr-medium640.ipynb
│   └── 04_lct-target-finetune-impact.ipynb
│
├── metrics/
│   ├── metrics_rfdetr_medium_576.csv
│   ├── metrics_rfdetr_medium_640_finetune.csv
│   └── TARGET_FT_COMPARISON.csv
│
├── dataset/
│   ├── scripts/
│   │   ├── build_dataset.py
│   │   ├── near_duplicate_audit.py
│   │   ├── finalize_near_duplicate_review.py
│   │   ├── remediate_near_duplicate_groups.py
│   │   ├── independent_dataset_audit.py
│   │   ├── finalize_remediation_reports.py
│   │   └── create_gt_contact_sheets.py
│   │
│   └── metadata/
│       ├── manifest.csv
│       ├── class_mapping.yaml
│       ├── excluded.csv
│       ├── exact_duplicates.csv
│       ├── group_split_audit.csv
│       ├── raw_class_mapping_audit.csv
│       ├── visual_audit_notes.csv
│       ├── near_duplicate_audit/
│       └── near_duplicate_remediation/
│
└── evaluation/
    ├── FINAL_TEST_REPORT.md
    ├── TEST_COMPARISON.csv
    └── examples/
```

---

# Воспроизводимость

## Запуск готовой модели

```text
GitHub Releases
        ↓
final80_deployment_bundle.zip
        ↓
final_model.pth
        ↓
inference.py
```

---

## Воспроизведение обучения

```text
construction_v1
        ↓
01_lct-rfdetr-medium-576.ipynb
        ↓
02_lct-rfdetr-medium-640-finetune.ipynb
        ↓
03_lct-final80-rfdetr-medium640.ipynb
        ↓
final_model.pth
```

---

## Воспроизведение подготовки датасета

```text
D1 + D6 + D7 + D8
        ↓
dataset/scripts/
        ↓
единая 13-классовая taxonomy
        ↓
bbox validation / filtering
        ↓
duplicate & leakage audit
        ↓
stored remediation decisions
        ↓
construction_v1
```

Итоговый split и происхождение каждого изображения зафиксированы в `dataset/metadata/manifest.csv`.

---

# Исследовательские эксперименты

В ходе разработки дополнительно проверялись:

- альтернативные detector-архитектуры;
- RF-DETR Small;
- разные варианты YOLO;
- дополнительные continuation-запуски;
- crop / tiled inference;
- branch-aware fusion;
- различные resolution.

Эти эксперименты не входят в основную training lineage репозитория.

В репозитории сохранена только финальная воспроизводимая цепочка и эксперимент, необходимый для оценки эффекта target-domain adaptation.
