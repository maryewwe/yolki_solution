# Frontend API: construction monitoring demo

Локальный base URL: `http://127.0.0.1:8000`

Swagger/OpenAPI после запуска: `http://127.0.0.1:8000/docs`

## Запуск backend

```bash
python3 -m pip install -r requirements-backend.txt
CV_PROVIDER=fixture \
CV_FIXTURE_PATH=backend/demo_detections.json \
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

`demo_detections.json` содержит synthetic detections для локальной проверки image
upload. Если `CV_FIXTURE_PATH` не задан, FixtureProvider возвращает пустой список;
это валидный анализ, а не ошибка сервера.

## GET /health

Проверяет доступность backend и показывает активный CV provider.

```json
{
  "status": "ok",
  "cv_provider": "fixture",
  "cv_ready": true,
  "confidence_threshold": 0.25
}
```

## GET /works

Возвращает допустимые значения `planned_work` для dropdown. Отправлять обратно
нужно значение `planned_work` без изменений. `active_conditions` содержит доступные
для этой работы явные переключатели; backend ничего не выводит эвристически.

```json
{
  "count": 359,
  "works": [
    {
      "planned_work": "Устройство котлована",
      "work_group": "Земляные работы",
      "active_conditions": ["для планировки/зачистки"]
    }
  ]
}
```

## POST /analyze: готовые detections

Используйте `Content-Type: application/json`, если detections уже получены от CV.
`timestamp` optional: при отсутствии backend подставит текущее UTC-время.
`camera_id` optional, default `demo-camera`. `zone_id` нужно опустить, если зона
неизвестна. Пустой массив `detections` разрешен.

```json
{
  "planned_work": "Устройство котлована",
  "active_conditions": {},
  "timestamp": "2026-09-28T12:00:00+03:00",
  "camera_id": "demo-camera",
  "zone_id": "pit-a",
  "detections": [
    {
      "class": "excavator",
      "confidence": 0.93,
      "bbox": [100, 80, 420, 360]
    },
    {
      "class": "dump_truck",
      "confidence": 0.88,
      "bbox": [440, 120, 760, 390]
    }
  ]
}
```

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "planned_work": "Устройство котлована",
    "active_conditions": {},
    "camera_id": "demo-camera",
    "detections": [
      {"class":"excavator","confidence":0.93,"bbox":[100,80,420,360]},
      {"class":"dump_truck","confidence":0.88,"bbox":[440,120,760,390]}
    ]
  }'
```

## POST /analyze: изображение

Используйте `multipart/form-data`. Поле `image` содержит файл, а
`active_conditions` передается JSON-строкой. В image mode detections дает текущий
CV provider.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F 'image=@/absolute/path/frame.jpg' \
  -F 'planned_work=Устройство котлована' \
  -F 'active_conditions={}' \
  -F 'camera_id=demo-camera' \
  -F 'zone_id=pit-a'
```

## POST /analyze-series: серия кадров

Это основной endpoint для demo. Он принимает несколько изображений одной камеры,
последовательно запускает один и тот же CV provider для каждого кадра, получает
instant result, а затем передает все observations в существующий temporal matcher.

`Content-Type`: `multipart/form-data`.

| Form field | Тип | Обязательное | Описание |
|---|---|---|---|
| `files` | повторяемый file | да | Изображения в порядке серии. Поле называется `files` для каждого файла. |
| `planned_work` | string | да | Точное значение из `GET /works`. |
| `camera_id` | string | да | Одна камера для всей серии. |
| `zone_id` | string | нет | Одна зона для всей серии; при неизвестной зоне поле не отправлять. |
| `active_conditions` | JSON string | нет | Объект boolean-переключателей, default `{}`. |
| `timestamps` | JSON string | нет | Массив уникальных ISO 8601 дат с timezone, ровно по одной на файл. |

Если `timestamps` не переданы, backend сохраняет upload order и создает
последовательные UTC timestamps с шагом одна секунда. В ответе это отмечено как
`metadata.timestamps_source = "generated_upload_order"`. EXIF не используется.

Лимит по умолчанию: **30 кадров** на один request. Его можно изменить на backend
через `MAX_SERIES_FRAMES`; фактически примененный лимит возвращается в
`metadata.max_series_frames`.

### curl

```bash
curl -X POST http://127.0.0.1:8000/analyze-series \
  -F 'files=@/absolute/path/frame_001.jpg' \
  -F 'files=@/absolute/path/frame_002.jpg' \
  -F 'files=@/absolute/path/frame_003.jpg' \
  -F 'planned_work=Устройство котлована' \
  -F 'camera_id=demo-camera' \
  -F 'zone_id=pit-a' \
  -F 'active_conditions={}' \
  -F 'timestamps=["2026-09-29T12:00:00+03:00","2026-09-29T12:00:05+03:00","2026-09-29T12:00:10+03:00"]'
```

### JavaScript FormData

```javascript
const form = new FormData();
for (const file of selectedFiles) {
  form.append("files", file);
}
form.append("planned_work", selectedWork);
form.append("camera_id", "demo-camera");
form.append("active_conditions", JSON.stringify(activeConditions));

// Optional. The array length must equal selectedFiles.length.
form.append("timestamps", JSON.stringify(frameTimestamps));

const response = await fetch("http://127.0.0.1:8000/analyze-series", {
  method: "POST",
  body: form,
});
const result = await response.json();
if (!response.ok) throw new Error(JSON.stringify(result));
```

### Response /analyze-series

- итог серии: `series_summary.persistent_status`;
- итоговое объяснение: `series_summary.explanation`;
- persistent missing/uncertain: `series_summary.temporal.persistent_missing` и
  `series_summary.temporal.persistent_uncertain`;
- кадры, подтверждающие persistent item: `frame_indices` внутри этого item;
- instant status кадра: `frames[index].instant_status`;
- bbox кадра: `frames[index].detections.accepted[].bbox`;
- class/confidence кадра: соседние поля `class` и `confidence`;
- instant missing/uncertain: `frames[index].match.missing` и `.uncertain`;
- порядок исходных файлов: `frames[index].index`, `.filename`, `.timestamp`;
- timings: `metadata.timings`.

Ответ использует существующие структуры matcher, поэтому `missing` и `uncertain`
не дублируются рядом с `match`: читать нужно именно `match.missing` и
`match.uncertain`. Ниже полный по уровням контрактный пример; массивы requirements
могут содержать больше элементов в зависимости от выбранной работы.

```json
{
  "series_summary": {
    "planned_work": "Устройство котлована",
    "work_group": "Земляные работы",
    "profile_id": 15,
    "status": "OK",
    "instant_status": "OK",
    "persistent_status": "OK",
    "observation_count": 3,
    "observability": {"score": 1.0, "level": "HIGH"},
    "match": {
      "score": 1.0,
      "coverage": 1.0,
      "confirmed_count": 2,
      "missing_count": 0,
      "uncertain_count": 0,
      "assessed_count": 2,
      "satisfied": [
        {"id": "required:excavator", "type": "required", "classes": ["excavator"], "display_names": ["экскаватор"], "found_classes": ["excavator"], "found_display_names": ["экскаватор"], "source_terms": []},
        {"id": "required:dump_truck", "type": "required", "classes": ["dump_truck"], "display_names": ["автосамосвал"], "found_classes": ["dump_truck"], "found_display_names": ["автосамосвал"], "source_terms": []}
      ],
      "missing": [],
      "uncertain": [],
      "supporting": []
    },
    "detections": {
      "accepted": [
        {"class": "excavator", "display_name": "экскаватор", "confidence": 0.93, "bbox": [100.0, 80.0, 420.0, 360.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true},
        {"class": "dump_truck", "display_name": "автосамосвал", "confidence": 0.88, "bbox": [440.0, 120.0, 760.0, 390.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}
      ],
      "rejected_low_confidence": [],
      "rejected_unknown_class": [],
      "rejected_zone_mismatch": [],
      "rejected_invalid": []
    },
    "possible_unexpected": [],
    "evidence": [],
    "metadata": {"timestamp": "2026-09-29T12:00:10+03:00", "camera_id": "demo-camera", "zone_id": "pit-a", "target_zone_id": "pit-a"},
    "explanation": "Устойчивых отклонений в выбранном окне не найдено.",
    "temporal": {
      "window_seconds": null,
      "min_observations": 3,
      "persistence_ratio": 0.75,
      "persistent_missing": [],
      "persistent_uncertain": []
    },
    "observations": [
      {"timestamp": "2026-09-29T12:00:00+03:00", "camera_id": "demo-camera", "zone_id": "pit-a", "instant_status": "OK", "match_score": 1.0, "missing_requirement_ids": [], "uncertain_requirement_ids": []},
      {"timestamp": "2026-09-29T12:00:05+03:00", "camera_id": "demo-camera", "zone_id": "pit-a", "instant_status": "OK", "match_score": 1.0, "missing_requirement_ids": [], "uncertain_requirement_ids": []},
      {"timestamp": "2026-09-29T12:00:10+03:00", "camera_id": "demo-camera", "zone_id": "pit-a", "instant_status": "OK", "match_score": 1.0, "missing_requirement_ids": [], "uncertain_requirement_ids": []}
    ],
    "latest_result": {
      "planned_work": "Устройство котлована",
      "work_group": "Земляные работы",
      "profile_id": 15,
      "status": "OK",
      "instant_status": "OK",
      "persistent_status": null,
      "observability": {"score": 1.0, "level": "HIGH"},
      "match": {"score": 1.0, "coverage": 1.0, "confirmed_count": 2, "missing_count": 0, "uncertain_count": 0, "assessed_count": 2, "satisfied": [{"id": "required:excavator", "type": "required", "classes": ["excavator"], "display_names": ["экскаватор"], "found_classes": ["excavator"], "found_display_names": ["экскаватор"], "source_terms": []}, {"id": "required:dump_truck", "type": "required", "classes": ["dump_truck"], "display_names": ["автосамосвал"], "found_classes": ["dump_truck"], "found_display_names": ["автосамосвал"], "source_terms": []}], "missing": [], "uncertain": [], "supporting": []},
      "detections": {"accepted": [{"class": "excavator", "display_name": "экскаватор", "confidence": 0.93, "bbox": [100.0, 80.0, 420.0, 360.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}, {"class": "dump_truck", "display_name": "автосамосвал", "confidence": 0.88, "bbox": [440.0, 120.0, 760.0, 390.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}], "rejected_low_confidence": [], "rejected_unknown_class": [], "rejected_zone_mismatch": [], "rejected_invalid": []},
      "possible_unexpected": [],
      "evidence": [],
      "explanation": "Все проверяемые требования подтверждены.",
      "requirements": {"source_text": "Экскаватор, автосамосвал; бульдозер для планировки/зачистки", "source_key": "ТК 62-04 — разработка грунта в котловане экскаваторами с погрузкой в автосамосвалы", "rule": {"required": ["excavator", "dump_truck"], "alternatives": [], "conditional": [{"condition": "для планировки/зачистки", "detectable": ["bulldozer"], "mode": "all", "undetectable": [], "undetectable_relation": "additional"}], "undetectable": [], "monitorable": true}},
      "metadata": {"timestamp": "2026-09-29T12:00:10+03:00", "camera_id": "demo-camera", "zone_id": "pit-a", "target_zone_id": "pit-a"},
      "uncertainty_present": false
    },
    "frames_count": 3
  },
  "frames": [
    {
      "index": 0,
      "filename": "frame_001.jpg",
      "timestamp": "2026-09-29T12:00:00+03:00",
      "instant_status": "OK",
      "observability": {"score": 1.0, "level": "HIGH"},
      "match": {"score": 1.0, "coverage": 1.0, "confirmed_count": 2, "missing_count": 0, "uncertain_count": 0, "assessed_count": 2, "satisfied": [{"id": "required:excavator"}, {"id": "required:dump_truck"}], "missing": [], "uncertain": [], "supporting": []},
      "detections": {"accepted": [{"class": "excavator", "display_name": "экскаватор", "confidence": 0.93, "bbox": [100.0, 80.0, 420.0, 360.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}, {"class": "dump_truck", "display_name": "автосамосвал", "confidence": 0.88, "bbox": [440.0, 120.0, 760.0, 390.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}], "rejected_low_confidence": [], "rejected_unknown_class": [], "rejected_zone_mismatch": [], "rejected_invalid": []},
      "possible_unexpected": [],
      "evidence": [],
      "explanation": "Все проверяемые требования подтверждены.",
      "timing_seconds": {"inference": 0.42, "matching": 0.001}
    },
    {
      "index": 1,
      "filename": "frame_002.jpg",
      "timestamp": "2026-09-29T12:00:05+03:00",
      "instant_status": "OK",
      "observability": {"score": 1.0, "level": "HIGH"},
      "match": {"score": 1.0, "coverage": 1.0, "confirmed_count": 2, "missing_count": 0, "uncertain_count": 0, "assessed_count": 2, "satisfied": [{"id": "required:excavator"}, {"id": "required:dump_truck"}], "missing": [], "uncertain": [], "supporting": []},
      "detections": {"accepted": [{"class": "excavator", "display_name": "экскаватор", "confidence": 0.92, "bbox": [102.0, 81.0, 421.0, 361.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}, {"class": "dump_truck", "display_name": "автосамосвал", "confidence": 0.87, "bbox": [441.0, 121.0, 761.0, 391.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}], "rejected_low_confidence": [], "rejected_unknown_class": [], "rejected_zone_mismatch": [], "rejected_invalid": []},
      "possible_unexpected": [],
      "evidence": [],
      "explanation": "Все проверяемые требования подтверждены.",
      "timing_seconds": {"inference": 0.41, "matching": 0.001}
    },
    {
      "index": 2,
      "filename": "frame_003.jpg",
      "timestamp": "2026-09-29T12:00:10+03:00",
      "instant_status": "OK",
      "observability": {"score": 1.0, "level": "HIGH"},
      "match": {"score": 1.0, "coverage": 1.0, "confirmed_count": 2, "missing_count": 0, "uncertain_count": 0, "assessed_count": 2, "satisfied": [{"id": "required:excavator"}, {"id": "required:dump_truck"}], "missing": [], "uncertain": [], "supporting": []},
      "detections": {"accepted": [{"class": "excavator", "display_name": "экскаватор", "confidence": 0.91, "bbox": [101.0, 82.0, 422.0, 362.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}, {"class": "dump_truck", "display_name": "автосамосвал", "confidence": 0.86, "bbox": [442.0, 122.0, 762.0, 392.0], "camera_id": "demo-camera", "zone_id": "pit-a", "zone_verified": true}], "rejected_low_confidence": [], "rejected_unknown_class": [], "rejected_zone_mismatch": [], "rejected_invalid": []},
      "possible_unexpected": [],
      "evidence": [],
      "explanation": "Все проверяемые требования подтверждены.",
      "timing_seconds": {"inference": 0.40, "matching": 0.001}
    }
  ],
  "metadata": {
    "camera_id": "demo-camera",
    "zone_id": "pit-a",
    "active_conditions": {},
    "timestamps_source": "provided",
    "max_series_frames": 30,
    "confidence_threshold": 0.25,
    "provider": "rfdetr",
    "timings": {
      "model_loaded_once": true,
      "provider_reused": true,
      "frame_inference_seconds": [0.42, 0.41, 0.40],
      "frame_matching_seconds": [0.001, 0.001, 0.001],
      "total_inference_seconds": 1.23,
      "total_matching_seconds": 0.003,
      "temporal_seconds": 0.002,
      "total_request_seconds": 1.24
    }
  }
}
```

Для `persistent_missing`/`persistent_uncertain` каждый элемент дополнительно
содержит `frame_indices`, рассчитанные из тех же observation results, которые
достигли temporal persistence threshold. Один случайный кадр сам по себе не
объявляется persistent evidence.

### Рекомендация для UI серии

Интерфейс должен позволять выбрать несколько изображений одной камеры, сохранить
их порядок и показать превью серии. На каждом кадре рисуйте его собственные bbox,
class, confidence и instant status. Отдельно покажите общий persistent status,
итоговое explanation и persistent missing/uncertain; `UNCERTAIN` нельзя называть
доказанным нарушением.

## Response /analyze

Backend возвращает исходный `construction_monitoring` contract без альтернативной
бизнес-логики. Основные поля для сайта:

- status: `status`;
- match score: `match.score`;
- coverage: `match.coverage`;
- обнаруженная техника и bbox: `detections.accepted`;
- отсутствующие требования: `match.missing`;
- неопределенные требования: `match.uncertain`;
- объяснение: `explanation`;
- информационные detections: `possible_unexpected`.

Сокращенный пример ответа (в реальном ответе присутствуют все показанные поля и
полные requirement objects):

```json
{
  "planned_work": "Устройство котлована",
  "work_group": "Земляные работы",
  "profile_id": 15,
  "status": "OK",
  "instant_status": "OK",
  "persistent_status": null,
  "observability": {"score": 1.0, "level": "HIGH"},
  "match": {
    "score": 1.0,
    "coverage": 1.0,
    "confirmed_count": 2,
    "missing_count": 0,
    "uncertain_count": 0,
    "assessed_count": 2,
    "satisfied": [
      {
        "id": "required:excavator",
        "type": "required",
        "classes": ["excavator"],
        "display_names": ["экскаватор"],
        "found_classes": ["excavator"],
        "found_display_names": ["экскаватор"],
        "source_terms": []
      },
      {
        "id": "required:dump_truck",
        "type": "required",
        "classes": ["dump_truck"],
        "display_names": ["автосамосвал"],
        "found_classes": ["dump_truck"],
        "found_display_names": ["автосамосвал"],
        "source_terms": []
      }
    ],
    "missing": [],
    "uncertain": [],
    "supporting": []
  },
  "detections": {
    "accepted": [
      {
        "class": "excavator",
        "display_name": "экскаватор",
        "confidence": 0.8528093695640564,
        "bbox": [91.1511459350586, 184.28504943847656, 440.36602783203125, 403.69061279296875],
        "camera_id": "cpu-smoke-camera",
        "zone_id": null,
        "zone_verified": true
      },
      {
        "class": "dump_truck",
        "display_name": "автосамосвал",
        "confidence": 0.555757999420166,
        "bbox": [410.6966552734375, 296.9217529296875, 806.4915771484375, 504.2975769042969],
        "camera_id": "cpu-smoke-camera",
        "zone_id": null,
        "zone_verified": true
      }
    ],
    "rejected_low_confidence": [],
    "rejected_unknown_class": [],
    "rejected_zone_mismatch": [],
    "rejected_invalid": []
  },
  "possible_unexpected": [],
  "evidence": [],
  "explanation": "Для запланированной работы ожидаются: экскаватор и автосамосвал. Обнаружены: экскаватор и автосамосвал. Все проверяемые требования подтверждены.",
  "requirements": {
    "source_text": "Экскаватор, автосамосвал; бульдозер для планировки/зачистки",
    "source_key": "ТК 62-04 — разработка грунта в котловане экскаваторами с погрузкой в автосамосвалы",
    "rule": {
      "required": ["excavator", "dump_truck"],
      "alternatives": [],
      "conditional": [
        {
          "condition": "для планировки/зачистки",
          "detectable": ["bulldozer"],
          "mode": "all",
          "undetectable": [],
          "undetectable_relation": "additional"
        }
      ],
      "undetectable": [],
      "monitorable": true
    }
  },
  "metadata": {
    "timestamp": "2026-09-28T14:05:00+03:00",
    "camera_id": "cpu-smoke-camera",
    "zone_id": null,
    "target_zone_id": null
  },
  "uncertainty_present": false
}
```

Не рассчитывайте score самостоятельно и не ищите `match_score` на верхнем уровне.
Показывайте `match.score` вместе с `match.coverage`; `score` может быть `null`.

## Отображение statuses

| Status | Как показывать |
|---|---|
| `OK` | Требования, которые можно проверить, подтверждены. |
| `WARNING` | Предупреждение: проверяемое требование доказуемо отсутствует. |
| `UNCERTAIN` | Неопределенность. Не называть доказанным нарушением. |
| `NOT_MONITORABLE` | Текущая работа не проверяется данной CV taxonomy. |
| `UNKNOWN_WORK` | Работа отсутствует в каталоге; HTTP API обычно возвращает `422`. |

`possible_unexpected` имеет уровень INFO и не является нарушением.

## BBox

Используйте `detections.accepted[].bbox` в формате
`[x_min, y_min, x_max, y_max]` в пикселях исходного изображения. При отображении
масштабируйте X и Y теми же коэффициентами, которыми масштабировано изображение.
Цвет/подпись можно брать из `class`, `display_name` и `confidence`.

## Ошибки

| HTTP | Причина |
|---|---|
| `400` | Пустой, поврежденный или нечитаемый image; лимит image 10 MB. |
| `413` | Число файлов в серии превышает `MAX_SERIES_FRAMES`. |
| `415` | Content-Type не JSON и не multipart. |
| `422` | Неизвестная работа/class, неверный bbox, condition или другое поле request. |
| `503` | RF-DETR provider выбран, но модель не настроена/недоступна. |

Формат ошибки:

```json
{
  "error": "INVALID_REQUEST",
  "issues": [
    {
      "path": "request.context.planned_work",
      "code": "unknown_planned_work",
      "message": "planned_work is not present in the construction work catalog."
    }
  ]
}
```

Пустой результат CV (`detections: []`) возвращает обычный `200` с вычисленным
status и не считается backend error.

Для series request ошибка изображения содержит индекс, например
`request.files[2]`. Пустая серия, несовпадение количества timestamps, timestamps
без timezone и отсутствие обязательного `camera_id` возвращают `422`.

## Статус RF-DETR

Локальный запуск по умолчанию использует `FixtureProvider`, чтобы API/UI можно было
разрабатывать без загрузки модели. FINAL80 deployment model включена в пакет.
`RFDETRProvider` передает Pillow image в `model.predict()` с единым production
threshold `0.25`, а native result направляет в существующий
`construction_monitoring.analyze_rfdetr()`.

Factory реализован в `backend.rfdetr_factory:create_model`, модель находится в
`models/rfdetr/final_model.pth`:

```bash
CV_PROVIDER=rfdetr \
RFDETR_MODEL_FACTORY=backend.rfdetr_factory:create_model \
RFDETR_CHECKPOINT=models/rfdetr/final_model.pth \
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Factory создает `RFDETRMedium(pretrain_weights=..., resolution=640,
num_classes=13)`, как в FINAL80 Kaggle notebook. На машине без CUDA/MPS RF-DETR автоматически
использует CPU; устройство можно зафиксировать через `RFDETR_DEVICE=cpu`. Конвертацию
native RF-DETR output писать повторно не нужно.

В `/analyze-series` этот же provider/model создается один раз при старте FastAPI и
переиспользуется для всех файлов. Независимого series threshold нет.

## CORS

По умолчанию разрешены локальные frontend origins на портах `3000` и `5173` для
`localhost` и `127.0.0.1`. Другие origins задаются через переменную окружения:

```bash
CORS_ORIGINS='http://localhost:4173,https://demo.example' ...
```
