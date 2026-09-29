# Integration contract: `construction_monitoring`

Версия: 1.1, 28 сентября 2026 г.

Knowledge base, matching semantics, observability, match score, coverage и temporal
logic считаются замороженными. Этот документ описывает внешний integration boundary,
не меняя алгоритм.

Машинные схемы:

- `integration/schemas/analysis_request.schema.json`;
- `integration/schemas/analysis_response.schema.json`;
- `integration/schemas/temporal_request.schema.json`;
- `integration/schemas/temporal_response.schema.json`.

## Entry points

```python
from construction_monitoring import (
    IntegrationValidationError,
    analyze,
    analyze_rfdetr,
    temporal_update,
)

instant_result = analyze(request)
rfdetr_result = analyze_rfdetr(rfdetr_request)
temporal_result = temporal_update(temporal_request)
```

### RF-DETR input adapter

Native API модели возвращает единый RF-DETR result object. Adapter читает:

- class: `det.data["class_name"]`;
- confidence: `det.confidence`;
- bbox: `det.xyxy`.

```python
det = model.predict(image, threshold=predict_threshold)
rfdetr_result = analyze_rfdetr(
    {
        "context": {
            "planned_work": "Устройство котлована",
            "active_conditions": {},
            "timestamp": "2026-09-28T10:00:00+03:00",
            "camera_id": "cam-rfdetr",
        },
        "detections": det,
    }
)
```

`score` существует в notebook только как колонка производного evaluation DataFrame.
Это не native поле RF-DETR и integration adapter его не принимает. После адаптации
canonical detection содержит `class`, `confidence`, `bbox`.

Production/demo confidence threshold согласован и равен `0.25`. Сравнение
включительное: `confidence >= 0.25` принимается, меньшее значение попадает в
`rejected_low_confidence`. RF-DETR adapter только преобразует native result в canonical
detections и не вводит отдельный cutoff. Evaluation predict threshold `0.001`
используется только для сбора predictions в RF-DETR evaluation и не входит в
production/demo pipeline.

Обе функции возвращают JSON-serializable `dict`. При невалидном input adapter
выбрасывает `IntegrationValidationError`. HTTP adapter должен преобразовать его в
`422 Unprocessable Entity` и вернуть `exc.as_dict()`:

```json
{
  "error": "INVALID_REQUEST",
  "issues": [
    {
      "path": "request.context.timestamp",
      "code": "timezone_required",
      "message": "Timestamp must include an explicit timezone offset or Z."
    }
  ]
}
```

## Exact instant input JSON

```json
{
  "context": {
    "planned_work": "Устройство котлована",
    "active_conditions": {},
    "timestamp": "2026-09-27T10:00:00+03:00",
    "camera_id": "cam-01",
    "zone_id": "pit-a"
  },
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

Root допускает только `context` и `detections`. Неизвестные поля отклоняются.

### A. Calendar context

| Поле | Обязательность | Тип | Validation / смысл |
|---|---|---|---|
| `context.planned_work` | required | non-empty string | Должно однозначно разрешаться в `work_catalog.json`; в response возвращается каноническое название из Excel. |
| `context.active_conditions` | optional, default `{}` | object: `string -> boolean` | Ключ обязан существовать в profile выбранной работы. Matcher не выводит condition из `planned_work`. |
| `context.timestamp` | required | string | ISO 8601 с явным offset или `Z`; naive timestamp отклоняется. |
| `context.camera_id` | required | non-empty string | ID источника кадра/потока. |
| `context.zone_id` | optional | non-empty string | Observation/target zone, уже назначенная zone layer. Не передавать `null`; при неизвестной зоне поле нужно опустить. |

### B. CV detections

| Поле | Обязательность | Тип | Validation / смысл |
|---|---|---|---|
| `detections` | required | array | Может быть пустым. |
| `detections[].class` | required | enum string | Ровно один из 13 class ID текущего model contract. |
| `detections[].confidence` | required для canonical `analyze` | finite number | Диапазон `[0.0, 1.0]`; production/demo threshold `0.25`, сравнение включительное. |
| `detections[].bbox` | required | four finite numbers | `[x_min, y_min, x_max, y_max]`, значения неотрицательны, `x_max > x_min`, `y_max > y_min`. |
| `detections[].zone_id` | optional enrichment | non-empty string | Добавляет zone layer после mapping bbox -> ROI. Это не обязательный output CV-модели. |

Допустимые `class`:

```text
excavator, dump_truck, truck, loader, bulldozer, motor_grader, roller,
concrete_mixer, telehandler, piling_machine, crane_manipulator, mobile_crane,
tower_crane
```

Строковые legacy detections поддерживаются core для старого notebook, но production
adapter их намеренно не принимает.

### Zone behavior

- Если `context.zone_id` известен, он является target zone анализа.
- Если у detection нет собственного `zone_id`, он наследует observation zone. Это
  означает, что backend/zone layer уже отнес весь observation к этой зоне.
- Если zone layer назначил отдельный `detections[].zone_id`, detection из другой зоны
  попадает в `detections.rejected_zone_mismatch` и не подтверждает requirement.
- Если `context.zone_id` отсутствует, анализ выполняется на уровне всего кадра:
  `metadata.zone_id=null`, `metadata.target_zone_id=null`; геометрическая зона не
  придумывается и zone filtering не выполняется.
- `bbox` не преобразуется в semantic zone внутри модуля. Mapping bbox -> configured ROI
  принадлежит zone layer.

## Exact instant output JSON

`match_score` не дублируется на верхнем уровне: фактическое поле называется
`match.score`. Coverage находится в `match.coverage`; timestamp/camera/zone находятся в
`metadata`.

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
        "confidence": 0.93,
        "bbox": [100.0, 80.0, 420.0, 360.0],
        "camera_id": "cam-01",
        "zone_id": "pit-a",
        "zone_verified": true
      },
      {
        "class": "dump_truck",
        "display_name": "автосамосвал",
        "confidence": 0.88,
        "bbox": [440.0, 120.0, 760.0, 390.0],
        "camera_id": "cam-01",
        "zone_id": "pit-a",
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
    "timestamp": "2026-09-27T10:00:00+03:00",
    "camera_id": "cam-01",
    "zone_id": "pit-a",
    "target_zone_id": "pit-a"
  },
  "uncertainty_present": false
}
```

### Output field map

| Требование интеграции | Фактическое поле |
|---|---|
| Instant status | `instant_status` (`status` совпадает с ним для instant call) |
| Persistent status | `persistent_status`; `null` в `analyze`, заполнено в `temporal_update` |
| Planned work | `planned_work` |
| Observability | `observability.score`, `observability.level` |
| Match score | `match.score` |
| Coverage | `match.coverage` |
| Satisfied | `match.satisfied` |
| Missing | `match.missing` |
| Uncertain | `match.uncertain` |
| Supporting | `match.supporting` |
| Evidence | `evidence` |
| Possible unexpected / INFO | `possible_unexpected`; соответствующий INFO также может быть в `evidence` |
| Explanation | `explanation` |
| Timestamp | `metadata.timestamp` |
| Camera | `metadata.camera_id` |
| Zone | `metadata.zone_id`, target — `metadata.target_zone_id` |
| Accepted/rejected detections | `detections.*` |
| Source/rule trace | `requirements` |

## Temporal update

Temporal layer остается batch/stateless. Backend хранит историю и передает выбранное
окно как массив тех же instant requests:

```json
{
  "observations": [
    {
      "context": {
        "planned_work": "Устройство котлована",
        "active_conditions": {},
        "timestamp": "2026-09-27T11:00:00+03:00",
        "camera_id": "cam-01",
        "zone_id": "pit-a"
      },
      "detections": [
        {"class": "excavator", "confidence": 0.93, "bbox": [100, 80, 420, 360]},
        {"class": "dump_truck", "confidence": 0.88, "bbox": [440, 120, 760, 390]}
      ]
    }
  ],
  "temporal_config": {
    "window_seconds": null,
    "min_observations": 3,
    "persistence_ratio": 0.75
  }
}
```

Все observations должны иметь одинаковые `planned_work`, `camera_id`, `zone_id` и
`active_conditions`; timestamps должны быть уникальны. При смене любого из этих полей
backend начинает новую историю. `temporal_config` optional; показанные значения — defaults.

Temporal response сохраняет `match`, `observability`, `detections`, `metadata` и evidence
последнего observation, добавляет `observation_count`, `temporal.persistent_missing`,
`temporal.persistent_uncertain`, summary `observations` и полный `latest_result`.

## Integration fixtures

Все planned works взяты из исходного Excel. Все detections в fixtures synthetic и явно
помечены `synthetic_fixture=true`, `data_origin.detections="synthetic"`.

| Fixture | Файл | Проверяемый результат |
|---|---|---|
| A | `a_ok.json` | `OK` |
| B | `b_warning.json` | доказуемый `WARNING` |
| C | `c_uncertain.json` | `UNCERTAIN` |
| D | `d_not_monitorable.json` | `NOT_MONITORABLE` |
| E | `e_low_confidence.json` | low-confidence detection отклонен, INFO evidence |
| F | `f_zone_unknown.json` | whole-frame analysis без придуманной зоны |
| G | `g_other_zone.json` | detection другой zone отклонен |
| H | `h_temporal_isolated_miss.json` | isolated miss, persistent `OK` |
| I | `i_temporal_persistent_warning.json` | persistent `WARNING` |
| J | `j_active_condition.json` | явный active condition добавляет requirement |

## CV TEAM SENDS

- Native RF-DETR result object: `det.data["class_name"]`, `det.confidence`, `det.xyxy`.
- In-process caller передает этот object в `analyze_rfdetr`; DataFrame `score` не используется.
- При межсервисном JSON transport CV/ingestion сериализует canonical `class`,
  `confidence`, `bbox` и вызывает `analyze`.
- `class` только из согласованного списка 13 классов.
- Все instances, а не только уникальные classes.
- `bbox` в координатах исходного кадра, формат finite non-negative XYXY.
- Model/version metadata и frame identity передаются ingestion/backend отдельным
  transport envelope; текущий matching request их не дублирует.
- CV не обязана определять semantic `zone_id`.
- Timestamp/camera metadata дает capture/ingestion layer. CV может технически вернуть
  их вместе с inference result, но владельцем и источником истины остается ingestion.

## CALENDAR/BACKEND SENDS

- `planned_work` из versioned календарного плана, канонически сопоставимый с каталогом.
- `active_conditions` как явный object `condition -> boolean`.
- `timestamp` и `camera_id`, полученные от ingestion/capture.
- `zone_id`, если observation уже привязан к configured zone.

Matcher не выводит conditions эвристикой из текста `planned_work`.

## ZONE LAYER SENDS

- Observation-level `context.zone_id`, если весь кадр/поток уже относится к одной зоне; или
- per-detection `detections[].zone_id` после mapping bbox -> configured ROI.

Если mapping отсутствует, zone fields опускаются и результат относится ко всему кадру.

## OUR MODULE RETURNS

- `status`, `instant_status`, `persistent_status`;
- `planned_work`, `work_group`, `profile_id`;
- `observability`;
- `match.score`, `match.coverage`;
- `match.satisfied`, `match.missing`, `match.uncertain`, `match.supporting`;
- `evidence`, `possible_unexpected`;
- accepted/rejected `detections`;
- русское `explanation`;
- `metadata.timestamp`, `metadata.camera_id`, `metadata.zone_id`;
- source/rule snapshot в `requirements`;
- для temporal: persistence details, observation summaries и `latest_result`.

## BACKEND RESPONSIBILITY

- Валидировать transport/auth и преобразовывать `IntegrationValidationError` в HTTP 422.
- Хранить raw request, полный response, plan/rule/model versions и temporal history.
- Группировать историю по работе, camera, zone и active-condition context.
- Вызывать `temporal_update` на выбранном окне; модуль сам историю не хранит.
- Передавать результат frontend без потери `uncertain`, coverage и evidence severity.
- Привязывать `frame_id`/`image_uri` и хранить confirming image; matching core хранит bbox,
  но не изображение.
- Сбрасывать/разделять историю при смене этапа, камеры, зоны или active conditions.

## FRONTEND RESPONSIBILITY

- `WARNING` показывать как предупреждение.
- `UNCERTAIN` не называть доказанным нарушением.
- `NOT_MONITORABLE` показывать как нейтральное ограничение наблюдаемости.
- INFO evidence и `possible_unexpected` не называть нарушением.
- Всегда показывать `match.score` вместе с `match.coverage`.
- Показывать explanation, timestamp, camera/zone и bbox на связанном backend кадре.
- Не писать «работа выполнена»: модуль оценивает соответствие наблюдаемой техники.

## Unresolved taxonomy mappings

Mapping `mini-excavator -> excavator` **не подтвержден**. В profiles 13 и 27 термин
`мини-экскаватор` остается undetectable. До изменения knowledge base CV-команда должна:

1. собрать реальные кадры мини-экскаваторов из целевых камер/ракурсов;
2. прогнать текущую model version;
3. предоставить per-class recall/precision и confusion examples для `excavator`;
4. согласовать, можно ли считать этот subtype надежно покрытым class `excavator`.

Profiles 13 и 27 остаются `REVIEW` до этой проверки. Profiles 18, 19 и 29 также
остаются `REVIEW`, пока CV-команда не определит subtype boundary класса `loader` для
исходных терминов `погрузчик` и `мини-погрузчик`. Exact source term для `telehandler`
в текущих 35 профилях отсутствует; грузовые/мачтовые/фасадные подъемники к нему не
приравниваются.

## Decisions required before full integration

1. Кто владеет camera-specific ROI и mapping bbox -> zone.
2. Какой сервис является источником `timestamp`, `camera_id`, `frame_id/image_uri`.
3. Как versioned calendar передает канонический `planned_work` и active conditions.
4. Какой temporal window/cadence вызывает backend для каждого потока.
5. Где хранятся raw detections, responses и confirming frames.
6. Как frontend визуально различает `WARNING`, `UNCERTAIN`, `NOT_MONITORABLE` и INFO.
