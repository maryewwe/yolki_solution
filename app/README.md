# Construction monitoring hackathon demo

## Quick Start

Frontend-разработчику сначала нужно прочитать `FRONTEND_API.md`. В нем находится
актуальный API contract, примеры `FormData` и структура response.

macOS/Linux:

```bash
unzip construction-hackathon.zip
cd construction-hackathon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rfdetr.txt

CV_PROVIDER=rfdetr \
RFDETR_MODEL_FACTORY=backend.rfdetr_factory:create_model \
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Windows PowerShell, после распаковки ZIP и перехода в папку проекта:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-rfdetr.txt
$env:CV_PROVIDER = "rfdetr"
$env:RFDETR_MODEL_FACTORY = "backend.rfdetr_factory:create_model"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Для Windows Command Prompt команда активации: `.venv\Scripts\activate.bat`.

После запуска:

- backend: `http://127.0.0.1:8000`;
- Swagger/OpenAPI: `http://127.0.0.1:8000/docs`;
- health check: `http://127.0.0.1:8000/health`.

Система сравнивает технику на строительной площадке с техникой, ожидаемой для
выбранной календарной работы:

```text
series of frames -> RF-DETR per frame -> instant matching
                 -> existing temporal aggregation
                 -> persistent plan/fact result -> FastAPI -> frontend
```

Одиночный режим сохранен для обратной совместимости и отладки:
`image -> RF-DETR -> instant result -> POST /analyze`.

Production/demo confidence threshold: `0.25` включительно.

## Структура

- `frontend/` - место для приложения frontend-разработчика.
- `backend/` - FastAPI, CV providers и factory настоящего RF-DETR.
- `construction_monitoring/` - замороженный matcher и каталог работ.
- `models/rfdetr/final_model.pth` - FINAL80 RF-DETR Medium @640 deployment model.
- `integration/` - JSON schemas и integration fixtures.
- `tests/` - regression tests и реальный строительный кадр для smoke test.
- `sources/` - исходное ТЗ и Excel с работами/техникой.
- `FRONTEND_API.md` - практический API-контракт для сайта.
- `INTEGRATION_CONTRACT.md` - точный контракт matcher/backend.

`construction_monitoring` заморожен. Не меняйте knowledge base, matching semantics,
taxonomy, observability, match score, coverage, temporal logic и REVIEW profiles без
отдельно согласованного изменения спецификации.

## CV taxonomy

RF-DETR возвращает ровно 13 canonical classes:

```text
excavator
dump_truck
truck
loader
bulldozer
motor_grader
roller
concrete_mixer
telehandler
piling_machine
crane_manipulator
mobile_crane
tower_crane
```

## Python и установка

Чистая установка, end-to-end и все тесты этого пакета проверены на Python `3.11.8`
и `3.13.5`, PyTorch `2.14.0` и `rfdetr==1.10.1`; рабочее окружение с PyTorch
`2.9.1` также проходило smoke test. RF-DETR заявляет Python `>=3.10`.

```bash
cd construction-hackathon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rfdetr.txt
```

`requirements-rfdetr.txt` включает backend-зависимости. Для режима без настоящей
модели достаточно `requirements-backend.txt`.

## Запуск с fixture detections

```bash
CV_PROVIDER=fixture \
CV_FIXTURE_PATH=backend/demo_detections.json \
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Fixture provider нужен только для разработки API/UI. Это не ML inference.

## Запуск с настоящим RF-DETR

```bash
CV_PROVIDER=rfdetr \
RFDETR_MODEL_FACTORY=backend.rfdetr_factory:create_model \
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Factory по умолчанию читает
`models/rfdetr/final_model.pth`. На CPU можно явно добавить
`RFDETR_DEVICE=cpu`; inference будет работать медленнее, чем на GPU.

FINAL80 deployment model (`last_ema.pth` после фиксированных 30 эпох, экспортирован
notebook `lct-final80-rfdetr-medium640.ipynb` как `final_model.pth`):

```text
size:    134409987 bytes
SHA-256: 94a68be921b4e24cf12f362278c97bc2ae4ac47b1872a306ad9189e8e3702058
```

FINAL80 обучена на 80 development images и не имеет отдельного held-out validation
набора; историческую метрику предыдущей модели нельзя приписывать FINAL80.

Файл занимает около 128 MiB. Обычный GitHub push может отклонить его из-за лимита
размера; для будущего репозитория понадобится Git LFS или отдельное хранилище.
В локальный передаваемый ZIP deployment model включена.

## Проверка API

После запуска:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/works
```

- `GET /health` показывает provider, readiness и threshold.
- `GET /works` возвращает допустимые `planned_work` и их `active_conditions`.
- `POST /analyze` принимает JSON с готовыми detections или multipart с image.
- `POST /analyze-series` принимает multipart-серию изображений одной камеры,
  выполняет instant-анализ каждого кадра и возвращает persistent result.
- Swagger/OpenAPI: `http://127.0.0.1:8000/docs`.

Реальный image request из включённого smoke-кадра:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F 'image=@tests/assets/real_construction_frame.jpg' \
  -F 'planned_work=Устройство котлована' \
  -F 'active_conditions={}' \
  -F 'camera_id=demo-camera'
```

Серия кадров:

```bash
curl -X POST http://127.0.0.1:8000/analyze-series \
  -F 'files=@tests/assets/real_construction_frame.jpg' \
  -F 'files=@tests/assets/real_construction_frame.jpg' \
  -F 'files=@tests/assets/real_construction_frame.jpg' \
  -F 'planned_work=Устройство котлована' \
  -F 'camera_id=demo-camera' \
  -F 'active_conditions={}'
```

Модель загружается один раз при старте приложения и переиспользуется для каждого
кадра. Если timestamps не переданы, backend сохраняет upload order и создает
последовательные UTC timestamps. Demo default `MAX_SERIES_FRAMES=30`; это
configurable safety limit, а не ограничение temporal core.

Точный multipart/JSON contract, полный пример ответа, bbox и HTTP errors описаны в
`FRONTEND_API.md`.

## Поля для frontend

Frontend должен читать:

- итоговый статус: `status`;
- наблюдаемость: `observability.score`, `observability.level`;
- качество сопоставления: `match.score` вместе с `match.coverage`;
- подтверждённые требования: `match.satisfied`;
- доказуемо отсутствующие требования: `match.missing`;
- ненаблюдаемую/неопределённую часть: `match.uncertain`;
- найденную технику: `detections.accepted[].class`, `confidence`, `bbox`;
- дополнительные сигналы: `possible_unexpected` и `evidence`;
- объяснение: `explanation`.

Статусы:

| Status | Значение |
|---|---|
| `OK` | Все активные проверяемые требования подтверждены. |
| `WARNING` | Есть доказуемо отсутствующее проверяемое требование. |
| `UNCERTAIN` | Наблюдаемой информации недостаточно; это не подтверждённое нарушение. |
| `NOT_MONITORABLE` | Активные требования нельзя проверить текущей CV taxonomy. |
| `UNKNOWN_WORK` | Работа не разрешена в каталоге; HTTP API обычно вернёт `422`. |

`possible_unexpected` является INFO-сигналом и автоматически не означает нарушение.

## CORS

По умолчанию разрешены локальные frontend origins:

```text
http://localhost:3000
http://127.0.0.1:3000
http://localhost:5173
http://127.0.0.1:5173
```

Дополнительные origins задаются списком через `CORS_ORIGINS`. Production wildcard
не включён.

## Environment variables

| Variable | Назначение |
|---|---|
| `CV_PROVIDER` | `fixture` (default) или `rfdetr`. |
| `CV_FIXTURE_PATH` | JSON-файл detections для FixtureProvider. |
| `RFDETR_MODEL_FACTORY` | Для real mode: `backend.rfdetr_factory:create_model`. |
| `RFDETR_CHECKPOINT` | Optional путь к checkpoint вместо project default. |
| `RFDETR_DEVICE` | Optional `cpu`, `cuda` или `mps`, если доступно. |
| `CORS_ORIGINS` | Дополнительный comma-separated список frontend origins. |
| `MAX_SERIES_FRAMES` | Максимум файлов в `/analyze-series`; default `30`. |

## Тесты

```bash
python -m unittest discover -s tests -v
```

Ожидаемый baseline: `97 tests passed` (`85` прежних + `12` series regression).
