# Codex handoff: frontend

## Контекст

Это готовый hackathon-проект контроля соответствия техники календарному плану
строительных работ. Backend уже реализован на FastAPI. Настоящий RF-DETR Medium
640 подключен, checkpoint находится в
`models/rfdetr/final_model.pth`, а production confidence threshold равен
`0.25` включительно.

Matcher, каталог работ, 13-class taxonomy и temporal aggregation готовы и
протестированы. Работают оба endpoint:

- `POST /analyze` для одного кадра;
- `POST /analyze-series` для серии кадров одной камеры с persistent result.

Не переписывай backend core, `construction_monitoring`, matching semantics,
knowledge base, observability, coverage, temporal formula или CV taxonomy без
обнаруженного интеграционного blocker. Следующая задача этого проекта именно
frontend.

## Главный контракт

Сначала прочитай `FRONTEND_API.md`: это основной источник API contract для сайта.
Для запуска и устройства пакета используй `README.md`. Более низкоуровневый
контракт matcher описан в `INTEGRATION_CONTRACT.md`.

Frontend должен:

1. Получать варианты `planned_work` и допустимые `active_conditions` через
   `GET /works`.
2. Позволять выбрать одну работу и загрузить несколько изображений одной серии.
3. Отправлять multipart request в `POST /analyze-series` с повторяемым полем
   `files`, а также `planned_work` и `camera_id`.
4. Показывать preview каждого кадра и рисовать
   `frames[].detections.accepted[].bbox`.
5. Показывать рядом с bbox поля `class` и `confidence`.
6. Показывать `frames[].instant_status` для каждого кадра.
7. Показывать общий `series_summary.persistent_status` и
   `series_summary.explanation`.
8. Показывать instant `frames[].match.missing` и `.uncertain`, а для итога серии
   использовать `series_summary.temporal.persistent_missing` и
   `.persistent_uncertain`.
9. Показывать `possible_unexpected` только как INFO, не как нарушение.
10. Использовать `frame_indices` в persistent missing/uncertain как ссылки на
    подтверждающие кадры, когда эти поля присутствуют.

`WARNING` является предупреждением о доказуемом устойчивом отсутствии.
`UNCERTAIN` нельзя называть доказанным нарушением. `NOT_MONITORABLE` означает,
что текущие требования нельзя проверить этой CV taxonomy.

Размещай frontend-код в `frontend/`. Backend по умолчанию доступен по адресу
`http://127.0.0.1:8000`, Swagger находится на
`http://127.0.0.1:8000/docs`. Локальные CORS origins для портов 3000 и 5173 уже
разрешены.

## Основные файлы

- `FRONTEND_API.md` - точный frontend-facing API contract и примеры.
- `README.md` - Quick Start, запуск backend и структура проекта.
- `INTEGRATION_CONTRACT.md` - canonical matcher contract.
- `backend/app.py` - FastAPI endpoints и series orchestration.
- `backend/schemas.py` - Pydantic/OpenAPI response schemas.

Перед изменениями frontend проверь `GET /health`, затем открой Swagger. Для
реальной модели запускай backend с `CV_PROVIDER=rfdetr` и
`RFDETR_MODEL_FACTORY=backend.rfdetr_factory:create_model` согласно README.
