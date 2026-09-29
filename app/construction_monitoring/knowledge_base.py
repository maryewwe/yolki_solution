from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


MODEL_CLASSES: tuple[str, ...] = (
    "excavator",
    "dump_truck",
    "truck",
    "loader",
    "bulldozer",
    "motor_grader",
    "roller",
    "concrete_mixer",
    "telehandler",
    "piling_machine",
    "crane_manipulator",
    "mobile_crane",
    "tower_crane",
)


DISPLAY_NAMES_RU: Mapping[str, str] = {
    "excavator": "экскаватор",
    "dump_truck": "автосамосвал",
    "truck": "грузовой автомобиль",
    "loader": "погрузчик",
    "bulldozer": "бульдозер",
    "motor_grader": "автогрейдер",
    "roller": "дорожный каток",
    "concrete_mixer": "автобетоносмеситель",
    "telehandler": "телескопический погрузчик",
    "piling_machine": "сваебойная установка",
    "crane_manipulator": "кран-манипулятор",
    "mobile_crane": "автокран",
    "tower_crane": "башенный кран",
}


# IDs reproduce the original Colab catalog, but are now anchored to the exact
# source wording. This avoids value_counts() tie-order changes between pandas
# versions silently applying a rule to the wrong construction work.
PROFILE_DESCRIPTIONS: tuple[str, ...] = (
    "Специализированная малая механизация; состав техники уточняется по ППР и конкретной технологии",
    "Малая механизация и электроинструмент; грузовой/мачтовый подъемник для подачи материалов; при высотных работах — автовышка/фасадный подъемник",
    "Строительная техника не является основной; при необходимости — грузовой автомобиль/манипулятор для доставки и погрузки",
    "Автобетоносмеситель, автобетононасос или кран с бадьей; глубинные вибраторы; при арматурных/монтажных операциях — автокран",
    "Грузовой подъемник/фасадный подъемник; при тяжелых блоках — кран-манипулятор",
    "Автокран/кран-манипулятор или грузовой подъемник; при высотных работах — автовышка/подъемник",
    "Кран/крышевой кран или грузовой подъемник для подачи материалов; при работах на высоте — фасадный/монтажный подъемник",
    "Фасадный подъемник/автогидроподъемник; кран/манипулятор для подачи материалов; малая механизация",
    "Экскаватор и автосамосвал при подземной прокладке; кабельный тягач/лебедка; автокран/манипулятор для барабанов — по ППР",
    "Экскаватор, автосамосвал; трубоукладчик/автокран или экскаватор с крановым оборудованием; при сварке — сварочный агрегат",
    "Автокран/кран-манипулятор или грузовой подъемник; при малых грузах — тележки/малая механизация",
    "Автомобиль/кран-манипулятор; автогидроподъемник для монтажа; для разметки — разметочная машина",
    "Монтажный подъемник/лестницы, электромонтажный инструмент; для тяжелого оборудования — кран-манипулятор/грузовой подъемник",
    "Мини-экскаватор/ямокопатель, мини-погрузчик, самосвал; поливочная машина; при посадке крупных деревьев — кран-манипулятор",
    "Автомобильный/гусеничный монтажный кран; при сборке — автокран/манипулятор, сварочное оборудование",
    "Экскаватор, автосамосвал; бульдозер для планировки/зачистки",
    "Буровая установка/бурильно-крановая машина, автокран, автобетоносмеситель; при необходимости — автобетононасос и автосамосвал",
    "Экскаватор или бульдозер, автосамосвал; каток/виброкаток для уплотнения",
    "Экскаватор с гидромолотом/гидроножницами, погрузчик, автосамосвал; при тяжелых элементах — автокран",
    "Мини-погрузчик/погрузчик, автосамосвал; виброплита/виброкаток; при резке — швонарезчик",
    "Автосамосвалы, асфальтоукладчик, дорожные катки; при обработке основания — автогудронатор/поливомоечная машина",
    "Автокран-манипулятор/автогидроподъемник; буровая/ямобур для оснований; грузовой автомобиль",
    "Автокран/башенный кран для подачи арматуры; сварочное оборудование; при бетонных работах — автобетоносмеситель и автобетононасос",
    "Каток дорожный/виброкаток; при ограниченных местах — виброплита",
    "Автосамосвал, автогрейдер/бульдозер, дорожный каток; для бетонного основания — автобетоносмеситель",
    "Кран + вибропогружатель/вибромолот; при разработке котлована — экскаватор и автосамосвал",
    "Монтажный кран, автокран; при бетонировании — автобетоносмеситель и автобетононасос",
    "Ямобур/мини-экскаватор, автокран/манипулятор для секций; бурильная машина при больших объемах",
    "Экскаватор, автосамосвал, автокран/трубоукладчик; сварочное оборудование",
    "Погрузчик/экскаватор, трактор, грузовой автомобиль; специализированное оборудование для валки/измельчения — по ППР",
    "Экскаватор или ямокопатель, автокран/манипулятор, грузовой автомобиль",
    "Автокран/кран-манипулятор, монтажный подъемник; при необходимости — сварочное оборудование",
    "Автокран/монтажный кран; при монолитном исполнении — автобетоносмеситель и автобетононасос",
    "Автогидроподъемник/монтажный подъемник, электролебедка; при крупных блоках — автокран",
    "Буровая/ямобур или экскаватор, автобетоносмеситель; автокран при монтаже закладных/опор",
)


DESCRIPTION_TO_PROFILE_ID: Mapping[str, int] = {
    description: profile_id
    for profile_id, description in enumerate(PROFILE_DESCRIPTIONS)
}


def _alt(
    detectable: Sequence[str] = (),
    undetectable: Sequence[str] = (),
) -> Dict[str, List[str]]:
    return {
        "detectable": list(detectable),
        "undetectable": list(undetectable),
    }


def _cond(
    condition: str,
    detectable: Sequence[str] = (),
    *,
    mode: str = "any",
    undetectable: Sequence[str] = (),
    undetectable_relation: str = "additional",
) -> Dict[str, Any]:
    return {
        "condition": condition,
        "detectable": list(detectable),
        "mode": mode,
        "undetectable": list(undetectable),
        "undetectable_relation": undetectable_relation,
    }


def _rule(
    *,
    required: Sequence[str] = (),
    alternatives: Sequence[Mapping[str, Any]] = (),
    conditional: Sequence[Mapping[str, Any]] = (),
    undetectable: Sequence[str] = (),
    monitorable: bool = True,
) -> Dict[str, Any]:
    # Kept in snapshots for backward compatibility; runtime monitorability is
    # derived from the requirements active in the current analysis.
    return {
        "required": list(required),
        "alternatives": [dict(item) for item in alternatives],
        "conditional": [dict(item) for item in conditional],
        "undetectable": list(undetectable),
        "monitorable": monitorable,
    }


RULES: Mapping[int, Mapping[str, Any]] = {
    0: _rule(
        undetectable=("специализированная малая механизация",),
        monitorable=False,
    ),
    1: _rule(
        alternatives=(
            _alt(undetectable=("грузовой подъемник", "мачтовый подъемник")),
        ),
        conditional=(
            _cond(
                "при высотных работах",
                undetectable=("автовышка", "фасадный подъемник"),
                undetectable_relation="alternative",
            ),
        ),
        undetectable=(
            "малая механизация",
            "электроинструмент",
        ),
        monitorable=False,
    ),
    2: _rule(
        conditional=(
            _cond(
                "при необходимости, для доставки и погрузки",
                ("truck", "crane_manipulator"),
                mode="any",
            ),
        ),
        monitorable=False,
    ),
    3: _rule(
        alternatives=(
            _alt(
                ("concrete_mixer",),
                ("автобетононасос", "кран с бадьей"),
            ),
        ),
        conditional=(
            _cond(
                "при арматурных/монтажных операциях",
                ("mobile_crane",),
                mode="all",
            ),
        ),
        undetectable=("глубинные вибраторы",),
    ),
    4: _rule(
        alternatives=(
            _alt(undetectable=("грузовой подъемник", "фасадный подъемник")),
        ),
        conditional=(
            _cond("при тяжелых блоках", ("crane_manipulator",), mode="all"),
        ),
        monitorable=False,
    ),
    5: _rule(
        alternatives=(
            _alt(
                ("mobile_crane", "crane_manipulator"),
                ("грузовой подъемник",),
            ),
        ),
        conditional=(
            _cond(
                "при высотных работах",
                undetectable=("автовышка", "подъемник"),
                undetectable_relation="alternative",
            ),
        ),
    ),
    6: _rule(
        alternatives=(
            _alt(undetectable=("кран", "крышевой кран", "грузовой подъемник")),
        ),
        conditional=(
            _cond(
                "при работах на высоте",
                undetectable=("фасадный подъемник", "монтажный подъемник"),
                undetectable_relation="alternative",
            ),
        ),
        monitorable=False,
    ),
    7: _rule(
        alternatives=(
            _alt(undetectable=("фасадный подъемник", "автогидроподъемник")),
            _alt(("crane_manipulator",), ("кран",)),
        ),
        undetectable=("малая механизация",),
    ),
    8: _rule(
        alternatives=(
            _alt(undetectable=("кабельный тягач", "лебедка")),
        ),
        conditional=(
            _cond(
                "при подземной прокладке",
                ("excavator", "dump_truck"),
                mode="all",
            ),
            _cond(
                "для барабанов, по ППР",
                ("mobile_crane", "crane_manipulator"),
                mode="any",
            ),
        ),
    ),
    9: _rule(
        required=("excavator", "dump_truck"),
        alternatives=(
            _alt(
                ("mobile_crane",),
                ("трубоукладчик", "экскаватор с крановым оборудованием"),
            ),
        ),
        conditional=(
            _cond("при сварке", undetectable=("сварочный агрегат",)),
        ),
    ),
    10: _rule(
        alternatives=(
            _alt(
                ("mobile_crane", "crane_manipulator"),
                ("грузовой подъемник",),
            ),
        ),
        conditional=(
            _cond(
                "при малых грузах",
                undetectable=("тележки", "малая механизация"),
                undetectable_relation="alternative",
            ),
        ),
    ),
    11: _rule(
        alternatives=(
            _alt(("crane_manipulator",), ("автомобиль",)),
        ),
        conditional=(
            _cond("для монтажа", undetectable=("автогидроподъемник",)),
            _cond("для разметки", undetectable=("разметочная машина",)),
        ),
    ),
    12: _rule(
        alternatives=(
            _alt(undetectable=("монтажный подъемник", "лестницы")),
        ),
        conditional=(
            _cond(
                "для тяжелого оборудования",
                ("crane_manipulator",),
                mode="any",
                undetectable=("грузовой подъемник",),
                undetectable_relation="alternative",
            ),
        ),
        undetectable=("электромонтажный инструмент",),
        monitorable=False,
    ),
    13: _rule(
        required=("dump_truck",),
        alternatives=(
            _alt(undetectable=("мини-экскаватор", "ямокопатель")),
            _alt(undetectable=("мини-погрузчик",)),
        ),
        conditional=(
            _cond(
                "при посадке крупных деревьев",
                ("crane_manipulator",),
                mode="all",
            ),
        ),
        undetectable=("поливочная машина",),
    ),
    14: _rule(
        alternatives=(
            _alt(("mobile_crane",), ("гусеничный монтажный кран",)),
        ),
        conditional=(
            _cond(
                "при сборке",
                ("mobile_crane", "crane_manipulator"),
                mode="any",
                undetectable=("сварочное оборудование",),
                undetectable_relation="additional",
            ),
        ),
    ),
    15: _rule(
        required=("excavator", "dump_truck"),
        conditional=(
            _cond(
                "для планировки/зачистки",
                ("bulldozer",),
                mode="all",
            ),
        ),
    ),
    16: _rule(
        required=("mobile_crane", "concrete_mixer"),
        alternatives=(
            _alt(undetectable=("буровая установка", "бурильно-крановая машина")),
        ),
        conditional=(
            _cond(
                "при необходимости",
                ("dump_truck",),
                mode="all",
                undetectable=("автобетононасос",),
                undetectable_relation="additional",
            ),
        ),
    ),
    17: _rule(
        required=("dump_truck",),
        alternatives=(
            _alt(("excavator", "bulldozer")),
        ),
        conditional=(
            _cond("для уплотнения", ("roller",), mode="all"),
        ),
    ),
    18: _rule(
        required=("excavator", "dump_truck"),
        alternatives=(
            _alt(undetectable=("гидромолот", "гидроножницы")),
        ),
        conditional=(
            _cond("при тяжелых элементах", ("mobile_crane",), mode="all"),
        ),
        undetectable=("погрузчик",),
    ),
    19: _rule(
        required=("dump_truck",),
        alternatives=(
            _alt(undetectable=("мини-погрузчик", "погрузчик")),
            _alt(("roller",), ("виброплита",)),
        ),
        conditional=(
            _cond("при резке", undetectable=("швонарезчик",)),
        ),
    ),
    20: _rule(
        required=("dump_truck", "roller"),
        conditional=(
            _cond(
                "при обработке основания",
                undetectable=("автогудронатор", "поливомоечная машина"),
                undetectable_relation="alternative",
            ),
        ),
        undetectable=("асфальтоукладчик",),
    ),
    21: _rule(
        required=("truck",),
        alternatives=(
            _alt(("crane_manipulator",), ("автогидроподъемник",)),
        ),
        conditional=(
            _cond(
                "для оснований",
                undetectable=("буровая", "ямобур"),
                undetectable_relation="alternative",
            ),
        ),
    ),
    22: _rule(
        alternatives=(
            _alt(("mobile_crane", "tower_crane")),
        ),
        conditional=(
            _cond(
                "при бетонных работах",
                ("concrete_mixer",),
                mode="all",
                undetectable=("автобетононасос",),
                undetectable_relation="additional",
            ),
        ),
        undetectable=("сварочное оборудование",),
    ),
    23: _rule(
        required=("roller",),
        conditional=(
            _cond("при ограниченных местах", undetectable=("виброплита",)),
        ),
    ),
    24: _rule(
        required=("dump_truck", "roller"),
        alternatives=(
            _alt(("motor_grader", "bulldozer")),
        ),
        conditional=(
            _cond("для бетонного основания", ("concrete_mixer",), mode="all"),
        ),
    ),
    25: _rule(
        alternatives=(
            _alt(undetectable=("вибропогружатель", "вибромолот")),
        ),
        conditional=(
            _cond(
                "при разработке котлована",
                ("excavator", "dump_truck"),
                mode="all",
            ),
        ),
        undetectable=("кран",),
        monitorable=False,
    ),
    26: _rule(
        alternatives=(
            _alt(("mobile_crane",), ("монтажный кран",)),
        ),
        conditional=(
            _cond(
                "при бетонировании",
                ("concrete_mixer",),
                mode="all",
                undetectable=("автобетононасос",),
                undetectable_relation="additional",
            ),
        ),
    ),
    27: _rule(
        alternatives=(
            _alt(undetectable=("ямобур", "мини-экскаватор")),
            _alt(("mobile_crane", "crane_manipulator")),
        ),
        conditional=(
            _cond("при больших объемах", undetectable=("бурильная машина",)),
        ),
    ),
    28: _rule(
        required=("excavator", "dump_truck"),
        alternatives=(
            _alt(("mobile_crane",), ("трубоукладчик",)),
        ),
        undetectable=("сварочное оборудование",),
    ),
    29: _rule(
        required=("truck",),
        alternatives=(
            _alt(("excavator",), ("погрузчик",)),
        ),
        conditional=(
            _cond(
                "по ППР",
                undetectable=("специализированное оборудование для валки/измельчения",),
            ),
        ),
        undetectable=("трактор",),
    ),
    30: _rule(
        required=("truck",),
        alternatives=(
            _alt(("excavator",), ("ямокопатель",)),
            _alt(("mobile_crane", "crane_manipulator")),
        ),
    ),
    31: _rule(
        alternatives=(
            _alt(("mobile_crane", "crane_manipulator")),
        ),
        conditional=(
            _cond("при необходимости", undetectable=("сварочное оборудование",)),
        ),
        undetectable=("монтажный подъемник",),
    ),
    32: _rule(
        alternatives=(
            _alt(("mobile_crane",), ("монтажный кран",)),
        ),
        conditional=(
            _cond(
                "при монолитном исполнении",
                ("concrete_mixer",),
                mode="all",
                undetectable=("автобетононасос",),
                undetectable_relation="additional",
            ),
        ),
    ),
    33: _rule(
        alternatives=(
            _alt(undetectable=("автогидроподъемник", "монтажный подъемник")),
        ),
        conditional=(
            _cond("при крупных блоках", ("mobile_crane",), mode="all"),
        ),
        undetectable=("электролебедка",),
        monitorable=False,
    ),
    34: _rule(
        required=("concrete_mixer",),
        alternatives=(
            _alt(("excavator",), ("буровая", "ямобур")),
        ),
        conditional=(
            _cond(
                "при монтаже закладных/опор",
                ("mobile_crane",),
                mode="all",
            ),
        ),
    ),
}


def detectable_classes(rule: Mapping[str, Any]) -> set[str]:
    classes = set(rule["required"])
    for alternative in rule["alternatives"]:
        classes.update(alternative["detectable"])
    for conditional in rule["conditional"]:
        classes.update(conditional["detectable"])
    return classes


def validate_rules() -> List[str]:
    issues: List[str] = []
    expected_ids = set(range(len(PROFILE_DESCRIPTIONS)))
    actual_ids = set(RULES)
    if actual_ids != expected_ids:
        issues.append(
            f"profile IDs mismatch: missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )

    model_classes = set(MODEL_CLASSES)
    for profile_id, rule in RULES.items():
        required_keys = {
            "required",
            "alternatives",
            "conditional",
            "undetectable",
            "monitorable",
        }
        missing_keys = required_keys - set(rule)
        if missing_keys:
            issues.append(f"profile {profile_id}: missing keys {sorted(missing_keys)}")
            continue

        unknown = detectable_classes(rule) - model_classes
        if unknown:
            issues.append(f"profile {profile_id}: unknown model classes {sorted(unknown)}")

        for index, alternative in enumerate(rule["alternatives"]):
            if not alternative["detectable"] and not alternative["undetectable"]:
                issues.append(f"profile {profile_id}: empty alternative {index}")

        for index, conditional in enumerate(rule["conditional"]):
            if conditional["mode"] not in {"all", "any"}:
                issues.append(
                    f"profile {profile_id}: conditional {index} has invalid mode"
                )
            if conditional["undetectable_relation"] not in {
                "additional",
                "alternative",
            }:
                issues.append(
                    f"profile {profile_id}: conditional {index} has invalid "
                    "undetectable_relation"
                )

    if len(DESCRIPTION_TO_PROFILE_ID) != len(PROFILE_DESCRIPTIONS):
        issues.append("profile descriptions are not unique")
    return issues


def assert_valid_rules() -> None:
    issues = validate_rules()
    if issues:
        raise ValueError("Invalid knowledge base:\n- " + "\n- ".join(issues))


assert_valid_rules()
