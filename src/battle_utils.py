TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}

TYPE_ORDER = list(TYPE_CHART.keys())
TYPE_INDEX = {type_name: idx for idx, type_name in enumerate(TYPE_ORDER)}

STAT_NAME_MAP = {
    "attack": "atk",
    "defense": "def",
    "special-attack": "sp_atk",
    "special-defense": "sp_def",
    "speed": "spd",
    "accuracy": "accuracy",
    "evasion": "evasion",
}


def normalize_type_name(type_name):
    return (type_name or "").strip().lower()


def format_name(value):
    text = (value or "").replace("-", " ").replace("_", " ").strip()
    return text.capitalize() if text else "Unknown"


def get_type_multiplier(move_type, defender_types):
    move_type = normalize_type_name(move_type)
    multiplier = 1.0
    for defender_type in defender_types or []:
        defender_type = normalize_type_name(defender_type)
        multiplier *= TYPE_CHART.get(move_type, {}).get(defender_type, 1.0)
    return multiplier


def describe_effectiveness(multiplier):
    if multiplier == 0:
        return "No effect"
    if multiplier > 1:
        return "Super effective"
    if multiplier < 1:
        return "Not very effective"
    return "Neutral"


def get_type_index(type_name):
    return TYPE_INDEX.get(normalize_type_name(type_name), 0)


def stage_multiplier(stage):
    stage = max(-6, min(6, int(stage)))
    if stage >= 0:
        return (2 + stage) / 2
    return 2 / (2 - stage)


def apply_stat_stages(base_stats, stat_stages):
    current_stats = dict(base_stats)
    for api_name, short_name in STAT_NAME_MAP.items():
        if short_name in {"accuracy", "evasion"}:
            continue
        if short_name in current_stats:
            current_stats[short_name] = max(
                1,
                int(round(current_stats[short_name] * stage_multiplier(stat_stages.get(short_name, 0)))),
            )
    return current_stats


def build_type_chart_rows():
    rows = []
    for attacker in TYPE_ORDER:
        row = {"attack_type": format_name(attacker)}
        for defender in TYPE_ORDER:
            row[format_name(defender)] = TYPE_CHART.get(attacker, {}).get(defender, 1.0)
        rows.append(row)
    return rows
