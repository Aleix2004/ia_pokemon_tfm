# Type colour palette — matches the official Pokémon game colour scheme.
# Used by the dashboard to render coloured type badges on moves and Pokémon.

TYPE_EMOJI: dict[str, str] = {
    "normal":   "⚪",
    "fire":     "🔥",
    "water":    "💧",
    "electric": "⚡",
    "grass":    "🌿",
    "ice":      "❄️",
    "fighting": "🥊",
    "poison":   "☠️",
    "ground":   "🌍",
    "flying":   "🌬️",
    "psychic":  "🔮",
    "bug":      "🐛",
    "rock":     "🪨",
    "ghost":    "👻",
    "dragon":   "🐉",
    "dark":     "🌑",
    "steel":    "⚙️",
    "fairy":    "✨",
}

TYPE_COLORS: dict[str, dict[str, str]] = {
    "normal":   {"bg": "#A8A878", "text": "#fff"},
    "fire":     {"bg": "#F08030", "text": "#fff"},
    "water":    {"bg": "#6890F0", "text": "#fff"},
    "electric": {"bg": "#F8D030", "text": "#333"},
    "grass":    {"bg": "#78C850", "text": "#fff"},
    "ice":      {"bg": "#98D8D8", "text": "#333"},
    "fighting": {"bg": "#C03028", "text": "#fff"},
    "poison":   {"bg": "#A040A0", "text": "#fff"},
    "ground":   {"bg": "#E0C068", "text": "#333"},
    "flying":   {"bg": "#A890F0", "text": "#fff"},
    "psychic":  {"bg": "#F85888", "text": "#fff"},
    "bug":      {"bg": "#A8B820", "text": "#fff"},
    "rock":     {"bg": "#B8A038", "text": "#fff"},
    "ghost":    {"bg": "#705898", "text": "#fff"},
    "dragon":   {"bg": "#7038F8", "text": "#fff"},
    "dark":     {"bg": "#705848", "text": "#fff"},
    "steel":    {"bg": "#B8B8D0", "text": "#333"},
    "fairy":    {"bg": "#EE99AC", "text": "#333"},
}

_DEFAULT = {"bg": "#888888", "text": "#fff"}

# Status condition badge colours.
STATUS_COLORS: dict[str, dict[str, str]] = {
    "burn":      {"bg": "#FF4500", "text": "#fff", "abbr": "BRN"},
    "poison":    {"bg": "#9400D3", "text": "#fff", "abbr": "PSN"},
    "paralysis": {"bg": "#DAA520", "text": "#333", "abbr": "PAR"},
    "sleep":     {"bg": "#708090", "text": "#fff", "abbr": "SLP"},
    "freeze":    {"bg": "#00BFFF", "text": "#333", "abbr": "FRZ"},
}

# Weather badge colours.
WEATHER_COLORS: dict[str, dict[str, str]] = {
    "rain":      {"bg": "#4169E1", "text": "#fff", "icon": "🌧️"},
    "sun":       {"bg": "#FF8C00", "text": "#fff", "icon": "☀️"},
    "sandstorm": {"bg": "#C2A060", "text": "#333", "icon": "🌪️"},
    "hail":      {"bg": "#B0E0E6", "text": "#333", "icon": "🌨️"},
}


def get_type_colors(type_name: str) -> dict[str, str]:
    """Return the bg/text colour dict for a type name (case-insensitive)."""
    return TYPE_COLORS.get((type_name or "").lower().strip(), _DEFAULT)


def type_badge_html(type_name: str, small: bool = False) -> str:
    """Return an HTML <span> badge for the given type name."""
    c = get_type_colors(type_name)
    label = (type_name or "???").capitalize()
    size = "10px" if small else "11px"
    padding = "1px 5px" if small else "2px 8px"
    return (
        f'<span style="background:{c["bg"]};color:{c["text"]};'
        f'font-size:{size};font-weight:bold;padding:{padding};'
        f'border-radius:4px;margin:1px;display:inline-block;">'
        f"{label}</span>"
    )


def status_badge_html(status: str | None) -> str:
    """Return a small HTML badge for a status condition, or empty string."""
    if not status:
        return ""
    info = STATUS_COLORS.get(status, {"bg": "#888", "text": "#fff", "abbr": status[:3].upper()})
    return (
        f'<span style="background:{info["bg"]};color:{info["text"]};'
        f'font-size:10px;font-weight:bold;padding:1px 5px;'
        f'border-radius:4px;margin-left:4px;display:inline-block;">'
        f'{info["abbr"]}</span>'
    )


def weather_badge_html(weather: str | None) -> str:
    """Return a small weather indicator badge, or empty string."""
    if not weather:
        return ""
    info = WEATHER_COLORS.get(weather, {"bg": "#888", "text": "#fff", "icon": "🌀"})
    return (
        f'<span style="background:{info["bg"]};color:{info["text"]};'
        f'font-size:12px;padding:2px 8px;border-radius:6px;'
        f'display:inline-block;margin:2px;">'
        f'{info["icon"]} {weather.capitalize()}</span>'
    )


def hp_bar_color(hp_ratio: float) -> str:
    """Return a CSS colour for an HP bar based on the remaining HP fraction."""
    if hp_ratio > 0.50:
        return "#4CAF50"   # green
    if hp_ratio > 0.25:
        return "#FFC107"   # yellow
    return "#F44336"       # red


def get_type_emoji(type_name: str) -> str:
    """Return the emoji icon for a Pokémon type."""
    return TYPE_EMOJI.get((type_name or "").lower().strip(), "❓")


def move_card_html(
    move: dict,
    effectiveness_label: str = "",
    disabled: bool = False,
) -> str:
    """
    Return an HTML card for a single move, styled with the move's type colour.

    Layout:
      LEFT  — type emoji + type label (small pill)
      RIGHT — move name (bold) + power or "Status"

    An optional effectiveness label is appended as a faint subtitle row.
    """
    type_name  = (move.get("type") or "normal").lower().strip()
    c          = get_type_colors(type_name)
    emoji      = get_type_emoji(type_name)
    type_label = type_name.capitalize()

    move_name  = (move.get("name") or "???").replace("-", " ").title()
    power      = move.get("power")
    pwr_str    = f"PWR {power}" if power else "Status"

    # Effectiveness tint on the right edge
    eff_color_map = {
        "Super effective": "#4CAF50",
        "Not very effective": "#FF7043",
        "Immune": "#9E9E9E",
        "Normal": "rgba(255,255,255,0.55)",
    }
    eff_color  = eff_color_map.get(effectiveness_label, "rgba(255,255,255,0.55)")
    eff_part   = (
        f'<span style="font-size:10px;color:{eff_color};'
        f'margin-left:6px;font-style:italic;">{effectiveness_label}</span>'
        if effectiveness_label else ""
    )

    opacity    = "0.55" if disabled else "1.0"
    bg_dark    = c["bg"]          # solid type colour
    text_col   = c["text"]

    return (
        f'<div style="'
        f'background:{bg_dark};'
        f'color:{text_col};'
        f'border-radius:8px;'
        f'padding:8px 12px;'
        f'display:flex;'
        f'align-items:center;'
        f'justify-content:space-between;'
        f'opacity:{opacity};'
        f'margin-bottom:2px;'
        f'min-height:44px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.3);'
        f'">'
        # Left: type icon + label
        f'<span style="display:flex;align-items:center;gap:5px;">'
        f'<span style="font-size:16px;">{emoji}</span>'
        f'<span style="font-size:10px;font-weight:bold;'
        f'background:rgba(0,0,0,0.20);padding:2px 7px;'
        f'border-radius:10px;">{type_label}</span>'
        f'</span>'
        # Right: move name + power + effectiveness
        f'<span style="text-align:right;">'
        f'<span style="font-size:13px;font-weight:bold;">{move_name}</span>'
        f'<br/>'
        f'<span style="font-size:11px;opacity:0.85;">{pwr_str}</span>'
        f'{eff_part}'
        f'</span>'
        f'</div>'
    )
