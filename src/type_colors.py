# Type colour palette — matches the official Pokémon game colour scheme.
# Used by the dashboard to render coloured type badges on moves and Pokémon.

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
