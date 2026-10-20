"""Brand mark and locally bundled visual asset (no third-party asset requests)."""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

ASSET = Path(__file__).with_name('assets') / 'stadium_night.webp'


@lru_cache(maxsize=1)
def stadium_data_uri() -> str:
    if not ASSET.is_file():
        return ''
    return 'data:image/webp;base64,' + base64.b64encode(ASSET.read_bytes()).decode('ascii')


def brand_icon_html(size: int = 36) -> str:
    """Original football-orbit badge, shared by loading state and app wordmark."""
    n = max(18, min(120, int(size)))
    return (f'<svg class="bp-brand-ball" width="{n}" height="{n}" viewBox="0 0 60 60" '
            'role="img" aria-label="UCL Predictor original football logo" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="30" cy="30" r="27" fill="none" stroke="#91c7fa" stroke-width="1.8"/>'
            '<path d="M23 15L37 15L46 29L39 43L22 43L14 29Z" fill="#eff7ff"/>'
            '<path d="M23 15L15 5M37 15L45 5M46 29L57 28M39 43L44 55M22 43L16 55M14 29L3 28" '
            'stroke="#eff7ff" stroke-width="5" stroke-linecap="round"/>'
            '<path d="M23 15L30 5L37 15M46 29L54 40L39 43M22 43L6 39L14 29" '
            'stroke="#65baff" stroke-width="1.5" fill="none" opacity=".85"/>'
            '</svg>')


def brand_lockup_html() -> str:
    return ('<header class="bp-brand-header" aria-label="UCL Predictor">'
            '<div class="bp-brand-lockup">' + brand_icon_html(35) +
            '<span class="bp-brand-name">UCL <strong>PREDICTOR<span>.</span></strong></span>'
            '<span class="bp-brand-version">V2</span></div>'
            '<span class="bp-brand-caption">MATCHDAY • FORECASTS • EVIDENCE</span></header>')
