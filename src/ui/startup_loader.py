"""Football-night intro with the same original logo as the app header.

No external images and no artificial sleep delay. Displayed once in a
Streamlit session, with a static accessible fallback for reduced motion.
"""
from __future__ import annotations

import streamlit as st
from src.ui.broadcast_identity import brand_icon_html

STARTUP_SESSION_KEY = '_ucl_predictor_startup_complete'


def build_startup_html() -> str:
    ball = (
        '<div class="nf-splash-ball-area" aria-hidden="true">'
        '<div class="nf-ball-trail"></div>'
        '<div class="nf-splash-ball">'
        '<svg viewBox="0 0 144 144" xmlns="http://www.w3.org/2000/svg" focusable="false">'
        '<defs><radialGradient id="bpball" cx="30%" cy="25%" r="80%">'
        '<stop offset="0" stop-color="#f7fcff"/>'
        '<stop offset=".62" stop-color="#d3e7fd"/>'
        '<stop offset="1" stop-color="#2d6dad"/></radialGradient></defs>'
        '<circle cx="72" cy="72" r="64" fill="url(#bpball)" stroke="#bce8ff" stroke-width="2"/>'
        '<path d="M59 29L84 28L106 52L95 82L69 93L43 77L38 52Z" fill="#0c3261" '
        'stroke="#66b2ec" stroke-width="2"/>'
        '<path d="M59 29L43 12M84 28L102 13M106 52L133 50M95 82L115 105M69 93L69 135M43 77L21 97M38 52L12 46" '
        'stroke="#18518b" stroke-width="3" fill="none"/>'
        '<path d="M59 29L75 13M84 28L113 33M106 52L120 78M95 82L88 120M69 93L46 119" '
        'stroke="#659ac9" stroke-width="2" fill="none"/></svg></div></div>'
    )
    return (
        '<div class="ucl-startup-overlay" role="status" aria-live="polite" '
        'aria-label="UCL Predictor is opening the matchday" '
        'data-legacy-title="INITIALISING FORECAST TERMINAL" '
        'data-legacy-progress="MATCH ENGINE // READYING DATA // V2">'
        '<div class="ucl-startup-scan" aria-hidden="true"></div>'
        '<div class="ucl-startup-inner">'
        '<div class="nf-splash-eyebrow">THE LIGHTS ARE COMING ON</div>'
        '<div class="bp-startup-brand">' + brand_icon_html(54) +
        '<span>UCL <strong>PREDICTOR<em>.</em></strong></span></div>'
        '<div class="bp-startup-tagline">THE NIGHT IS OURS.</div>'
        + ball +
        '<div class="ucl-startup-fallback">UCL<span>PREDICTOR</span></div>'
        '<div class="ucl-startup-copy">A NIGHT OF POSSIBILITIES</div>'
        '<div class="ucl-startup-progress"><div class="ucl-startup-progress-fill"></div></div>'
        '<div class="ucl-startup-system">PREPARING YOUR MATCHDAY EXPERIENCE</div>'
        '</div></div>'
    )


def show_startup_loading_screen() -> None:
    if st.session_state.get(STARTUP_SESSION_KEY, False):
        return
    st.session_state[STARTUP_SESSION_KEY] = True
    st.markdown(build_startup_html(), unsafe_allow_html=True)
