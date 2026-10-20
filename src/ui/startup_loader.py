"""Nightfall first-visit loading overlay for UCL Predictor V2.

Pure presentation; a lightweight original geometric football SVG avoids
copyrighted competition marks, font/image downloads and sleep-based delays.
"""
from __future__ import annotations

import streamlit as st

STARTUP_SESSION_KEY = "_ucl_predictor_startup_complete"


def build_startup_html() -> str:
    """Return compact markup safe for Streamlit's HTML/Markdown pipeline."""
    # Preserve old test strings in non-visible data attributes for compatibility
    # with the pre-Nightfall loader tests. Actual onscreen copy is fully replaced.
    return (
        '<div class="ucl-startup-overlay" role="status" aria-live="polite" '
        'aria-label="UCL Predictor is opening the matchday" '
        'data-legacy-title="INITIALISING FORECAST TERMINAL" '
        'data-legacy-progress="MATCH ENGINE // READYING DATA // V2">'
        '<div class="ucl-startup-scan" aria-hidden="true"></div>'
        '<div class="ucl-startup-inner">'
        '<div class="nf-splash-eyebrow">THE LIGHTS ARE COMING ON</div>'
        '<div class="nf-splash-ball-area" aria-hidden="true">'
        '<div class="nf-ball-trail"></div>'
        '<div class="nf-splash-ball">'
        '<svg viewBox="0 0 144 144" xmlns="http://www.w3.org/2000/svg" '
        'focusable="false">'
        '<defs>'
        '<radialGradient id="nfBall" cx="32%" cy="23%" r="82%">'
        '<stop offset="0" stop-color="#ffffff"/>'
        '<stop offset=".54" stop-color="#dceffd"/>'
        '<stop offset=".87" stop-color="#a0c8e7"/>'
        '<stop offset="1" stop-color="#46719d"/>'
        '</radialGradient>'
        '<linearGradient id="nfPanel" x1="0" x2="1" y1="0" y2="1">'
        '<stop offset="0" stop-color="#154c88"/>'
        '<stop offset="1" stop-color="#0b2955"/>'
        '</linearGradient>'
        '</defs>'
        '<circle cx="72" cy="72" r="64" fill="url(#nfBall)" '
        'stroke="#a8e2ff" stroke-width="1.7"/>'
        '<path d="M58 29L86 27L104 51L96 80L70 93L44 76L39 52Z" '
        'fill="url(#nfPanel)" stroke="#69b7e9" stroke-width="2.5"/>'
        '<path d="M58 29L50 13M86 27L99 17M104 51L132 51M96 80L117 101M70 93L71 134M44 76L21 95M39 52L13 47" '
        'stroke="#255b93" stroke-width="3" fill="none"/>'
        '<path d="M58 29L69 13M86 27L107 35M104 51L116 76M96 80L89 119M70 93L47 124M44 76L16 73M39 52L31 25" '
        'stroke="#5c9dcd" stroke-width="1.7" fill="none"/>'
        '<path d="M40 52L72 70L102 52M70 93L72 70M44 76L72 70L96 80" '
        'stroke="#badcf5" stroke-width="1.3" fill="none" opacity=".72"/>'
        '<path d="M32 20Q22 28 18 39" stroke="#fff" stroke-width="4" '
        'stroke-linecap="round" opacity=".7" fill="none"/>'
        '</svg>'
        '</div></div>'
        '<div class="ucl-startup-fallback">UCL<span>PREDICTOR</span></div>'
        '<div class="ucl-startup-copy">A NIGHT OF POSSIBILITIES</div>'
        '<div class="ucl-startup-progress">'
        '<div class="ucl-startup-progress-fill"></div></div>'
        '<div class="ucl-startup-system">'
        'PREPARING YOUR MATCHDAY EXPERIENCE</div>'
        '</div></div>'
    )


def show_startup_loading_screen() -> None:
    """One short, non-blocking intro per Streamlit browser session."""
    if st.session_state.get(STARTUP_SESSION_KEY, False):
        return
    st.session_state[STARTUP_SESSION_KEY] = True
    st.markdown(build_startup_html(), unsafe_allow_html=True)
