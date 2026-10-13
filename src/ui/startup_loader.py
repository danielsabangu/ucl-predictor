from __future__ import annotations

import streamlit as st


# ============================================================
# UCL PREDICTOR V2 — STARTUP LOADER
# ============================================================
#
# Presentation only.
#
# Important implementation detail:
# The complete splash markup is deliberately emitted as one
# compact HTML string. Streamlit's Markdown parser can expose
# indented/multi-line nested HTML as literal <div> text when
# blank lines occur inside raw HTML blocks.
#
# The loading screen uses a CSS wordmark rather than an image
# asset. This removes asset-path/base64 failure modes and keeps
# the startup identity lightweight.
# ============================================================


STARTUP_SESSION_KEY = (
    "_ucl_predictor_startup_complete"
)


def build_startup_html() -> str:
    """
    Return compact, single-line HTML for the startup overlay.

    Keeping this markup free of Markdown-significant indentation
    and blank lines prevents Streamlit from rendering nested HTML
    tags as visible code/text.
    """

    return (
        '<div class="ucl-startup-overlay" '
        'role="status" '
        'aria-live="polite" '
        'aria-label="UCL Predictor is starting">'
        '<div class="ucl-startup-scan"></div>'
        '<div class="ucl-startup-inner">'
        '<div class="ucl-startup-fallback">'
        'UCL'
        '<span>PREDICTOR</span>'
        '</div>'
        '<div class="ucl-startup-copy">'
        'INITIALISING FORECAST TERMINAL'
        '</div>'
        '<div class="ucl-startup-progress">'
        '<div class="ucl-startup-progress-fill"></div>'
        '</div>'
        '<div class="ucl-startup-system">'
        'MATCH ENGINE // READYING DATA // V2'
        '</div>'
        '</div>'
        '</div>'
    )


def show_startup_loading_screen() -> None:
    """
    Render the startup overlay once per Streamlit session.

    CSS controls the visible duration and fade-out. There is no
    blocking sleep and no image file dependency.
    """

    if st.session_state.get(
        STARTUP_SESSION_KEY,
        False,
    ):

        return

    st.session_state[
        STARTUP_SESSION_KEY
    ] = True

    st.markdown(
        build_startup_html(),
        unsafe_allow_html=True,
    )
