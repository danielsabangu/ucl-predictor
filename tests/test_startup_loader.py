from __future__ import annotations

from src.ui.startup_loader import (
    STARTUP_SESSION_KEY,
    build_startup_html,
)


def test_startup_html_is_compact_and_contains_wordmark():

    html = build_startup_html()

    assert "\n" not in html
    assert "ucl-startup-overlay" in html
    assert "ucl-startup-inner" in html
    assert "ucl-startup-fallback" in html

    assert ">UCL<" in html
    assert ">PREDICTOR<" in html

    assert "INITIALISING FORECAST TERMINAL" in html
    assert "MATCH ENGINE // READYING DATA // V2" in html


def test_startup_html_has_balanced_main_div_structure():

    html = build_startup_html()

    assert (
        html.count(
            "<div"
        )
        ==
        html.count(
            "</div>"
        )
    )


def test_startup_session_key_is_stable():

    assert (
        STARTUP_SESSION_KEY
        ==
        "_ucl_predictor_startup_complete"
    )
