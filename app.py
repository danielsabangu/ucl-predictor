from __future__ import annotations

import html
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.production.app_health import (
    evaluate_app_health,
    filter_upcoming_predictions,
)

from src.production.app_sources import (
    resolve_app_data_source,
)

from src.production.club_crests import (
    crest_html,
)

from src.production.receipt_exports import (
    build_receipt_pdf,
    build_receipt_png,
    export_filename,
    margin_pp,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="UCL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLICATION SOURCE
# ============================================================

try:

    APP_SOURCE = (
        resolve_app_data_source()
    )

except Exception as error:

    st.error(
        "UCL Predictor could not resolve "
        "an application data source."
    )

    st.code(
        str(error)
    )

    st.stop()


PREDICTIONS_FILE = (
    APP_SOURCE.predictions_file
)

STATUS_FILE = (
    APP_SOURCE.status_file
)

LOCAL_TZ = ZoneInfo(
    "Africa/Johannesburg"
)


# ============================================================
# HTML
# ============================================================


def render_html(
    markup: str,
) -> None:

    cleaned = textwrap.dedent(
        markup
    ).strip()

    if hasattr(
        st,
        "html",
    ):

        st.html(
            cleaned
        )

    else:

        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


# ============================================================
# CSS
# ============================================================

render_html(
    """
    <style>

    :root {
        --bg: #070707;
        --panel: #101010;
        --receipt: #eee9dc;
        --receipt-text: #101010;
        --border: #292929;

        --green: #8dff63;
        --amber: #ffd666;
        --red: #ff685f;

        --white: #f2f2f2;
    }


    /* ========================================================
       GLOBAL
    ======================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 50% -20%,
                #1d1d1d 0%,
                #0d0d0d 35%,
                #070707 72%
            );
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    .app-kicker,
    .mono,
    .receipt,
    .status-pill,
    .fixture-meta {
        font-family:
            "Courier New",
            "Lucida Console",
            Monaco,
            monospace;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .app-title {
        color: var(--white);

        font-size:
            clamp(
                2.6rem,
                6vw,
                5.4rem
            );

        font-weight: 900;
        letter-spacing: -0.075em;
        line-height: 0.88;
        margin: 0;
        text-transform: uppercase;
    }


    .app-kicker {
        color: #999;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }


    .app-subtitle {
        color: #a5a5a5;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-top: 1.1rem;
        max-width: 680px;
    }


    .top-rule {
        border-top:
            1px solid
            #242424;

        margin:
            1.7rem 0
            1.3rem;
    }


    /* ========================================================
       STATUS
    ======================================================== */

    .status-row {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.5rem 0 1rem;
    }


    .status-pill {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 100px;
        color: #aaa;
        display: inline-block;
        font-size: 0.69rem;
        letter-spacing: 0.06em;
        padding: 0.42rem 0.7rem;
        text-transform: uppercase;
    }


    .status-pill.green {
        background:
            rgba(
                141,
                255,
                99,
                0.06
            );

        border-color:
            rgba(
                141,
                255,
                99,
                0.35
            );

        color:
            var(--green);
    }


    .status-pill.amber {
        background:
            rgba(
                255,
                214,
                102,
                0.06
            );

        border-color:
            rgba(
                255,
                214,
                102,
                0.35
            );

        color:
            var(--amber);
    }


    .status-pill.red {
        background:
            rgba(
                255,
                104,
                95,
                0.06
            );

        border-color:
            rgba(
                255,
                104,
                95,
                0.35
            );

        color:
            var(--red);
    }


    .health-panel {
        align-items: center;

        background:
            rgba(
                255,
                255,
                255,
                0.018
            );

        border:
            1px solid
            #202020;

        border-radius:
            9px;

        display:
            grid;

        gap:
            0.5rem
            1rem;

        grid-template-columns:
            repeat(
                4,
                minmax(
                    0,
                    1fr
                )
            );

        margin:
            0.25rem 0
            1.55rem;

        padding:
            0.85rem
            1rem;
    }


    .health-item {
        min-width: 0;
    }


    .health-label {
        color: #626262;

        font-family:
            "Courier New",
            monospace;

        font-size: 0.57rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }


    .health-value {
        color: #dedede;

        font-family:
            "Courier New",
            monospace;

        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 0.18rem;
        overflow-wrap: anywhere;
    }


    .production-alert {
        border-left: 3px solid;

        font-family:
            "Courier New",
            monospace;

        font-size: 0.7rem;
        line-height: 1.55;

        margin:
            0 0
            1.4rem;

        padding:
            0.75rem
            0.9rem;
    }


    .production-alert.amber {
        background:
            rgba(
                255,
                214,
                102,
                0.05
            );

        border-color:
            var(--amber);

        color:
            #d7c489;
    }


    .production-alert.red {
        background:
            rgba(
                255,
                104,
                95,
                0.05
            );

        border-color:
            var(--red);

        color:
            #d9a3a0;
    }


    /* ========================================================
       SELECT
    ======================================================== */

    div[data-baseweb="select"] > div {
        background-color:
            #111 !important;

        border-color:
            #2c2c2c !important;

        border-radius:
            3px !important;

        color:
            #eee !important;

        min-height:
            48px;
    }


    div[data-baseweb="select"] span {
        color:
            #eee !important;
    }


    [data-testid="stSelectbox"] label {
        color:
            #8a8a8a !important;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.7rem !important;

        letter-spacing:
            0.12em;

        text-transform:
            uppercase;
    }


    /* ========================================================
       TERMINAL
    ======================================================== */

    .terminal-shell {
        background:
            linear-gradient(
                145deg,
                #181818,
                #0d0d0d
            );

        border:
            1px solid
            #282828;

        border-radius:
            24px;

        box-shadow:
            0 28px 80px
            rgba(
                0,
                0,
                0,
                0.55
            ),
            inset
            0 1px
            rgba(
                255,
                255,
                255,
                0.04
            );

        margin:
            1.2rem auto
            0;

        max-width:
            760px;

        padding:
            30px
            clamp(
                16px,
                4vw,
                42px
            )
            38px;
    }


    .terminal-slot {
        background:
            #050505;

        border:
            1px solid
            #222;

        border-radius:
            4px;

        box-shadow:
            inset
            0 5px 14px
            rgba(
                0,
                0,
                0,
                0.9
            );

        height:
            12px;

        margin:
            0 auto
            22px;

        max-width:
            440px;
    }


    .terminal-label-row {
        align-items: center;
        display: flex;
        justify-content: space-between;

        margin:
            0 auto
            12px;

        max-width:
            590px;
    }


    .terminal-machine-label {
        color:
            #565656;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.65rem;

        letter-spacing:
            0.1em;

        text-transform:
            uppercase;
    }


    .terminal-led {
        align-items: center;
        display: flex;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.64rem;

        gap:
            6px;

        text-transform:
            uppercase;
    }


    /* ========================================================
       PRINTER STATUS
    ======================================================== */

    .printer-state {
        height:
            16px;

        min-width:
            115px;

        position:
            relative;

        text-align:
            right;
    }


    .printer-state-printing,
    .printer-state-ready {
        align-items: center;
        display: flex;
        gap: 6px;
        justify-content: flex-end;

        position:
            absolute;

        right: 0;
        top: 0;

        white-space:
            nowrap;
    }


    .printer-state-printing {
        animation:
            printingStateOut
            0.25s
            ease
            1.35s
            forwards;

        color:
            var(--amber);

        opacity:
            1;
    }


    .printer-state-ready {
        animation:
            readyStateIn
            0.35s
            ease
            1.45s
            forwards;

        color:
            var(--green);

        opacity:
            0;
    }


    .printer-dot-printing {
        animation:
            printerPulse
            0.35s
            ease-in-out
            infinite
            alternate;

        background:
            var(--amber);

        border-radius:
            50%;

        box-shadow:
            0 0 7px
            rgba(
                255,
                214,
                102,
                0.55
            );

        display:
            inline-block;

        height:
            6px;

        width:
            6px;
    }


    .printer-dot-ready {
        background:
            var(--green);

        border-radius:
            50%;

        box-shadow:
            0 0 8px
            rgba(
                141,
                255,
                99,
                0.7
            );

        display:
            inline-block;

        height:
            6px;

        width:
            6px;
    }


    /* ========================================================
       PRINT FEED
    ======================================================== */

    .print-stage {
        margin:
            0 auto;

        max-width:
            590px;

        overflow:
            hidden;

        position:
            relative;
    }


    .print-feed {
        animation:
            receiptFeed
            1.55s
            cubic-bezier(
                0.18,
                0.78,
                0.25,
                1
            )
            forwards;

        max-height:
            0;

        overflow:
            hidden;

        position:
            relative;

        transform-origin:
            top center;
    }


    .print-paper {
        animation:
            thermalVibration
            0.10s
            linear
            0s
            12,

            paperSettle
            0.35s
            ease-out
            1.15s
            forwards;

        position:
            relative;
    }


    .print-paper::after {
        animation:
            thermalScan
            1.35s
            linear
            0.08s
            forwards;

        background:
            linear-gradient(
                90deg,
                transparent 0%,
                rgba(25,25,25,0.04) 10%,
                rgba(25,25,25,0.22) 48%,
                rgba(25,25,25,0.04) 90%,
                transparent 100%
            );

        content:
            "";

        height:
            2px;

        left:
            2%;

        opacity:
            0;

        pointer-events:
            none;

        position:
            absolute;

        top:
            0;

        width:
            96%;

        z-index:
            20;
    }


    .print-stage::before {
        animation:
            printerShadow
            1.45s
            ease-out
            forwards;

        background:
            linear-gradient(
                to bottom,
                rgba(0,0,0,0.52),
                rgba(0,0,0,0.16),
                transparent
            );

        content:
            "";

        height:
            24px;

        left:
            0;

        pointer-events:
            none;

        position:
            absolute;

        top:
            0;

        width:
            100%;

        z-index:
            30;
    }


    /* ========================================================
       RECEIPT
    ======================================================== */

    .receipt {
        background:
            linear-gradient(
                90deg,
                rgba(0,0,0,0.015),
                transparent 8%,
                transparent 92%,
                rgba(0,0,0,0.02)
            ),
            var(--receipt);

        box-shadow:
            0 14px 35px
            rgba(
                0,
                0,
                0,
                0.46
            );

        color:
            var(--receipt-text);

        margin:
            0 auto;

        max-width:
            590px;

        overflow:
            hidden;

        padding:
            30px
            clamp(
                18px,
                5vw,
                38px
            )
            28px;

        position:
            relative;
    }


    .receipt::before,
    .receipt::after {
        content:
            "";

        display:
            block;

        height:
            12px;

        left:
            0;

        position:
            absolute;

        width:
            100%;
    }


    .receipt::before {
        background:
            linear-gradient(
                135deg,
                #101010 6px,
                transparent 0
            )
            0 0
            /
            12px 12px
            repeat-x,

            linear-gradient(
                225deg,
                #101010 6px,
                transparent 0
            )
            0 0
            /
            12px 12px
            repeat-x;

        top:
            -7px;
    }


    .receipt::after {
        background:
            linear-gradient(
                45deg,
                #101010 6px,
                transparent 0
            )
            0 0
            /
            12px 12px
            repeat-x,

            linear-gradient(
                315deg,
                #101010 6px,
                transparent 0
            )
            0 0
            /
            12px 12px
            repeat-x;

        bottom:
            -7px;
    }


    .receipt-brand {
        font-size:
            1.3rem;

        font-weight:
            900;

        letter-spacing:
            -0.05em;

        line-height:
            1;

        text-align:
            center;
    }


    .receipt-brand-sub {
        font-size:
            0.62rem;

        letter-spacing:
            0.17em;

        margin-top:
            0.38rem;

        text-align:
            center;

        text-transform:
            uppercase;
    }


    .receipt-id {
        font-size:
            0.64rem;

        margin-top:
            1.15rem;

        text-align:
            center;
    }


    .receipt-rule {
        border-top:
            1px dashed
            rgba(
                15,
                15,
                15,
                0.55
            );

        margin:
            1.15rem
            0;
    }


    .receipt-rule.double {
        border-top:
            3px double
            rgba(
                15,
                15,
                15,
                0.68
            );
    }


    .fixture-meta {
        color:
            #414141;

        font-size:
            0.68rem;

        letter-spacing:
            0.035em;

        line-height:
            1.45;

        margin-bottom:
            1.3rem;

        text-align:
            center;

        text-transform:
            uppercase;
    }


    /* ========================================================
       CLUBS
    ======================================================== */

    .club-lockup {
        align-items:
            end;

        display:
            grid;

        gap:
            0.55rem;

        grid-template-columns:
            minmax(
                0,
                1fr
            )
            auto
            minmax(
                0,
                1fr
            );

        margin:
            0.35rem 0
            0.45rem;
    }


    .club-side {
        align-items:
            center;

        display:
            flex;

        flex-direction:
            column;

        gap:
            0.55rem;

        min-width:
            0;
    }


    .club-crest-wrap {
        align-items:
            center;

        display:
            flex;

        height:
            76px;

        justify-content:
            center;

        width:
            92px;
    }


    .club-crest {
        display:
            block;

        filter:
            grayscale(1)
            contrast(1.07);

        height:
            70px;

        max-width:
            86px;

        object-fit:
            contain;

        opacity:
            0.93;

        width:
            auto;
    }


    .club-crest-fallback {
        align-items:
            center;

        border:
            2px solid
            #111;

        border-radius:
            50%;

        display:
            flex;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.8rem;

        font-weight:
            900;

        height:
            56px;

        justify-content:
            center;

        width:
            56px;
    }


    .club-side-name {
        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size:
            clamp(
                1.05rem,
                3vw,
                1.55rem
            );

        font-weight:
            900;

        letter-spacing:
            -0.055em;

        line-height:
            0.95;

        overflow-wrap:
            anywhere;

        text-align:
            center;

        text-transform:
            uppercase;
    }


    .club-vs {
        align-self:
            center;

        color:
            #77705f;

        font-size:
            0.62rem;

        font-weight:
            800;

        margin-bottom:
            0.8rem;
    }


    /* ========================================================
       SCORE
    ======================================================== */

    .score-lockup {
        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size:
            clamp(
                4rem,
                14vw,
                7rem
            );

        font-weight:
            950;

        letter-spacing:
            -0.1em;

        line-height:
            0.86;

        margin:
            1.15rem 0
            0.6rem;

        padding-right:
            0.08em;

        text-align:
            center;
    }


    .score-caption {
        font-size:
            0.63rem;

        letter-spacing:
            0.13em;

        margin-bottom:
            1rem;

        text-align:
            center;

        text-transform:
            uppercase;
    }


    .model-pick-box {
        border:
            2px solid
            #121212;

        margin:
            1.05rem 0;

        padding:
            0.75rem
            0.8rem;

        text-align:
            center;
    }


    .model-pick-label {
        font-size:
            0.58rem;

        letter-spacing:
            0.13em;

        text-transform:
            uppercase;
    }


    .model-pick {
        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size:
            clamp(
                1.35rem,
                5vw,
                2.15rem
            );

        font-weight:
            950;

        letter-spacing:
            -0.055em;

        line-height:
            1;

        margin-top:
            0.28rem;

        text-transform:
            uppercase;
    }


    /* ========================================================
       H/D/A
    ======================================================== */

    .hda {
        display:
            grid;

        gap:
            0.5rem;

        grid-template-columns:
            repeat(
                3,
                1fr
            );

        margin:
            1.25rem
            0;
    }


    .hda-cell {
        border:
            1px solid
            rgba(
                0,
                0,
                0,
                0.25
            );

        min-width:
            0;

        padding:
            0.7rem
            0.25rem;

        text-align:
            center;
    }


    .hda-cell.winner {
        background:
            #111;

        color:
            var(--receipt);
    }


    .hda-label {
        font-size:
            0.58rem;

        letter-spacing:
            0.12em;

        text-transform:
            uppercase;
    }


    .hda-value {
        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size:
            clamp(
                1.15rem,
                4vw,
                1.65rem
            );

        font-weight:
            900;

        letter-spacing:
            -0.04em;

        margin-top:
            0.14rem;
    }


    /* ========================================================
       CONFIDENCE
    ======================================================== */

    .confidence-wrap {
        margin:
            1rem 0
            0.5rem;
    }


    .confidence-head {
        display:
            flex;

        font-size:
            0.65rem;

        justify-content:
            space-between;

        margin-bottom:
            0.35rem;

        text-transform:
            uppercase;
    }


    .confidence-track {
        background:
            rgba(
                0,
                0,
                0,
                0.13
            );

        height:
            10px;

        overflow:
            hidden;

        width:
            100%;
    }


    .confidence-fill {
        background:
            #111;

        height:
            100%;
    }


    .confidence-note {
        color:
            #4e4a40;

        font-size:
            0.6rem;

        line-height:
            1.5;

        margin-top:
            0.45rem;
    }


    /* ========================================================
       ROWS
    ======================================================== */

    .receipt-row {
        align-items:
            flex-start;

        display:
            flex;

        font-size:
            0.77rem;

        gap:
            1rem;

        justify-content:
            space-between;

        line-height:
            1.45;

        margin:
            0.22rem 0;
    }


    .receipt-row .label {
        color:
            #505050;

        flex:
            0 0 auto;
    }


    .receipt-row .value {
        font-weight:
            700;

        text-align:
            right;
    }


    .warning-strip {
        background:
            repeating-linear-gradient(
                -45deg,
                #111,
                #111 5px,
                #e2dccd 5px,
                #e2dccd 10px
            );

        height:
            4px;

        margin:
            1rem 0
            0.8rem;
    }


    .receipt-warning {
        font-size:
            0.59rem;

        line-height:
            1.5;

        text-align:
            center;

        text-transform:
            uppercase;
    }


    .barcode {
        background:
            repeating-linear-gradient(
                90deg,
                #111 0,
                #111 2px,
                transparent 2px,
                transparent 4px,
                #111 4px,
                #111 5px,
                transparent 5px,
                transparent 8px,
                #111 8px,
                #111 11px,
                transparent 11px,
                transparent 13px
            );

        height:
            46px;

        margin:
            1rem auto
            0.35rem;

        max-width:
            330px;
    }


    .barcode-number {
        font-size:
            0.57rem;

        letter-spacing:
            0.21em;

        text-align:
            center;
    }


    /* ========================================================
       EXPORT
    ======================================================== */

    .export-panel {
        background:
            rgba(
                255,
                255,
                255,
                0.018
            );

        border:
            1px solid
            #222;

        border-radius:
            8px;

        margin:
            1rem auto
            0;

        max-width:
            760px;

        padding:
            0.9rem
            1rem
            1rem;
    }


    .export-title {
        color:
            #777;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.62rem;

        letter-spacing:
            0.14em;

        margin-bottom:
            0.25rem;

        text-transform:
            uppercase;
    }


    .export-note {
        color:
            #555;

        font-size:
            0.67rem;

        line-height:
            1.4;
    }


    div.stDownloadButton > button {
        background:
            var(--receipt) !important;

        border:
            0 !important;

        border-radius:
            3px !important;

        color:
            #111 !important;

        font-family:
            "Courier New",
            monospace !important;

        font-size:
            0.72rem !important;

        font-weight:
            900 !important;

        letter-spacing:
            0.08em !important;

        min-height:
            48px !important;

        text-transform:
            uppercase !important;

        width:
            100% !important;
    }


    div.stDownloadButton > button:hover {
        background:
            #fff !important;

        color:
            #111 !important;
    }


    /* ========================================================
       CONTENT
    ======================================================== */

    .section-title {
        color:
            #e5e5e5;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.68rem;

        letter-spacing:
            0.15em;

        margin:
            2rem 0
            0.75rem;

        text-transform:
            uppercase;
    }


    .dark-card {
        background:
            rgba(
                255,
                255,
                255,
                0.025
            );

        border:
            1px solid
            #232323;

        border-radius:
            10px;

        color:
            #bbb;

        min-height:
            150px;

        padding:
            1rem
            1.05rem;
    }


    .dark-card-label {
        color:
            #686868;

        font-family:
            "Courier New",
            monospace;

        font-size:
            0.62rem;

        letter-spacing:
            0.1em;

        text-transform:
            uppercase;
    }


    .dark-card-value {
        color:
            #efefef;

        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size:
            1.25rem;

        font-weight:
            800;

        letter-spacing:
            -0.04em;

        margin-top:
            0.3rem;
    }


    .dark-card-note {
        color:
            #777;

        font-size:
            0.69rem;

        line-height:
            1.45;

        margin-top:
            0.25rem;
    }


    [data-testid="stExpander"] {
        background:
            rgba(
                255,
                255,
                255,
                0.02
            );

        border:
            1px solid
            #222;

        border-radius:
            6px;
    }


    /* ========================================================
       ANIMATION
    ======================================================== */

    @keyframes receiptFeed {

        0% {
            max-height: 0;
        }

        7% {
            max-height: 60px;
        }

        16% {
            max-height: 150px;
        }

        28% {
            max-height: 300px;
        }

        42% {
            max-height: 520px;
        }

        56% {
            max-height: 760px;
        }

        70% {
            max-height: 1000px;
        }

        84% {
            max-height: 1400px;
        }

        100% {
            max-height: 2800px;
        }
    }


    @keyframes thermalVibration {

        0% {
            transform:
                translateX(0);
        }

        25% {
            transform:
                translateX(-0.65px);
        }

        50% {
            transform:
                translateX(0.7px);
        }

        75% {
            transform:
                translateX(-0.35px);
        }

        100% {
            transform:
                translateX(0);
        }
    }


    @keyframes paperSettle {

        from {
            transform:
                translateY(-2px);
        }

        to {
            transform:
                translateY(0);
        }
    }


    @keyframes thermalScan {

        0% {
            opacity:
                0;

            top:
                0;
        }

        7% {
            opacity:
                0.72;
        }

        90% {
            opacity:
                0.42;
        }

        100% {
            opacity:
                0;

            top:
                calc(
                    100% - 2px
                );
        }
    }


    @keyframes printerShadow {

        0% {
            opacity:
                1;
        }

        70% {
            opacity:
                0.65;
        }

        100% {
            opacity:
                0.18;
        }
    }


    @keyframes printerPulse {

        from {
            opacity:
                0.45;

            transform:
                scale(
                    0.82
                );
        }

        to {
            opacity:
                1;

            transform:
                scale(
                    1.08
                );
        }
    }


    @keyframes printingStateOut {

        from {
            opacity:
                1;
        }

        to {
            opacity:
                0;

            visibility:
                hidden;
        }
    }


    @keyframes readyStateIn {

        from {
            opacity:
                0;
        }

        to {
            opacity:
                1;
        }
    }


    /* ========================================================
       RESPONSIVE
    ======================================================== */

    @media (
        max-width:
        800px
    ) {

        .health-panel {
            grid-template-columns:
                repeat(
                    2,
                    minmax(
                        0,
                        1fr
                    )
                );
        }
    }


    @media (
        max-width:
        700px
    ) {

        .block-container {
            padding-left:
                0.8rem;

            padding-right:
                0.8rem;

            padding-top:
                1rem;
        }


        .app-title {
            font-size:
                3.1rem;
        }


        .terminal-shell {
            border-radius:
                15px;

            padding:
                22px
                10px
                28px;
        }


        .receipt {
            padding-left:
                17px;

            padding-right:
                17px;
        }


        .club-crest-wrap {
            height:
                64px;

            width:
                72px;
        }


        .club-crest {
            height:
                58px;

            max-width:
                68px;
        }


        .club-side-name {
            font-size:
                1rem;
        }


        .score-lockup {
            font-size:
                4.5rem;
        }
    }


    @media (
        max-width:
        460px
    ) {

        .health-panel {
            grid-template-columns:
                1fr
                1fr;

            padding:
                0.7rem;
        }


        .health-value {
            font-size:
                0.66rem;
        }


        .status-row {
            gap:
                0.38rem;
        }


        .status-pill {
            font-size:
                0.58rem;

            padding:
                0.36rem
                0.55rem;
        }


        .club-lockup {
            gap:
                0.25rem;
        }


        .club-crest-wrap {
            height:
                56px;

            width:
                62px;
        }


        .club-crest {
            height:
                50px;

            max-width:
                58px;
        }


        .club-side-name {
            font-size:
                0.82rem;
        }


        .club-vs {
            font-size:
                0.55rem;
        }


        .hda {
            gap:
                0.28rem;
        }


        .hda-value {
            font-size:
                1rem;
        }


        .printer-state {
            min-width:
                95px;
        }
    }


    /* ========================================================
       REDUCED MOTION
    ======================================================== */

    @media (
        prefers-reduced-motion:
        reduce
    ) {

        .print-feed,
        .print-paper,
        .print-paper::after,
        .print-stage::before,
        .printer-state-printing,
        .printer-state-ready,
        .printer-dot-printing {
            animation:
                none !important;
        }


        .print-feed {
            max-height:
                none;
        }


        .printer-state-printing {
            display:
                none;
        }


        .printer-state-ready {
            opacity:
                1;
        }
    }

    </style>
    """
)


# ============================================================
# DATA
# ============================================================


@st.cache_data
def load_predictions(
    path_string: str,
    modified_ns: int,
) -> pd.DataFrame:

    del modified_ns

    path = PREDICTIONS_FILE.__class__(
        path_string
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_result",
        "predicted_selection",
        "max_probability",
        "confidence",
        "confidence_percent",
        "confidence_label",
        "confidence_caps",
        "model_top_two_margin",
        "model_decisiveness_profile",
        "elo_features_available",
        "ucl_form_features_available",
        "domestic_features_available",
        "squad_features_available",
        "overall_features_available",
        "overall_feature_availability",
        "predicted_home_goals",
        "predicted_away_goals",
        "predicted_scoreline",
        "predicted_scoreline_probability",
        "global_modal_scoreline",
        "global_modal_result",
        "score_home_lambda",
        "score_away_lambda",
        "score_total_lambda",
        "score_bridge_max_error_pp",
        "score_bridge_quality",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Prediction artifact is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "kickoff_utc"
    ] = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        errors="raise",
        utc=True,
    )

    return (
        dataframe
        .sort_values(
            [
                "kickoff_utc",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================


def escape(
    value,
) -> str:

    return html.escape(
        str(value)
    )


def percentage(
    value,
) -> str:

    return (
        f"{float(value) * 100:.2f}%"
    )


def display_confidence_label(
    label: str,
) -> str:

    return (
        str(label)
        .replace(
            "_",
            " "
        )
    )


def display_profile(
    profile: str,
) -> str:

    return (
        str(profile)
        .replace(
            "_",
            " "
        )
    )


def match_identifier(
    row: pd.Series,
) -> str:

    kickoff = pd.Timestamp(
        row[
            "kickoff_utc"
        ]
    )

    return (
        f"{kickoff.strftime('%Y%m%d%H%M')}-"
        f"{row['home_team'][:3].upper()}-"
        f"{row['away_team'][:3].upper()}"
    )


def fixture_label(
    row,
) -> str:

    kickoff_local = (
        pd.Timestamp(
            row.kickoff_utc
        )
        .tz_convert(
            LOCAL_TZ
        )
    )

    return (
        f"{kickoff_local.strftime('%H:%M')}"
        f"  •  "
        f"{row.home_team} vs "
        f"{row.away_team}"
    )


def result_class_name(
    result: str,
) -> str:

    if result == "H":
        return "HOME"

    if result == "D":
        return "DRAW"

    return "AWAY"


def winning_cell_class(
    *,
    result: str,
    cell: str,
) -> str:

    if result == cell:

        return (
            "hda-cell winner"
        )

    return (
        "hda-cell"
    )


def confidence_explanation(
    row: pd.Series,
) -> str:

    profile = (
        display_profile(
            row[
                "model_decisiveness_profile"
            ]
        )
    )

    if profile == "CLEAR LEAN":

        text = (
            "The outcome distribution "
            "has a clear leader."
        )

    elif profile == "LEAN":

        text = (
            "The model has a lean, but "
            "alternatives remain material."
        )

    else:

        text = (
            "The outcome distribution "
            "is tight."
        )

    return (
        f"{text} Evidence confidence "
        "reflects data quality, not result "
        "probability."
    )


def health_css_class(
    severity: str,
) -> str:

    severity = (
        str(severity)
        .upper()
    )

    if severity == "GREEN":
        return "green"

    if severity == "AMBER":
        return "amber"

    return "red"


def format_health_finish_time(
    health: dict,
) -> str:

    value = health.get(
        "finished_at_utc"
    )

    if not value:

        return "UNKNOWN"

    try:

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:

            timestamp = (
                timestamp
                .tz_localize(
                    "UTC"
                )
            )

        else:

            timestamp = (
                timestamp
                .tz_convert(
                    "UTC"
                )
            )

        timestamp = (
            timestamp
            .tz_convert(
                LOCAL_TZ
            )
        )

        return timestamp.strftime(
            "%H:%M SAST"
        )

    except Exception:

        return "UNKNOWN"


def format_next_kickoff_local(
    health: dict,
) -> str:

    value = health.get(
        "next_kickoff_utc"
    )

    if not value:

        return "NONE"

    try:

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:

            timestamp = (
                timestamp
                .tz_localize(
                    "UTC"
                )
            )

        else:

            timestamp = (
                timestamp
                .tz_convert(
                    "UTC"
                )
            )

        timestamp = (
            timestamp
            .tz_convert(
                LOCAL_TZ
            )
        )

        return timestamp.strftime(
            "%H:%M SAST"
        )

    except Exception:

        return "UNKNOWN"


# ============================================================
# HEALTH
# ============================================================

health = (
    evaluate_app_health(
        source=APP_SOURCE
    )
)

health_class = (
    health_css_class(
        health[
            "severity"
        ]
    )
)

health_finish_display = (
    format_health_finish_time(
        health
    )
)

next_kickoff_display = (
    format_next_kickoff_local(
        health
    )
)


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="app-kicker">
        UEFA CHAMPIONS LEAGUE // PRODUCTION MODEL
    </div>

    <h1 class="app-title">
        UCL<br>PREDICTOR
    </h1>

    <div class="app-subtitle">
        A chronological pre-match forecasting system using
        Elo, Champions League form, domestic form and squad
        evidence. Raw result probability, evidence confidence
        and scoreline estimation remain separate.
    </div>

    <div class="top-rule"></div>
    """
)


# ============================================================
# HEALTH HEADER
# ============================================================

source_label = (
    "PUBLIC SNAPSHOT"
    if
    APP_SOURCE.is_public_snapshot
    else
    "LOCAL LIVE"
)

render_html(
    f"""
    <div class="status-row">

        <span class="status-pill {health_class}">
            ●
            {escape(health["state"])}
        </span>

        <span class="status-pill">
            PIPELINE
            {escape(health["pipeline_status"])}
        </span>

        <span class="status-pill">
            {escape(source_label)}
        </span>

        <span class="status-pill">
            ucl_predictor_v1
        </span>

        <span class="status-pill">
            {escape(health["upcoming_fixture_count"])}
            UPCOMING
        </span>

        <span class="status-pill">
            290 FEATURES
        </span>

    </div>


    <div class="health-panel">

        <div class="health-item">

            <div class="health-label">
                Last production refresh
            </div>

            <div class="health-value">
                {escape(health_finish_display)}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Data age
            </div>

            <div class="health-value">
                {escape(health["pipeline_age_display"])}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Next kickoff
            </div>

            <div class="health-value">
                {escape(next_kickoff_display)}
            </div>

        </div>


        <div class="health-item">

            <div class="health-label">
                Countdown
            </div>

            <div class="health-value">
                {escape(health["next_kickoff_display"])}
            </div>

        </div>

    </div>
    """
)


# ============================================================
# BLOCK INVALID STATE
# ============================================================

if not health[
    "can_serve_predictions"
]:

    warning_text = (
        " / ".join(
            str(warning)
            for warning in health.get(
                "warnings",
                [],
            )
        )
    )

    render_html(
        f"""
        <div class="production-alert red">

            <strong>
                FORECAST SERVICE BLOCKED
            </strong>

            <br><br>

            State:
            {escape(health["state"])}

            <br>

            Reason:
            {escape(health["reason"])}

            <br>

            Source:
            {escape(health["source_mode"])}

            <br>

            Warnings:
            {escape(warning_text or "UNKNOWN")}

        </div>
        """
    )

    if not APP_SOURCE.is_public_snapshot:

        st.code(
            "python -m src.production.run"
        )

    st.stop()


# ============================================================
# AGING WARNING
# ============================================================

if health[
    "refresh_recommended"
]:

    render_html(
        """
        <div class="production-alert amber">

            <strong>
                REFRESH RECOMMENDED
            </strong>

            <br>

            The current forecast remains usable,
            but production evidence is aging.

        </div>
        """
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

try:

    modified_ns = int(
        PREDICTIONS_FILE
        .stat()
        .st_mtime_ns
    )

    predictions = (
        load_predictions(
            str(
                PREDICTIONS_FILE
            ),
            modified_ns,
        )
    )

except Exception as error:

    st.error(
        "Could not load the selected "
        "prediction artifact."
    )

    st.code(
        str(error)
    )

    st.stop()


predictions = (
    filter_upcoming_predictions(
        predictions
    )
)

if predictions.empty:

    st.error(
        "No upcoming fixtures remain "
        "in this forecast artifact."
    )

    st.stop()


# ============================================================
# MATCHDAY INFO
# ============================================================

matchday_date = (
    predictions[
        "kickoff_utc"
    ]
    .min()
    .tz_convert(
        LOCAL_TZ
    )
    .strftime(
        "%d %B %Y"
    )
)

render_html(
    f"""
    <div class="status-row">

        <span class="status-pill">
            MATCHDAY
            {escape(matchday_date)}
        </span>

        <span class="status-pill">
            RUN
            {escape(
                health["run_id"]
                or
                "UNKNOWN"
            )}
        </span>

        <span class="status-pill">
            SOURCE
            {escape(APP_SOURCE.mode)}
        </span>

    </div>
    """
)


# ============================================================
# SELECTOR
# ============================================================

options = list(
    range(
        len(predictions)
    )
)

selected_index = (
    st.selectbox(
        "Select forecast",
        options,
        format_func=lambda index: (
            fixture_label(
                predictions.iloc[
                    index
                ]
            )
        ),
    )
)

row = predictions.iloc[
    int(selected_index)
]


# ============================================================
# SELECTED VALUES
# ============================================================

kickoff_utc = pd.Timestamp(
    row[
        "kickoff_utc"
    ]
)

kickoff_local = (
    kickoff_utc
    .tz_convert(
        LOCAL_TZ
    )
)

receipt_id = (
    match_identifier(
        row
    )
)

home_probability = float(
    row[
        "home_probability"
    ]
)

draw_probability = float(
    row[
        "draw_probability"
    ]
)

away_probability = float(
    row[
        "away_probability"
    ]
)

confidence_percent = float(
    row[
        "confidence_percent"
    ]
)

confidence_width = max(
    0.0,
    min(
        100.0,
        confidence_percent,
    ),
)

top_two_margin = (
    margin_pp(
        row[
            "model_top_two_margin"
        ]
    )
)

home_class = (
    winning_cell_class(
        result=row[
            "predicted_result"
        ],
        cell="H",
    )
)

draw_class = (
    winning_cell_class(
        result=row[
            "predicted_result"
        ],
        cell="D",
    )
)

away_class = (
    winning_cell_class(
        result=row[
            "predicted_result"
        ],
        cell="A",
    )
)


# ============================================================
# RECEIPT
# ============================================================

receipt_html = f"""
<div class="terminal-shell">

    <div class="terminal-label-row">

        <div class="terminal-machine-label">
            UCL//POS-290
        </div>


        <div class="terminal-led printer-state">

            <div class="printer-state-printing">

                <span class="printer-dot-printing"></span>

                PRINTING...

            </div>


            <div class="printer-state-ready">

                <span class="printer-dot-ready"></span>

                READY

            </div>

        </div>

    </div>


    <div class="terminal-slot"></div>


    <div class="print-stage">

        <div class="print-feed">

            <div class="receipt print-paper">


                <div class="receipt-brand">
                    UCL//FORECAST
                </div>


                <div class="receipt-brand-sub">
                    Production prediction receipt
                </div>


                <div class="receipt-id">
                    TXN {escape(receipt_id)}
                </div>


                <div class="receipt-rule"></div>


                <div class="fixture-meta">

                    {escape(
                        kickoff_local.strftime(
                            "%A %d %B %Y"
                        )
                    )}

                    <br>

                    {escape(
                        kickoff_local.strftime(
                            "%H:%M"
                        )
                    )}

                    SAST

                    //

                    UEFA CHAMPIONS LEAGUE

                </div>


                <div class="club-lockup">

                    <div class="club-side">

                        {crest_html(
                            row["home_team"],
                            side="home",
                        )}

                        <div class="club-side-name">

                            {escape(
                                row[
                                    "home_team"
                                ]
                            )}

                        </div>

                    </div>


                    <div class="club-vs">
                        VS
                    </div>


                    <div class="club-side">

                        {crest_html(
                            row["away_team"],
                            side="away",
                        )}

                        <div class="club-side-name">

                            {escape(
                                row[
                                    "away_team"
                                ]
                            )}

                        </div>

                    </div>

                </div>


                <div class="score-lockup">

                    {int(
                        row[
                            "predicted_home_goals"
                        ]
                    )}

                    -

                    {int(
                        row[
                            "predicted_away_goals"
                        ]
                    )}

                </div>


                <div class="score-caption">
                    estimated scoreline // poisson bridge
                </div>


                <div class="model-pick-box">

                    <div class="model-pick-label">
                        CatBoost result pick
                    </div>

                    <div class="model-pick">

                        {escape(
                            row[
                                "predicted_selection"
                            ]
                        )}

                    </div>

                </div>


                <div class="receipt-rule double"></div>


                <div class="hda">

                    <div class="{home_class}">

                        <div class="hda-label">
                            Home
                        </div>

                        <div class="hda-value">
                            {home_probability * 100:.2f}%
                        </div>

                    </div>


                    <div class="{draw_class}">

                        <div class="hda-label">
                            Draw
                        </div>

                        <div class="hda-value">
                            {draw_probability * 100:.2f}%
                        </div>

                    </div>


                    <div class="{away_class}">

                        <div class="hda-label">
                            Away
                        </div>

                        <div class="hda-value">
                            {away_probability * 100:.2f}%
                        </div>

                    </div>

                </div>


                <div class="receipt-rule"></div>


                <div class="confidence-wrap">

                    <div class="confidence-head">

                        <span>
                            Evidence confidence
                        </span>

                        <span>
                            {confidence_percent:.2f}%
                        </span>

                    </div>


                    <div class="confidence-track">

                        <div
                            class="confidence-fill"
                            style="
                                width:
                                {confidence_width:.2f}%;
                            "
                        >
                        </div>

                    </div>


                    <div class="confidence-note">

                        {escape(
                            display_confidence_label(
                                row[
                                    "confidence_label"
                                ]
                            )
                        )}

                        //

                        {escape(
                            display_profile(
                                row[
                                    "model_decisiveness_profile"
                                ]
                            )
                        )}

                        probability profile

                        //

                        top-two margin

                        {top_two_margin:.2f}

                        pp

                    </div>

                </div>


                <div class="receipt-rule"></div>


                <div class="receipt-row">

                    <div class="label">
                        EXACT SCORE SHARE
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "predicted_scoreline_probability"
                            ]
                        ) * 100:.2f}%

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        EXPECTED GOALS
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "score_home_lambda"
                            ]
                        ):.2f}

                        -

                        {float(
                            row[
                                "score_away_lambda"
                            ]
                        ):.2f}

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        TOTAL GOAL RATE
                    </div>

                    <div class="value">

                        {float(
                            row[
                                "score_total_lambda"
                            ]
                        ):.2f}

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        SCORE BRIDGE
                    </div>

                    <div class="value">

                        {escape(
                            row[
                                "score_bridge_quality"
                            ]
                        )}

                        //

                        {float(
                            row[
                                "score_bridge_max_error_pp"
                            ]
                        ):.3f}

                        PP

                    </div>

                </div>


                <div class="receipt-rule"></div>


                <div class="receipt-row">

                    <div class="label">
                        ELO
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "elo_features_available"
                            ]
                        )}

                        / 6

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        UCL FORM
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "ucl_form_features_available"
                            ]
                        )}

                        / 81

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        DOMESTIC
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "domestic_features_available"
                            ]
                        )}

                        / 102

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        SQUAD
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "squad_features_available"
                            ]
                        )}

                        / 101

                    </div>

                </div>


                <div class="receipt-row">

                    <div class="label">
                        TOTAL DATA
                    </div>

                    <div class="value">

                        {int(
                            row[
                                "overall_features_available"
                            ]
                        )}

                        / 290

                        //

                        {float(
                            row[
                                "overall_feature_availability"
                            ]
                        ) * 100:.2f}%

                    </div>

                </div>


                <div class="warning-strip"></div>


                <div class="receipt-warning">

                    Probability ≠ confidence.

                    Scoreline is an auxiliary estimate,
                    not a separately trained goal model.

                    Recent squad usage is intentionally missing
                    where no trustworthy live source was available.

                </div>


                <div class="barcode"></div>


                <div class="barcode-number">

                    {escape(
                        receipt_id.replace(
                            "-",
                            "",
                        )
                    )}

                </div>


            </div>

        </div>

    </div>

</div>
"""

render_html(
    receipt_html
)


# ============================================================
# EXPORTS
# ============================================================

render_html(
    """
    <div class="export-panel">

        <div class="export-title">
            TAKE YOUR FORECAST
        </div>

        <div class="export-note">
            Download the selected forecast as a
            shareable thermal receipt.
        </div>

    </div>
    """
)


try:

    png_bytes = (
        build_receipt_png(
            row
        )
    )

    pdf_bytes = (
        build_receipt_pdf(
            row
        )
    )

except Exception as error:

    st.error(
        "Could not generate the "
        "forecast exports."
    )

    st.code(
        str(error)
    )

    png_bytes = None
    pdf_bytes = None


download_col_1, download_col_2 = (
    st.columns(
        2
    )
)


with download_col_1:

    if png_bytes is not None:

        st.download_button(
            label="DOWNLOAD PNG",
            data=png_bytes,
            file_name=(
                export_filename(
                    row,
                    extension="png",
                )
            ),
            mime="image/png",
            use_container_width=True,
        )


with download_col_2:

    if pdf_bytes is not None:

        st.download_button(
            label="DOWNLOAD PDF",
            data=pdf_bytes,
            file_name=(
                export_filename(
                    row,
                    extension="pdf",
                )
            ),
            mime="application/pdf",
            use_container_width=True,
        )


# ============================================================
# CONTEXT
# ============================================================

render_html(
    """
    <div class="section-title">
        Forecast context
    </div>
    """
)

col1, col2, col3 = (
    st.columns(
        3
    )
)


with col1:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Result probability
            </div>

            <div class="dark-card-value">

                {float(
                    row[
                        "max_probability"
                    ]
                ) * 100:.2f}%

            </div>

            <div class="dark-card-note">

                Probability assigned to the selected
                H/D/A result by the frozen CatBoost
                classifier.

            </div>

        </div>
        """
    )


with col2:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Evidence confidence
            </div>

            <div class="dark-card-value">
                {confidence_percent:.2f}%
            </div>

            <div class="dark-card-note">

                {escape(
                    confidence_explanation(
                        row
                    )
                )}

            </div>

        </div>
        """
    )


with col3:

    render_html(
        f"""
        <div class="dark-card">

            <div class="dark-card-label">
                Score estimate
            </div>

            <div class="dark-card-value">

                {escape(
                    row[
                        "predicted_scoreline"
                    ]
                )}

            </div>

            <div class="dark-card-note">

                Class-consistent Poisson estimate.

                Exact-score mass:

                {float(
                    row[
                        "predicted_scoreline_probability"
                    ]
                ) * 100:.2f}%.

            </div>

        </div>
        """
    )


# ============================================================
# MODAL SCORE NOTE
# ============================================================

if (
    str(
        row[
            "global_modal_scoreline"
        ]
    )
    !=
    str(
        row[
            "predicted_scoreline"
        ]
    )
):

    st.info(
        "The unconstrained Poisson modal score is "
        f"{row['global_modal_scoreline']}, while the "
        f"displayed {row['predicted_scoreline']} score "
        "is constrained to CatBoost's "
        f"{result_class_name(row['predicted_result'])} "
        "result class."
    )


# ============================================================
# DETAILS
# ============================================================

with st.expander(
    "Model & evidence detail"
):

    detail_data = pd.DataFrame(
        [
            {
                "Item":
                    "Home probability",

                "Value":
                    percentage(
                        row[
                            "home_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Draw probability",

                "Value":
                    percentage(
                        row[
                            "draw_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Away probability",

                "Value":
                    percentage(
                        row[
                            "away_probability"
                        ]
                    ),
            },

            {
                "Item":
                    "Model pick",

                "Value":
                    row[
                        "predicted_selection"
                    ],
            },

            {
                "Item":
                    "Top-two margin",

                "Value":
                    f"{top_two_margin:.2f} pp",
            },

            {
                "Item":
                    "Probability profile",

                "Value":
                    display_profile(
                        row[
                            "model_decisiveness_profile"
                        ]
                    ),
            },

            {
                "Item":
                    "Evidence confidence",

                "Value":
                    f"{confidence_percent:.2f}%",
            },

            {
                "Item":
                    "Confidence label",

                "Value":
                    display_confidence_label(
                        row[
                            "confidence_label"
                        ]
                    ),
            },

            {
                "Item":
                    "Confidence caps",

                "Value":
                    str(
                        row[
                            "confidence_caps"
                        ]
                    ).replace(
                        "|",
                        " / ",
                    ),
            },

            {
                "Item":
                    "Score estimate",

                "Value":
                    row[
                        "predicted_scoreline"
                    ],
            },

            {
                "Item":
                    "Home Poisson λ",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_home_lambda'
                            ]
                        ):.3f}"
                    ),
            },

            {
                "Item":
                    "Away Poisson λ",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_away_lambda'
                            ]
                        ):.3f}"
                    ),
            },

            {
                "Item":
                    "Bridge max error",

                "Value":
                    (
                        f"{float(
                            row[
                                'score_bridge_max_error_pp'
                            ]
                        ):.3f} pp"
                    ),
            },

            {
                "Item":
                    "Application source",

                "Value":
                    APP_SOURCE.mode,
            },
        ]
    )

    st.dataframe(
        detail_data,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Methodology"
):

    st.markdown(
        """
**Result probabilities**

The H/D/A distribution comes directly from the frozen
`ucl_predictor_v1` CatBoost production pipeline using exactly
290 features:

- 6 Elo features
- 81 Champions League form features
- 102 domestic-form features
- 101 squad features

**Evidence confidence**

Confidence is separate from probability. It summarizes how
complete and trustworthy the evidence is for the fixture.

Missing live squad-usage evidence lowers confidence rather than
being fabricated.

**Scoreline**

The displayed score is an auxiliary independent-Poisson bridge
fitted to the CatBoost H/D/A probabilities.

The displayed exact score is constrained to agree with
CatBoost's predicted result.

**Production health**

The application validates its prediction source before serving
forecasts.

Failed, stale or expired artifacts are blocked rather than
silently displayed.

**Deployment**

The public deployment consumes a sanitized forecast snapshot.
The training data, production model artifact, raw datasets and
API credentials are not required by the public interface.

**Exports**

PNG and PDF receipts are generated from the already-computed
forecast. Export generation does not alter model probabilities.

**Leakage rule**

All forecasting features are pre-match.

Future match goals, results and scorelines are not prediction
inputs.
        """
    )


# ============================================================
# MATCHDAY BOARD
# ============================================================

render_html(
    """
    <div class="section-title">
        Matchday board
    </div>
    """
)

board = predictions[
    [
        "kickoff_utc",
        "home_team",
        "away_team",
        "predicted_scoreline",
        "predicted_selection",
        "home_probability",
        "draw_probability",
        "away_probability",
        "confidence_percent",
    ]
].copy()


board[
    "kickoff"
] = (
    board[
        "kickoff_utc"
    ]
    .dt.tz_convert(
        LOCAL_TZ
    )
    .dt.strftime(
        "%H:%M"
    )
)


board[
    "HOME"
] = (
    board[
        "home_probability"
    ]
    *
    100
).round(
    1
)


board[
    "DRAW"
] = (
    board[
        "draw_probability"
    ]
    *
    100
).round(
    1
)


board[
    "AWAY"
] = (
    board[
        "away_probability"
    ]
    *
    100
).round(
    1
)


board[
    "CONF"
] = (
    board[
        "confidence_percent"
    ]
    .round(
        1
    )
)


board = board.rename(
    columns={
        "home_team":
            "HOME TEAM",

        "away_team":
            "AWAY TEAM",

        "predicted_scoreline":
            "SCORE",

        "predicted_selection":
            "PICK",
    }
)


board = board[
    [
        "kickoff",
        "HOME TEAM",
        "AWAY TEAM",
        "SCORE",
        "PICK",
        "HOME",
        "DRAW",
        "AWAY",
        "CONF",
    ]
]


st.dataframe(
    board,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

file_modified = (
    datetime.fromtimestamp(
        PREDICTIONS_FILE
        .stat()
        .st_mtime,
        tz=LOCAL_TZ,
    )
)

render_html(
    f"""
    <div
        style="
            border-top:
                1px solid
                #222;

            color:
                #555;

            font-family:
                'Courier New',
                monospace;

            font-size:
                0.62rem;

            line-height:
                1.6;

            margin-top:
                2rem;

            padding-top:
                1rem;

            text-transform:
                uppercase;
        "
    >

        ucl_predictor_v1

        //

        frozen catboost

        //

        290 features

        <br>

        pipeline:
        {escape(
            health[
                "pipeline_status"
            ]
        )}

        //

        health:
        {escape(
            health[
                "state"
            ]
        )}

        //

        source:
        {escape(
            APP_SOURCE.mode
        )}

        //

        run:
        {escape(
            health[
                "run_id"
            ]
            or
            "UNKNOWN"
        )}

        <br>

        forecast artifact updated

        {escape(
            file_modified.strftime(
                "%d %b %Y %H:%M SAST"
            )
        )}

        <br>

        For analytical / portfolio use.

        Probability is not certainty.

    </div>
    """
)