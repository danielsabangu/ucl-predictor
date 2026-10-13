from __future__ import annotations


# ============================================================
# UCL PREDICTOR V2 — VISUAL SYSTEM
# ============================================================
#
# Presentation only.
#
# This module must not:
# - load prediction data
# - alter CatBoost probabilities
# - alter confidence
# - alter scoreline logic
# - perform application routing
# ============================================================


V2_CSS = r"""
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
   MAIN APP WORDMARK
======================================================== */

.main-app-brand {
    align-items:
        center;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        center;

    margin:
        0.4rem auto
        1.15rem;

    text-align:
        center;
}


.main-app-wordmark-ucl {
    color:
        #f0eee7;

    font-family:
        "Arial Black",
        Arial,
        sans-serif;

    font-size:
        clamp(
            2.8rem,
            6vw,
            4.8rem
        );

    font-weight:
        900;

    letter-spacing:
        -0.065em;

    line-height:
        0.86;

    text-transform:
        uppercase;
}


.main-app-wordmark-predictor {
    color:
        var(--green);

    font-family:
        "Courier New",
        monospace;

    font-size:
        clamp(
            0.88rem,
            2vw,
            1.28rem
        );

    font-weight:
        900;

    letter-spacing:
        0.25em;

    margin-left:
        0.25em;

    margin-top:
        0.42rem;

    text-transform:
        uppercase;
}


.main-app-kicker {
    margin-bottom:
        0;

    text-align:
        center;
}


.main-app-subtitle {
    margin-left:
        auto;

    margin-right:
        auto;

    text-align:
        center;
}


.consumer-status-row {
    justify-content:
        center;
}


.creator-status-pill {
    border-style:
        dashed;
}


.app-footer {
    border-top:
        1px solid
        #222;

    color:
        #555;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.62rem;

    line-height:
        1.6;

    margin-top:
        2rem;

    padding-top:
        1rem;

    text-align:
        center;

    text-transform:
        uppercase;
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


.score-probability-line {
    align-items:
        center;

    display:
        flex;

    font-size:
        0.61rem;

    justify-content:
        center;

    letter-spacing:
        0.08em;

    margin:
        -0.2rem 0
        1rem;

    text-transform:
        uppercase;
}


.score-probability-line strong {
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    font-size:
        0.9rem;

    letter-spacing:
        -0.02em;

    margin-right:
        0.35rem;
}


.score-distribution {
    border:
        1px solid
        rgba(
            0,
            0,
            0,
            0.22
        );

    margin:
        1.05rem 0;

    padding:
        0.85rem
        0.8rem
        0.75rem;
}


.score-distribution-head {
    align-items:
        flex-end;

    display:
        flex;

    gap:
        0.8rem;

    justify-content:
        space-between;

    margin-bottom:
        0.7rem;
}


.score-distribution-title {
    font-size:
        0.62rem;

    font-weight:
        900;

    letter-spacing:
        0.12em;

    text-transform:
        uppercase;
}


.score-distribution-subtitle {
    color:
        #5a564d;

    font-size:
        0.53rem;

    line-height:
        1.35;

    max-width:
        220px;

    text-align:
        right;

    text-transform:
        uppercase;
}


.score-distribution-row {
    align-items:
        center;

    display:
        grid;

    gap:
        0.45rem;

    grid-template-columns:
        42px
        minmax(
            0,
            1fr
        )
        54px;

    margin:
        0.42rem 0;
}


.score-distribution-score {
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    font-size:
        0.8rem;

    font-weight:
        900;

    letter-spacing:
        -0.03em;
}


.score-distribution-track {
    background:
        rgba(
            0,
            0,
            0,
            0.11
        );

    height:
        8px;

    overflow:
        hidden;

    position:
        relative;
}


.score-distribution-fill {
    background:
        #6f6a5e;

    height:
        100%;

    min-width:
        2px;
}


.score-distribution-row.projected
.score-distribution-fill {
    background:
        #111;
}


.score-distribution-probability {
    font-size:
        0.64rem;

    font-weight:
        800;

    text-align:
        right;
}


.score-distribution-tags {
    grid-column:
        2 / 4;

    margin-top:
        -0.24rem;
}


.score-distribution-tag {
    border:
        1px solid
        rgba(
            0,
            0,
            0,
            0.38
        );

    display:
        inline-block;

    font-size:
        0.47rem;

    font-weight:
        900;

    letter-spacing:
        0.08em;

    margin-right:
        0.28rem;

    padding:
        0.12rem
        0.25rem;

    text-transform:
        uppercase;
}


.score-distribution-note {
    border-top:
        1px dashed
        rgba(
            0,
            0,
            0,
            0.3
        );

    color:
        #514d44;

    font-size:
        0.54rem;

    line-height:
        1.45;

    margin-top:
        0.7rem;

    padding-top:
        0.62rem;

    text-transform:
        uppercase;
}


.goal-range-grid {
    display:
        grid;

    gap:
        0.55rem;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    margin-top:
        0.75rem;
}


.goal-range-cell {
    border:
        1px solid
        rgba(
            0,
            0,
            0,
            0.2
        );

    min-width:
        0;

    padding:
        0.55rem
        0.45rem;

    text-align:
        center;
}


.goal-range-label {
    font-size:
        0.5rem;

    letter-spacing:
        0.08em;

    line-height:
        1.25;

    overflow-wrap:
        anywhere;

    text-transform:
        uppercase;
}


.goal-range-value {
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    font-size:
        1.15rem;

    font-weight:
        950;

    letter-spacing:
        -0.04em;

    margin-top:
        0.18rem;
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
        max-height: 3800px;
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
   STAGE 6F — EVIDENCE VAULT
======================================================== */

.vault-summary-grid {
    display:
        grid;

    gap:
        0.75rem;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    margin:
        1rem 0;
}


.vault-summary-card {
    background:
        rgba(
            255,
            255,
            255,
            0.025
        );

    border:
        1px solid
        #292929;

    border-radius:
        8px;

    min-width:
        0;

    padding:
        1rem;
}


.vault-summary-card span,
.vault-record-label,
.vault-compare-card span,
.vault-proof-grid span,
.vault-detail-kicker {
    color:
        #6d6d6d;

    display:
        block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.55rem;

    letter-spacing:
        0.10em;

    text-transform:
        uppercase;
}


.vault-summary-card strong {
    color:
        #f0f0f0;

    display:
        block;

    font-size:
        clamp(
            1.6rem,
            4vw,
            2.4rem
        );

    font-weight:
        900;

    letter-spacing:
        -0.055em;

    line-height:
        1;

    margin-top:
        0.35rem;
}


.vault-summary-card small {
    color:
        #555;

    display:
        block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.48rem;

    letter-spacing:
        0.08em;

    margin-top:
        0.42rem;
}


.vault-sample-warning {
    background:
        rgba(
            255,
            214,
            102,
            0.055
        );

    border:
        1px solid
        rgba(
            255,
            214,
            102,
            0.22
        );

    border-radius:
        7px;

    color:
        #91866c;

    font-size:
        0.72rem;

    line-height:
        1.55;

    margin:
        0.8rem 0
        1.1rem;

    padding:
        0.85rem
        0.95rem;
}


.vault-sample-warning strong {
    color:
        var(--amber);

    display:
        inline-block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.61rem;

    letter-spacing:
        0.08em;

    margin-right:
        0.45rem;
}



.vault-provenance {
    color:
        #7f7f7f;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.49rem;

    font-weight:
        900;

    letter-spacing:
        0.08em;

    margin-top:
        0.55rem;

    text-transform:
        uppercase;
}


.vault-provenance-detail {
    color:
        var(--green);

    margin-top:
        0.42rem;
}


.vault-provenance-summary {
    align-items:
        center;

    color:
        #656565;

    display:
        flex;

    flex-wrap:
        wrap;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.5rem;

    gap:
        0.5rem
        1rem;

    justify-content:
        center;

    letter-spacing:
        0.06em;

    margin:
        0.5rem 0
        1rem;

    text-transform:
        uppercase;
}


.vault-proof-source {
    border-top:
        1px solid
        rgba(
            141,
            255,
            99,
            0.12
        );

    color:
        #64725e;

    font-size:
        0.65rem;

    line-height:
        1.5;

    margin-top:
        0.8rem;

    padding-top:
        0.7rem;
}


.vault-filter-heading {
    color:
        #656565;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.56rem;

    letter-spacing:
        0.12em;

    margin:
        1.15rem 0
        0.45rem;

    text-transform:
        uppercase;
}


.vault-record-card {
    background:
        linear-gradient(
            145deg,
            rgba(
                255,
                255,
                255,
                0.028
            ),
            rgba(
                255,
                255,
                255,
                0.01
            )
        );

    border:
        1px solid
        #292929;

    border-radius:
        9px;

    margin-top:
        0.85rem;

    min-height:
        330px;

    padding:
        0.95rem;
}


.vault-record-card.correct {
    border-color:
        rgba(
            141,
            255,
            99,
            0.24
        );
}


.vault-record-card.missed {
    border-color:
        rgba(
            255,
            104,
            95,
            0.22
        );
}


.vault-record-topline {
    align-items:
        center;

    color:
        #666;

    display:
        flex;

    flex-wrap:
        wrap;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.51rem;

    gap:
        0.5rem;

    justify-content:
        space-between;

    letter-spacing:
        0.06em;
}


.vault-outcome-badge {
    border:
        1px solid
        #333;

    border-radius:
        999px;

    display:
        inline-flex;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.49rem;

    font-weight:
        900;

    letter-spacing:
        0.06em;

    padding:
        0.27rem
        0.48rem;

    text-transform:
        uppercase;
}


.vault-outcome-badge.correct {
    background:
        rgba(
            141,
            255,
            99,
            0.065
        );

    border-color:
        rgba(
            141,
            255,
            99,
            0.28
        );

    color:
        var(--green);
}


.vault-outcome-badge.missed {
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
            0.27
        );

    color:
        var(--red);
}


.vault-record-fixture,
.vault-final-score {
    align-items:
        center;

    display:
        grid;

    gap:
        0.65rem;

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

    margin-top:
        1rem;
}


.vault-record-team,
.vault-final-score > div {
    min-width:
        0;

    text-align:
        center;
}


.vault-record-team
.club-crest-wrap,
.vault-final-score
.club-crest-wrap {
    height:
        58px;

    margin:
        0 auto
        0.4rem;

    width:
        64px;
}


.vault-record-team
.club-crest,
.vault-final-score
.club-crest {
    height:
        52px;

    max-width:
        58px;
}


.vault-record-team span,
.vault-final-score > div > span {
    color:
        #dfdfdf;

    display:
        block;

    font-size:
        0.73rem;

    font-weight:
        800;

    line-height:
        1.15;

    overflow-wrap:
        anywhere;
}


.vault-record-score {
    color:
        #f3f3f3;

    font-family:
        "Arial Black",
        Arial,
        sans-serif;

    font-size:
        1.5rem;

    font-weight:
        900;

    letter-spacing:
        -0.07em;
}


.vault-record-grid {
    border-top:
        1px solid
        #272727;

    display:
        grid;

    gap:
        0.55rem;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    margin-top:
        1rem;

    padding-top:
        0.8rem;
}


.vault-record-grid > div {
    min-width:
        0;
}


.vault-record-grid strong {
    color:
        #eeeeee;

    display:
        block;

    font-size:
        0.77rem;

    line-height:
        1.15;

    margin-top:
        0.2rem;

    overflow-wrap:
        anywhere;
}


.vault-record-grid small {
    color:
        #555;

    display:
        block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.43rem;

    line-height:
        1.35;

    margin-top:
        0.22rem;
}


.vault-proof-line {
    color:
        #59744e;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.5rem;

    letter-spacing:
        0.05em;

    margin-top:
        0.9rem;

    text-align:
        center;
}


.vault-empty {
    border:
        1px dashed
        #323232;

    border-radius:
        8px;

    margin-top:
        1rem;

    padding:
        2rem
        1rem;

    text-align:
        center;
}


.vault-empty-title {
    color:
        #cfcfcf;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.72rem;

    font-weight:
        900;

    letter-spacing:
        0.10em;
}


.vault-empty-copy {
    color:
        #666;

    font-size:
        0.72rem;

    margin:
        0.5rem auto
        0;

    max-width:
        520px;
}


/* ========================================================
   VAULT DETAIL
======================================================== */

.vault-detail {
    background:
        rgba(
            255,
            255,
            255,
            0.018
        );

    border:
        1px solid
        #292929;

    border-radius:
        10px;

    margin:
        0.75rem 0;

    padding:
        clamp(
            1rem,
            3vw,
            1.5rem
        );
}


.vault-detail-head {
    align-items:
        flex-start;

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        0.75rem;

    justify-content:
        space-between;
}


.vault-detail-title {
    color:
        #f0f0f0;

    font-size:
        clamp(
            1.25rem,
            3vw,
            1.9rem
        );

    font-weight:
        900;

    letter-spacing:
        -0.045em;

    margin-top:
        0.25rem;
}


.vault-final-score {
    margin:
        1.3rem auto;

    max-width:
        610px;
}


.vault-final-score > strong {
    color:
        #f3f3f3;

    font-family:
        "Arial Black",
        Arial,
        sans-serif;

    font-size:
        clamp(
            2.6rem,
            8vw,
            4.6rem
        );

    letter-spacing:
        -0.08em;
}


.vault-compare-grid {
    display:
        grid;

    gap:
        0.65rem;

    grid-template-columns:
        repeat(
            4,
            minmax(
                0,
                1fr
            )
        );

    margin-top:
        1.1rem;
}


.vault-compare-card {
    background:
        #0e0e0e;

    border:
        1px solid
        #252525;

    border-radius:
        7px;

    min-width:
        0;

    padding:
        0.8rem;
}


.vault-compare-card strong {
    color:
        #efefef;

    display:
        block;

    font-size:
        1rem;

    font-weight:
        900;

    line-height:
        1.15;

    margin-top:
        0.28rem;

    overflow-wrap:
        anywhere;
}


.vault-compare-card small {
    color:
        #555;

    display:
        block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.45rem;

    margin-top:
        0.32rem;
}


.vault-proof-panel {
    background:
        rgba(
            141,
            255,
            99,
            0.035
        );

    border:
        1px solid
        rgba(
            141,
            255,
            99,
            0.20
        );

    border-radius:
        8px;

    margin-top:
        1rem;

    padding:
        0.9rem;
}


.vault-proof-title {
    color:
        var(--green);

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.62rem;

    font-weight:
        900;

    letter-spacing:
        0.09em;
}


.vault-proof-grid {
    display:
        grid;

    gap:
        0.65rem;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    margin-top:
        0.75rem;
}


.vault-proof-grid > div {
    min-width:
        0;
}


.vault-proof-grid strong {
    color:
        #dedede;

    display:
        block;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.62rem;

    line-height:
        1.4;

    margin-top:
        0.2rem;

    overflow-wrap:
        anywhere;
}


@media (
    max-width:
    760px
) {

    .vault-summary-grid,
    .vault-compare-grid,
    .vault-proof-grid {
        grid-template-columns:
            1fr;
    }


    .vault-record-grid {
        grid-template-columns:
            1fr;
    }


    .vault-record-card {
        min-height:
            0;
    }
}


/* ========================================================
   STAGE 6E — SCREEN-BASED PRODUCT FLOW
======================================================== */

.v2-page-shell {
    margin:
        1.3rem
        0
        1.2rem;
}


div[data-testid="stHorizontalBlock"]:has(
    button[kind="primary"]
) {
    gap:
        0.5rem;
}


button[kind="primary"] {
    background:
        rgba(
            141,
            255,
            99,
            0.08
        )
        !important;

    border-color:
        rgba(
            141,
            255,
            99,
            0.55
        )
        !important;

    color:
        var(--green)
        !important;
}


button:disabled {
    cursor:
        not-allowed
        !important;

    opacity:
        0.34
        !important;
}


.printer-page-intro,
.evidence-page-intro {
    margin-bottom:
        0.35rem;
}


.printer-page-intro
.interaction-title,
.evidence-page-intro
.interaction-title {
    font-size:
        clamp(
            1.25rem,
            3vw,
            1.8rem
        );
}


@media (
    max-width:
    700px
) {

    .v2-page-shell {
        margin-top:
            0.9rem;
    }
}


/* ========================================================
   STAGE 6D — 3D FORECAST PRINTER HERO
======================================================== */

.printer-hero {
    margin:
        1.5rem auto
        0;

    max-width:
        900px;

    position:
        relative;
}


.printer-3d-scene {
    margin:
        0 auto;

    max-width:
        760px;

    padding:
        34px
        54px
        22px;

    perspective:
        1250px;

    perspective-origin:
        50%
        36%;

    position:
        relative;
}


.printer-3d-unit {
    margin:
        0 auto;

    max-width:
        610px;

    position:
        relative;

    transform:
        rotateX(
            4deg
        )
        rotateY(
            -5.5deg
        );

    transform-style:
        preserve-3d;

    width:
        100%;
}


.printer-3d-front {
    background:
        linear-gradient(
            155deg,
            #202020
            0%,
            #151515
            38%,
            #0d0d0d
            100%
        );

    border:
        1px solid
        #303030;

    border-radius:
        20px
        20px
        15px
        15px;

    box-shadow:
        inset
        0 1px
        rgba(
            255,
            255,
            255,
            0.06
        ),
        inset
        0 -20px
        38px
        rgba(
            0,
            0,
            0,
            0.34
        ),
        0 35px 80px
        rgba(
            0,
            0,
            0,
            0.58
        );

    min-height:
        310px;

    padding:
        22px
        28px
        28px;

    position:
        relative;

    transform:
        translateZ(
            28px
        );

    z-index:
        5;
}


.printer-3d-top {
    background:
        linear-gradient(
            180deg,
            #2d2d2d,
            #171717
        );

    border:
        1px solid
        #383838;

    border-radius:
        18px
        18px
        8px
        8px;

    height:
        76px;

    left:
        7%;

    position:
        absolute;

    right:
        7%;

    top:
        -49px;

    transform:
        rotateX(
            72deg
        )
        translateZ(
            35px
        );

    transform-origin:
        bottom center;

    z-index:
        3;
}


.printer-top-ridge {
    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(
                255,
                255,
                255,
                0.08
            ),
            transparent
        );

    height:
        1px;

    left:
        9%;

    position:
        absolute;

    right:
        9%;

    top:
        24px;
}


.printer-3d-side {
    background:
        linear-gradient(
            180deg,
            #111,
            #070707
        );

    border:
        1px solid
        #262626;

    border-radius:
        0 13px 13px 0;

    bottom:
        20px;

    position:
        absolute;

    right:
        -43px;

    top:
        20px;

    transform:
        rotateY(
            80deg
        )
        translateZ(
            23px
        );

    transform-origin:
        left center;

    width:
        72px;

    z-index:
        1;
}


.printer-side-vent {
    background:
        #292929;

    border-radius:
        999px;

    height:
        3px;

    margin:
        18px auto;

    opacity:
        0.75;

    width:
        32px;
}


.printer-floor-shadow {
    background:
        radial-gradient(
            ellipse,
            rgba(
                0,
                0,
                0,
                0.68
            ) 0%,
            rgba(
                0,
                0,
                0,
                0.32
            ) 44%,
            transparent
            72%
        );

    bottom:
        -3px;

    filter:
        blur(
            10px
        );

    height:
        70px;

    left:
        12%;

    position:
        absolute;

    right:
        8%;

    transform:
        rotate(
            -1deg
        );

    z-index:
        0;
}


.printer-front-header {
    align-items:
        center;

    display:
        flex;

    gap:
        1rem;

    justify-content:
        space-between;
}


.printer-engraving {
    color:
        #7d7d7d;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.64rem;

    font-weight:
        900;

    letter-spacing:
        0.11em;

    text-transform:
        uppercase;
}


.printer-engraving span {
    color:
        #414141;

    margin-left:
        0.45rem;
}


.printer-led-bank {
    align-items:
        center;

    display:
        flex;

    gap:
        7px;
}


.printer-led {
    background:
        #343434;

    border:
        1px solid
        #111;

    border-radius:
        50%;

    box-shadow:
        inset
        0 1px 1px
        rgba(
            255,
            255,
            255,
            0.08
        );

    display:
        block;

    height:
        8px;

    width:
        8px;
}


.printer-led.led-active {
    animation:
        printer3dAmberPulse
        0.46s
        ease-in-out
        infinite
        alternate;

    background:
        var(--amber);

    box-shadow:
        0 0 10px
        rgba(
            255,
            214,
            102,
            0.58
        );
}


.printer-led.led-active.delay-1 {
    animation-delay:
        0.16s;
}


.printer-led.led-ready {
    animation:
        printer3dGreenReady
        0.3s
        ease
        1.48s
        forwards;

    background:
        #333;
}


.printer-control-deck {
    align-items:
        stretch;

    display:
        grid;

    gap:
        16px;

    grid-template-columns:
        minmax(
            0,
            1fr
        )
        auto;

    margin-top:
        20px;
}


.printer-control-screen {
    background:
        linear-gradient(
            180deg,
            #060806,
            #020302
        );

    border:
        1px solid
        #222a22;

    border-radius:
        7px;

    box-shadow:
        inset
        0 0 22px
        rgba(
            0,
            0,
            0,
            0.88
        ),
        0 1px
        rgba(
            255,
            255,
            255,
            0.03
        );

    min-height:
        72px;

    padding:
        12px
        14px;

    position:
        relative;
}


.printer-control-label {
    color:
        #3f5140;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.47rem;

    letter-spacing:
        0.15em;

    text-transform:
        uppercase;
}


.printer-control-value {
    color:
        #6b776b;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.85rem;

    font-weight:
        900;

    letter-spacing:
        0.08em;

    margin-top:
        7px;

    text-transform:
        uppercase;
}


.printer-control-value.idle {
    color:
        #606760;
}


.printer-control-sub {
    bottom:
        8px;

    color:
        #313b31;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.42rem;

    letter-spacing:
        0.08em;

    position:
        absolute;

    right:
        11px;

    text-transform:
        uppercase;
}


.printer-control-buttons {
    align-items:
        center;

    display:
        flex;

    gap:
        9px;

    justify-content:
        flex-end;
}


.printer-control-button,
.printer-power-button {
    align-items:
        center;

    background:
        linear-gradient(
            145deg,
            #292929,
            #111
        );

    border:
        1px solid
        #363636;

    border-radius:
        50%;

    box-shadow:
        0 4px 9px
        rgba(
            0,
            0,
            0,
            0.6
        ),
        inset
        0 1px
        rgba(
            255,
            255,
            255,
            0.06
        );

    display:
        flex;

    height:
        36px;

    justify-content:
        center;

    width:
        36px;
}


.printer-control-button span {
    background:
        #666;

    border-radius:
        999px;

    display:
        block;

    height:
        3px;

    width:
        13px;
}


.printer-control-button.active span {
    background:
        var(--amber);

    box-shadow:
        0 0 7px
        rgba(
            255,
            214,
            102,
            0.45
        );
}


.printer-power-button span {
    border:
        2px solid
        #555;

    border-top-color:
        transparent;

    border-radius:
        50%;

    display:
        block;

    height:
        12px;

    position:
        relative;

    width:
        12px;
}


.printer-power-button span::before {
    background:
        #555;

    content:
        "";

    height:
        7px;

    left:
        50%;

    position:
        absolute;

    top:
        -5px;

    transform:
        translateX(
            -50%
        );

    width:
        2px;
}


.printer-power-button.on span {
    border-color:
        var(--green);

    border-top-color:
        transparent;

    box-shadow:
        0 0 8px
        rgba(
            141,
            255,
            99,
            0.32
        );
}


.printer-power-button.on span::before {
    background:
        var(--green);
}


.printer-mouth-wrap {
    margin-top:
        27px;

    position:
        relative;
}


.printer-mouth-bezel {
    background:
        linear-gradient(
            180deg,
            #080808,
            #191919
        );

    border:
        1px solid
        #2d2d2d;

    border-radius:
        7px;

    box-shadow:
        inset
        0 4px 13px
        rgba(
            0,
            0,
            0,
            0.95
        );

    margin:
        0 auto;

    max-width:
        470px;

    padding:
        7px;
}


.printer-mouth {
    background:
        #010101;

    border:
        1px solid
        #111;

    border-radius:
        3px;

    box-shadow:
        inset
        0 5px 13px
        rgba(
            0,
            0,
            0,
            0.95
        );

    height:
        16px;

    overflow:
        hidden;

    position:
        relative;
}


.printer-mouth.active {
    animation:
        printerMouthWake
        0.42s
        ease
        0.18s
        both;
}


.printer-mouth-line {
    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(
                255,
                255,
                255,
                0.12
            ),
            transparent
        );

    height:
        1px;

    left:
        5%;

    position:
        absolute;

    right:
        5%;

    top:
        7px;
}


.printer-mouth-label {
    color:
        #3b3b3b;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.44rem;

    letter-spacing:
        0.14em;

    margin-top:
        6px;

    text-align:
        center;

    text-transform:
        uppercase;
}


.printer-lower-panel {
    align-items:
        flex-end;

    display:
        flex;

    justify-content:
        space-between;

    margin-top:
        22px;
}


.printer-vent-grid {
    display:
        flex;

    gap:
        5px;
}


.printer-vent-grid span {
    background:
        #252525;

    border-radius:
        99px;

    display:
        block;

    height:
        3px;

    width:
        24px;
}


.printer-serial {
    color:
        #363636;

    font-family:
        "Courier New",
        monospace;

    font-size:
        0.46rem;

    letter-spacing:
        0.08em;
}


.printer-output-stage {
    margin-top:
        -31px;

    max-width:
        560px;

    overflow:
        hidden;

    padding-top:
        18px;

    position:
        relative;

    z-index:
        2;
}


.printer-output-stage
.print-feed {
    filter:
        drop-shadow(
            0 18px 18px
            rgba(
                0,
                0,
                0,
                0.34
            )
        );
}


.printer-output-stage
.receipt {
    border-radius:
        0
        0
        4px
        4px;

    transform-origin:
        top center;
}


.printer-hero-active
.printer-3d-unit {
    animation:
        printer3dWake
        0.48s
        cubic-bezier(
            0.2,
            0.8,
            0.2,
            1
        )
        both;
}


.printer-hero-active
.printer-3d-top {
    animation:
        printerTopWake
        0.48s
        ease
        both;
}


.printer-hero-active
.printer-floor-shadow {
    animation:
        printerShadowWake
        0.5s
        ease
        both;
}


@keyframes printer3dWake {

    0% {
        filter:
            brightness(
                0.78
            );

        transform:
            rotateX(
                4deg
            )
            rotateY(
                -5.5deg
            )
            translateY(
                4px
            );
    }

    55% {
        filter:
            brightness(
                1.07
            );

        transform:
            rotateX(
                3deg
            )
            rotateY(
                -4deg
            )
            translateY(
                -2px
            );
    }

    100% {
        filter:
            brightness(
                1
            );

        transform:
            rotateX(
                4deg
            )
            rotateY(
                -5.5deg
            )
            translateY(
                0
            );
    }
}


@keyframes printerTopWake {

    from {
        filter:
            brightness(
                0.72
            );
    }

    to {
        filter:
            brightness(
                1
            );
    }
}


@keyframes printerShadowWake {

    from {
        opacity:
            0.62;

        transform:
            rotate(
                -1deg
            )
            scaleX(
                0.94
            );
    }

    to {
        opacity:
            1;

        transform:
            rotate(
                -1deg
            )
            scaleX(
                1
            );
    }
}


@keyframes printerMouthWake {

    0% {
        box-shadow:
            inset
            0 5px 13px
            rgba(
                0,
                0,
                0,
                0.95
            );
    }

    45% {
        box-shadow:
            inset
            0 5px 13px
            rgba(
                0,
                0,
                0,
                0.95
            ),
            0 0 12px
            rgba(
                255,
                214,
                102,
                0.08
            );
    }

    100% {
        box-shadow:
            inset
            0 5px 13px
            rgba(
                0,
                0,
                0,
                0.95
            );
    }
}


@keyframes printer3dAmberPulse {

    from {
        opacity:
            0.5;
    }

    to {
        opacity:
            1;
    }
}


@keyframes printer3dGreenReady {

    from {
        background:
            #333;

        box-shadow:
            none;
    }

    to {
        background:
            var(--green);

        box-shadow:
            0 0 10px
            rgba(
                141,
                255,
                99,
                0.62
            );
    }
}


@media (
    hover:
    hover
)
and (
    pointer:
    fine
) {

    .printer-hero-idle
    .printer-3d-unit {
        transition:
            transform
            0.35s
            ease;
    }


    .printer-hero-idle:hover
    .printer-3d-unit {
        transform:
            rotateX(
                3deg
            )
            rotateY(
                -3deg
            )
            translateY(
                -2px
            );
    }
}


@media (
    max-width:
    760px
) {

    .printer-3d-scene {
        padding:
            30px
            24px
            18px;
    }


    .printer-3d-unit {
        transform:
            rotateX(
                3deg
            )
            rotateY(
                -3.5deg
            );
    }


    .printer-3d-front {
        min-height:
            285px;

        padding:
            19px
            18px
            24px;
    }


    .printer-3d-top {
        left:
            9%;

        right:
            9%;
    }


    .printer-3d-side {
        right:
            -28px;

        width:
            55px;
    }


    .printer-control-deck {
        gap:
            10px;
    }


    .printer-control-buttons {
        gap:
            6px;
    }


    .printer-control-button,
    .printer-power-button {
        height:
            31px;

        width:
            31px;
    }


    .printer-vent-grid span {
        width:
            15px;
    }


    .printer-output-stage {
        margin-top:
            -24px;

        max-width:
            calc(
                100%
                -
                34px
            );
    }
}


@media (
    max-width:
    480px
) {

    .printer-3d-scene {
        padding-left:
            12px;

        padding-right:
            18px;
    }


    .printer-engraving {
        font-size:
            0.55rem;
    }


    .printer-control-deck {
        grid-template-columns:
            1fr;
    }


    .printer-control-buttons {
        display:
            none;
    }


    .printer-3d-side {
        right:
            -19px;

        width:
            42px;
    }


    .printer-output-stage {
        max-width:
            calc(
                100%
                -
                20px
            );
    }
}



/* ========================================================
   V2 INTERACTIVE MATCH PICKER
======================================================== */

.interaction-kicker {
    color: var(--green);
    font-family: "Courier New", monospace;
    font-size: 0.64rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    margin-top: 1.4rem;
    text-transform: uppercase;
}


.interaction-title {
    color: #f0f0f0;
    font-size: clamp(1.35rem, 3vw, 2rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-top: 0.2rem;
}


.interaction-copy {
    color: #777;
    font-size: 0.8rem;
    line-height: 1.55;
    margin: 0.55rem 0 1rem;
    max-width: 700px;
}


.fixture-choice-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.035),
            rgba(255,255,255,0.012)
        );

    border: 1px solid #292929;
    border-radius: 10px;
    min-height: 176px;
    padding: 0.85rem;
    transition:
        border-color 0.18s ease,
        background 0.18s ease,
        transform 0.18s ease;
}


.fixture-choice-card.selected {
    background:
        rgba(
            141,
            255,
            99,
            0.045
        );

    border-color:
        rgba(
            141,
            255,
            99,
            0.55
        );
}


.fixture-choice-topline {
    align-items: center;
    color: #6f6f6f;
    display: flex;
    font-family: "Courier New", monospace;
    font-size: 0.55rem;
    gap: 0.5rem;
    justify-content: space-between;
    letter-spacing: 0.08em;
    min-height: 24px;
    text-transform: uppercase;
}


.fixture-choice-card.selected
.fixture-choice-topline {
    color: var(--green);
}


.fixture-choice-teams {
    align-items: center;
    display: grid;
    gap: 0.4rem;
    grid-template-columns:
        minmax(0, 1fr)
        auto
        minmax(0, 1fr);
    margin-top: 0.65rem;
}


.fixture-choice-team {
    min-width: 0;
    text-align: center;
}


.fixture-choice-card
.club-crest-wrap {
    height: 58px;
    margin: 0 auto 0.35rem;
    width: 66px;
}


.fixture-choice-card
.club-crest {
    height: 52px;
    max-width: 60px;
}


.fixture-choice-name {
    color: #e7e7e7;
    font-size: 0.78rem;
    font-weight: 800;
    line-height: 1.15;
    overflow-wrap: anywhere;
}


.fixture-choice-vs {
    color: #4f4f4f;
    font-family: "Courier New", monospace;
    font-size: 0.57rem;
    font-weight: 900;
}


/* Streamlit actions become machine controls rather than
   generic dashboard buttons. */
div.stButton > button {
    background: #111 !important;
    border: 1px solid #343434 !important;
    border-radius: 4px !important;
    color: #d7d7d7 !important;
    font-family: "Courier New", monospace !important;
    font-size: 0.67rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.075em !important;
    min-height: 46px !important;
    text-transform: uppercase !important;
    transition:
        background 0.16s ease,
        border-color 0.16s ease,
        color 0.16s ease !important;
    width: 100% !important;
}


div.stButton > button:hover {
    background: #181818 !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
}


div.stButton > button:focus-visible {
    outline: 2px solid var(--green) !important;
    outline-offset: 2px !important;
}


/* ========================================================
   MACHINE IDLE / ACTIVE STATES
======================================================== */

.machine-idle {
    margin-top: 1.4rem;
}


.idle-led {
    color: #666;
}


.printer-dot-idle {
    background: #595959;
    border-radius: 50%;
    display: inline-block;
    height: 6px;
    width: 6px;
}


.machine-idle-screen {
    border: 1px dashed #282828;
    border-radius: 8px;
    margin: 0 auto;
    max-width: 590px;
    padding: 3.1rem 1.2rem;
    text-align: center;
}


.machine-idle-kicker {
    color: #5f5f5f;
    font-family: "Courier New", monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}


.machine-idle-title {
    color: #d8d8d8;
    font-size: clamp(1.1rem, 3vw, 1.65rem);
    font-weight: 900;
    letter-spacing: -0.035em;
    margin-top: 0.55rem;
}


.machine-idle-note {
    color: #686868;
    font-size: 0.72rem;
    line-height: 1.55;
    margin: 0.75rem auto 0;
    max-width: 430px;
}


.printer-state-waking,
.printer-state-printing,
.printer-state-ready {
    align-items: center;
    display: flex;
    gap: 6px;
    justify-content: flex-end;
    position: absolute;
    right: 0;
    top: 0;
    white-space: nowrap;
}


.printer-state-waking {
    animation:
        wakingStateOut
        0.18s
        ease
        0.30s
        forwards;

    color: var(--amber);
    opacity: 1;
}


.printer-state-printing {
    animation:
        printingStateIn
        0.20s
        ease
        0.26s
        forwards,

        printingStateOut
        0.22s
        ease
        1.42s
        forwards;

    color: var(--amber);
    opacity: 0;
}


.printer-state-ready {
    animation:
        readyStateIn
        0.30s
        ease
        1.52s
        forwards;

    color: var(--green);
    opacity: 0;
}


.machine-active {
    animation:
        machineWake
        0.38s
        ease-out
        both;
}


.model-pick-probability {
    color: #676767;
    font-family: "Courier New", monospace;
    font-size: 0.56rem;
    letter-spacing: 0.05em;
    margin-top: 0.22rem;
    text-transform: uppercase;
}


/* ========================================================
   POST-PRINT ACTIONS
======================================================== */

.prediction-complete {
    animation:
        postPrintReveal
        0.35s
        ease
        1.72s
        both;

    background:
        rgba(
            141,
            255,
            99,
            0.035
        );

    border:
        1px solid
        rgba(
            141,
            255,
            99,
            0.20
        );

    border-radius: 8px;
    margin: 0.85rem auto 0;
    max-width: 760px;
    padding: 0.85rem 1rem;
}


.prediction-complete-top {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    justify-content: space-between;
}


.prediction-complete-state {
    color: var(--green);
    font-family: "Courier New", monospace;
    font-size: 0.63rem;
    font-weight: 900;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}


.prediction-complete-copy {
    color: #757575;
    font-size: 0.7rem;
    line-height: 1.45;
}


/* ========================================================
   PLAIN-LANGUAGE EVIDENCE
======================================================== */

.explain-grid {
    display: grid;
    gap: 0.7rem;
    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );
    margin-top: 0.8rem;
}


.explain-card {
    background:
        rgba(
            255,
            255,
            255,
            0.025
        );
    border: 1px solid #252525;
    border-radius: 8px;
    min-width: 0;
    padding: 0.9rem;
}


.explain-label {
    color: #676767;
    font-family: "Courier New", monospace;
    font-size: 0.57rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}


.explain-value {
    color: #f0f0f0;
    font-size: 1.25rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin-top: 0.28rem;
}


.explain-copy {
    color: #777;
    font-size: 0.7rem;
    line-height: 1.5;
    margin-top: 0.32rem;
}


.share-panel {
    background: #0e0e0e;
    border: 1px solid #282828;
    border-radius: 8px;
    margin: 0.8rem 0;
    padding: 0.8rem 0.9rem;
}


.share-panel-title {
    color: #ddd;
    font-family: "Courier New", monospace;
    font-size: 0.66rem;
    font-weight: 900;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}


.share-panel-copy {
    color: #777;
    font-size: 0.69rem;
    line-height: 1.45;
    margin-top: 0.25rem;
}


/* ========================================================
   INTERACTION ANIMATIONS
======================================================== */

@keyframes wakingStateOut {
    to {
        opacity: 0;
    }
}


@keyframes printingStateIn {
    to {
        opacity: 1;
    }
}


@keyframes machineWake {
    0% {
        border-color: #282828;
        box-shadow:
            0 28px 80px rgba(0,0,0,0.55),
            inset 0 1px rgba(255,255,255,0.04);
    }

    55% {
        border-color:
            rgba(
                255,
                214,
                102,
                0.36
            );

        box-shadow:
            0 28px 80px rgba(0,0,0,0.55),
            0 0 28px rgba(255,214,102,0.05),
            inset 0 1px rgba(255,255,255,0.04);
    }

    100% {
        border-color: #282828;
    }
}


@keyframes postPrintReveal {
    from {
        opacity: 0;
        transform: translateY(-7px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@media (
    max-width:
    760px
) {

    .explain-grid {
        grid-template-columns: 1fr;
    }

    .fixture-choice-card {
        min-height: 164px;
    }
}



/* ========================================================
   V2 STARTUP LOADING SCREEN
======================================================== */

.ucl-startup-overlay {
    align-items: center;
    animation:
        uclStartupOverlayExit
        0.48s
        cubic-bezier(
            0.4,
            0,
            0.2,
            1
        )
        1.85s
        forwards;
    will-change:
        opacity,
        transform;
    background:
        radial-gradient(
            circle at 50% 44%,
            #111 0%,
            #090909 42%,
            #050505 100%
        );
    display: flex;
    inset: 0;
    justify-content: center;
    opacity: 1;
    overflow: hidden;
    pointer-events: auto;
    position: fixed;
    visibility: visible;
    z-index: 2147483000;
}


.ucl-startup-inner {
    align-items: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2rem;
    position: relative;
    width: min(680px, 90vw);
    z-index: 2;
}


.ucl-startup-wordmark {
    animation:
        uclStartupLogoIn
        0.55s
        cubic-bezier(
            0.2,
            0.8,
            0.2,
            1
        )
        both;

    display: block;
    height: auto;
    max-width: 520px;
    object-fit: contain;
    width: min(74vw, 520px);
}


.ucl-startup-fallback {
    color: #f0eee7;
    font-family:
        "Arial Black",
        Arial,
        sans-serif;
    font-size: clamp(2.1rem, 8vw, 4.4rem);
    font-weight: 900;
    letter-spacing: -0.06em;
    line-height: 0.9;
    text-align: center;
}


.ucl-startup-fallback span {
    color: var(--green);
    display: block;
    font-family:
        "Courier New",
        monospace;
    font-size: 0.38em;
    letter-spacing: 0.25em;
    margin-top: 0.45rem;
}


.ucl-startup-copy {
    animation:
        uclStartupCopyIn
        0.4s
        ease
        0.30s
        both;

    color: #888;
    font-family:
        "Courier New",
        monospace;
    font-size: clamp(0.58rem, 1.8vw, 0.72rem);
    font-weight: 700;
    letter-spacing: 0.18em;
    margin-top: 1.25rem;
    text-align: center;
}


.ucl-startup-progress {
    background: #1c1c1c;
    height: 2px;
    margin-top: 1rem;
    max-width: 350px;
    overflow: hidden;
    position: relative;
    width: 68%;
}


.ucl-startup-progress-fill {
    animation:
        uclStartupProgress
        1.48s
        cubic-bezier(
            0.65,
            0,
            0.35,
            1
        )
        0.15s
        both;

    background: var(--green);
    height: 100%;
    transform-origin: left center;
    width: 100%;
}


.ucl-startup-system {
    animation:
        uclStartupCopyIn
        0.4s
        ease
        0.55s
        both;

    color: #4f4f4f;
    font-family:
        "Courier New",
        monospace;
    font-size: 0.52rem;
    letter-spacing: 0.09em;
    margin-top: 0.65rem;
    text-align: center;
}


.ucl-startup-scan {
    animation:
        uclStartupScan
        1.65s
        linear
        both;

    background:
        linear-gradient(
            180deg,
            transparent,
            rgba(
                180,
                255,
                80,
                0.05
            ),
            transparent
        );
    height: 20%;
    left: 0;
    pointer-events: none;
    position: absolute;
    top: -22%;
    width: 100%;
    z-index: 1;
}


@keyframes uclStartupLogoIn {

    from {
        opacity: 0;
        transform:
            translateY(8px)
            scale(0.985);
    }

    to {
        opacity: 1;
        transform:
            translateY(0)
            scale(1);
    }
}


@keyframes uclStartupCopyIn {

    from {
        opacity: 0;
        transform:
            translateY(5px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}


@keyframes uclStartupProgress {

    from {
        transform:
            scaleX(0);
    }

    to {
        transform:
            scaleX(1);
    }
}


@keyframes uclStartupScan {

    from {
        transform:
            translateY(0);
    }

    to {
        transform:
            translateY(610%);
    }
}


@keyframes uclStartupOverlayExit {

    0% {
        opacity: 1;
        pointer-events: auto;
        transform:
            translateY(0);
        visibility: visible;
    }

    70% {
        opacity: 0;
        pointer-events: none;
        transform:
            translateY(-8%);
        visibility: visible;
    }

    99% {
        opacity: 0;
        pointer-events: none;
        transform:
            translateY(-110%);
        visibility: visible;
    }

    100% {
        opacity: 0;
        pointer-events: none;
        transform:
            translateY(-110%);
        visibility: hidden;
    }
}


@media (
    max-width:
    520px
) {

    .ucl-startup-inner {
        padding:
            1.2rem;
    }


    .ucl-startup-wordmark {
        width:
            min(
                82vw,
                450px
            );
    }


    .ucl-startup-progress {
        width:
            76%;
    }
}


/* ========================================================
   REDUCED MOTION
======================================================== */

@media (
    prefers-reduced-motion:
    reduce
) {


    .ucl-startup-wordmark,
    .ucl-startup-copy,
    .ucl-startup-progress-fill,
    .ucl-startup-system,
    .ucl-startup-scan {
        animation:
            none !important;
        opacity:
            1 !important;
        transform:
            none !important;
    }


    .ucl-startup-progress-fill {
        transform:
            scaleX(1) !important;
    }


    .ucl-startup-overlay {
        animation:
            uclStartupOverlayExit
            0.01s
            linear
            0.55s
            forwards !important;
        will-change:
            opacity,
            transform;
    }

    .print-feed,
    .print-paper,
    .print-paper::after,
    .print-stage::before,
    .printer-state-waking,
    .printer-state-printing,
    .printer-state-ready,
    .printer-dot-printing,
    .machine-active,
    .prediction-complete,
    .printer-3d-unit,
    .printer-3d-top,
    .printer-floor-shadow,
    .printer-mouth.active,
    .printer-led.led-active,
    .printer-led.led-ready {
        animation:
            none !important;
    }


    .print-feed {
        max-height:
            none;
    }


    .printer-state-waking,
    .printer-state-printing {
        display:
            none;
    }


    .printer-state-ready {
        opacity:
            1;
    }


    .printer-led.led-ready {
        background:
            var(--green);
        box-shadow:
            0 0 10px
            rgba(
                141,
                255,
                99,
                0.62
            );
    }


    .prediction-complete {
        opacity:
            1;
        transform:
            none;
    }
}
"""




# ============================================================
# UCL PREDICTOR V2 — LIGHT THEME
# ============================================================
#
# The printer and thermal receipt intentionally retain their
# physical dark-machine / paper identity. The surrounding app,
# controls, navigation, cards and evidence surfaces switch to
# a light product theme.
# ============================================================


LIGHT_MODE_CSS = r"""
:root {
    --light-page:
        #f1efe8;

    --light-page-2:
        #faf9f5;

    --light-panel:
        #f7f5ef;

    --light-card:
        #fbfaf6;

    --light-border:
        #ccc8bd;

    --light-border-strong:
        #aaa69b;

    --light-text:
        #181818;

    --light-muted:
        #66635d;

    --light-accent-text:
        #2f7c12;
}


/* ========================================================
   PAGE / STREAMLIT CHROME
======================================================== */

html,
body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 50% -20%,
            #ffffff 0%,
            var(--light-page-2) 36%,
            var(--light-page) 78%
        )
        !important;

    color:
        var(--light-text)
        !important;

    color-scheme:
        light;
}


[data-testid="stHeader"] {
    background:
        transparent
        !important;
}


[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    color:
        var(--light-text);
}


/* ========================================================
   BRAND / HEADER
======================================================== */

.main-app-wordmark-ucl {
    color:
        #151515;
}


.main-app-wordmark-predictor {
    color:
        var(--light-accent-text);
}


.app-kicker,
.main-app-kicker {
    color:
        #67645e;
}


.app-subtitle,
.main-app-subtitle {
    color:
        #5f5c56;
}


.top-rule {
    border-color:
        #d2cec3;
}


/* ========================================================
   STATUS
======================================================== */

.status-pill {
    background:
        rgba(
            255,
            255,
            255,
            0.72
        );

    border-color:
        var(--light-border);

    color:
        #55524d;
}


.status-pill.green {
    background:
        rgba(
            75,
            158,
            40,
            0.08
        );

    border-color:
        rgba(
            56,
            128,
            26,
            0.32
        );

    color:
        #347c18;
}


.status-pill.amber {
    background:
        rgba(
            184,
            125,
            0,
            0.08
        );

    border-color:
        rgba(
            165,
            112,
            0,
            0.32
        );

    color:
        #916500;
}


.status-pill.red {
    background:
        rgba(
            179,
            52,
            45,
            0.07
        );

    border-color:
        rgba(
            179,
            52,
            45,
            0.28
        );

    color:
        #a33731;
}


.creator-status-pill {
    border-color:
        #a7a297;
}


/* ========================================================
   NAVIGATION + USER CONTROLS
======================================================== */

div.stButton > button {
    background:
        rgba(
            255,
            255,
            255,
            0.72
        )
        !important;

    border-color:
        #c8c4b9
        !important;

    color:
        #242424
        !important;

    box-shadow:
        0 1px 0
        rgba(
            255,
            255,
            255,
            0.78
        )
        inset;
}


div.stButton > button:hover {
    background:
        #ffffff
        !important;

    border-color:
        #78a866
        !important;

    color:
        #2f7615
        !important;
}


div.stButton > button:focus-visible {
    outline-color:
        #48872f
        !important;
}


button[kind="primary"] {
    background:
        rgba(
            72,
            143,
            43,
            0.10
        )
        !important;

    border-color:
        rgba(
            60,
            128,
            33,
            0.48
        )
        !important;

    color:
        #2f7615
        !important;
}


button:disabled {
    color:
        #8c8982
        !important;
}


div.stDownloadButton > button {
    background:
        #222
        !important;

    color:
        #f6f3e9
        !important;
}


div.stDownloadButton > button:hover {
    background:
        #080808
        !important;

    color:
        #ffffff
        !important;
}


/* ========================================================
   SCREEN HEADINGS
======================================================== */

.interaction-kicker {
    color:
        #337d18;
}


.interaction-title {
    color:
        var(--light-text);
}


.interaction-copy {
    color:
        var(--light-muted);
}


/* ========================================================
   FIXTURE CARDS
======================================================== */

.fixture-choice-card {
    background:
        linear-gradient(
            145deg,
            rgba(
                255,
                255,
                255,
                0.92
            ),
            rgba(
                246,
                244,
                237,
                0.92
            )
        );

    border-color:
        var(--light-border);

    box-shadow:
        0 8px 24px
        rgba(
            53,
            49,
            40,
            0.045
        );
}


.fixture-choice-card.selected {
    background:
        rgba(
            82,
            157,
            48,
            0.07
        );

    border-color:
        rgba(
            61,
            130,
            31,
            0.46
        );
}


.fixture-choice-name {
    color:
        #1a1a1a;
}


.fixture-choice-topline,
.fixture-choice-vs {
    color:
        #747069;
}


.fixture-choice-card.selected
.fixture-choice-topline {
    color:
        #337d18;
}


/* ========================================================
   GENERAL CONTENT CARDS
======================================================== */

.dark-card,
.explain-card {
    background:
        rgba(
            255,
            255,
            255,
            0.68
        );

    border-color:
        var(--light-border);
}


.dark-card {
    color:
        #44413c;
}


.dark-card-label,
.explain-label {
    color:
        #6e6a63;
}


.dark-card-value,
.explain-value {
    color:
        #171717;
}


.dark-card-note,
.explain-copy {
    color:
        #66625b;
}


.section-title {
    color:
        #242424;
}


/* ========================================================
   PRINTED / SHARE ACTIONS
======================================================== */

.prediction-complete {
    background:
        rgba(
            70,
            145,
            39,
            0.06
        );

    border-color:
        rgba(
            60,
            128,
            33,
            0.25
        );
}


.prediction-complete-state {
    color:
        #337d18;
}


.prediction-complete-copy {
    color:
        #66625c;
}


.share-panel {
    background:
        rgba(
            255,
            255,
            255,
            0.70
        );

    border-color:
        var(--light-border);
}


.share-panel-title {
    color:
        #242424;
}


.share-panel-copy {
    color:
        #66625c;
}


/* ========================================================
   HEALTH / ALERT SURFACES
======================================================== */

.health-panel {
    background:
        rgba(
            255,
            255,
            255,
            0.60
        );

    border-color:
        var(--light-border);
}


.health-label {
    color:
        #737068;
}


.health-value {
    color:
        #242424;
}


.production-alert.amber {
    background:
        rgba(
            184,
            125,
            0,
            0.06
        );

    color:
        #805c0f;
}


.production-alert.red {
    background:
        rgba(
            179,
            52,
            45,
            0.055
        );

    color:
        #9a413b;
}


/* ========================================================
   STREAMLIT EXPANDERS / SECONDARY PANELS
======================================================== */

[data-testid="stExpander"] {
    background:
        rgba(
            255,
            255,
            255,
            0.62
        );

    border-color:
        var(--light-border)
        !important;
}


[data-testid="stExpander"] summary {
    color:
        #242424
        !important;
}


[data-testid="stCode"] {
    border:
        1px solid
        var(--light-border);
}


/* ========================================================
   FOOTER
======================================================== */

.app-footer {
    border-color:
        #d1cdc2;

    color:
        #77736b;
}


/* ========================================================
   IDLE MESSAGE BELOW THE PRINTER
======================================================== */

.machine-idle-screen {
    background:
        rgba(
            255,
            255,
            255,
            0.48
        );

    border-color:
        #c8c4b9;
}


.machine-idle-kicker {
    color:
        #757169;
}


.machine-idle-title {
    color:
        #1c1c1c;
}


.machine-idle-note {
    color:
        #67635c;
}


/* ========================================================
   KEEP PHYSICAL OBJECTS PHYSICAL
======================================================== */

/*
The printer remains dark in both themes by design. It should
read as a physical black forecasting machine sitting on either
a dark or light environment.

The thermal receipt also keeps its paper palette in both modes.
*/

.printer-3d-front,
.printer-3d-top,
.printer-3d-side,
.printer-control-screen,
.printer-mouth,
.printer-mouth-bezel {
    color-scheme:
        dark;
}


.receipt {
    color-scheme:
        light;
}



/* ========================================================
   EVIDENCE VAULT — LIGHT MODE
======================================================== */


.vault-provenance,
.vault-provenance-summary {
    color:
        #77736b;
}


.vault-provenance-detail {
    color:
        #337d18;
}


.vault-proof-source {
    border-color:
        rgba(
            58,
            127,
            28,
            0.14
        );

    color:
        #5f6d58;
}



.vault-summary-card,
.vault-record-card,
.vault-detail,
.vault-compare-card {
    background:
        rgba(
            255,
            255,
            255,
            0.72
        );

    border-color:
        var(--light-border);
}


.vault-summary-card strong,
.vault-record-team span,
.vault-record-score,
.vault-record-grid strong,
.vault-detail-title,
.vault-final-score > strong,
.vault-final-score > div > span,
.vault-compare-card strong,
.vault-proof-grid strong {
    color:
        #1b1b1b;
}


.vault-summary-card span,
.vault-record-label,
.vault-compare-card span,
.vault-proof-grid span,
.vault-detail-kicker,
.vault-record-topline,
.vault-filter-heading {
    color:
        #6f6b63;
}


.vault-record-grid {
    border-color:
        #d6d1c5;
}


.vault-record-grid small,
.vault-summary-card small,
.vault-compare-card small {
    color:
        #7a766e;
}


.vault-record-card.correct {
    border-color:
        rgba(
            58,
            127,
            28,
            0.28
        );
}


.vault-record-card.missed {
    border-color:
        rgba(
            164,
            58,
            50,
            0.26
        );
}


.vault-outcome-badge.correct {
    color:
        #337d18;
}


.vault-outcome-badge.missed {
    color:
        #a43d36;
}


.vault-proof-line {
    color:
        #477336;
}


.vault-proof-panel {
    background:
        rgba(
            71,
            143,
            41,
            0.055
        );

    border-color:
        rgba(
            58,
            127,
            28,
            0.24
        );
}


.vault-proof-title {
    color:
        #337d18;
}


.vault-sample-warning {
    background:
        rgba(
            166,
            114,
            0,
            0.055
        );

    border-color:
        rgba(
            150,
            102,
            0,
            0.23
        );

    color:
        #6f6247;
}


.vault-sample-warning strong {
    color:
        #8a6208;
}


.vault-empty {
    border-color:
        #bbb7ad;
}


.vault-empty-title {
    color:
        #262626;
}


.vault-empty-copy {
    color:
        #6b675f;
}



/* ========================================================
   CREATOR / DATA TABLE CONTAINERS
======================================================== */

[data-testid="stDataFrame"] {
    border:
        1px solid
        var(--light-border);

    border-radius:
        6px;

    overflow:
        hidden;
}
"""


def apply_v2_styles(
    render_html,
    *,
    theme: str = "dark",
) -> None:
    """
    Inject the complete UCL Predictor V2 visual system.

    theme:
        "dark"  -> original industrial dark interface
        "light" -> light surrounding UI while preserving the
                   dark physical printer and thermal receipt
    """

    normalized_theme = (
        str(
            theme
        )
        .strip()
        .lower()
    )

    if normalized_theme not in {
        "dark",
        "light",
    }:

        normalized_theme = "dark"

    active_css = V2_CSS

    if normalized_theme == "light":

        active_css += (
            "\n"
            +
            LIGHT_MODE_CSS
        )

    render_html(
        "<style>\n"
        + active_css
        + "\n</style>"
    )
