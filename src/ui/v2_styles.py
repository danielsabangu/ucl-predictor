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
"""


def apply_v2_styles(
    render_html,
) -> None:
    """
    Inject the complete UCL Predictor V2 visual system.
    """

    render_html(
        "<style>\n"
        + V2_CSS
        + "\n</style>"
    )
