"""Broadcast Stage 3 visual system. Presentation only; never modifies receipt data."""
from __future__ import annotations

BROADCAST_CSS = r"""
/* The --bp-stadium-image variable is populated by app.py; do not redeclare it here. */
:root{--bp-bg:#061324;--bp-surface:#0b1c30;--bp-surface-2:#102640;--bp-line:#294360;
--bp-text:#edf5ff;--bp-subtle:#9eb5cf;--bp-sky:#5fa7ff;--bp-accent:#3088f8}
/* An editorial football product: sharp grid, compact headers, consistent rhythm. */
[data-testid="stAppViewContainer"],[data-testid="stMain"],html,body{
background:linear-gradient(180deg,#061222 0%,#07172a 48%,#091827 100%)!important;
color:var(--bp-text)!important}
.block-container{max-width:1510px!important;padding:0.4rem 2rem 4rem!important}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden!important}
.bp-sr-only{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;
margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}
.bp-brand-header{display:flex;align-items:center;justify-content:space-between;gap:14px;
margin-bottom:6px;border-bottom:1px solid #24405e;padding:8px 0 13px}
.bp-brand-lockup{display:flex;align-items:center;gap:10px;min-width:0}
.bp-brand-name{font-family:"Sora",sans-serif;font-size:clamp(17px,2vw,22px);font-weight:800;letter-spacing:-.065em;white-space:nowrap}
.bp-brand-name strong{font-weight:800}.bp-brand-name strong span{color:#60b5ff}
.bp-brand-version{border:1px solid #2a68a6;color:#86c8ff;border-radius:5px;padding:4px 7px;font-size:10px;font-weight:800}
.bp-brand-caption{color:#93b1d2;font-size:10px;letter-spacing:.1em;font-weight:750}
.bp-brand-ball{flex-shrink:0}
.nf-app-header{display:none!important}
/* Remove the rows of debugging badges visible in earlier iterations for public. */
.consumer-status-row{margin:4px 0!important}
.st-key-matchday_top_nav{margin:2px 0 10px!important;background:transparent!important;border:0!important;
padding:0!important;border-bottom:1px solid #1e3a58!important;border-radius:0!important;box-shadow:none!important}
.st-key-matchday_top_nav [data-testid="stHorizontalBlock"]{gap:2px!important;flex-wrap:nowrap!important}
.st-key-matchday_top_nav [data-testid="stColumn"]{min-width:0!important}
.st-key-matchday_top_nav div.stButton>button{height:43px!important;min-height:43px!important;
font-size:12px!important;font-weight:750!important;color:#b5cbe4!important;border-radius:7px 7px 0 0!important}
.st-key-matchday_top_nav div.stButton>button[kind="primary"]{background:#142e52!important;color:#fff!important;
border:1px solid #214d82!important;border-bottom:3px solid #68b6ff!important;box-shadow:none!important}
.st-key-matchday_top_nav div.stButton>button:hover{background:#153454!important;transform:none!important}
.st-key-matchday_top_nav div.stButton>button:focus-visible{outline:2px solid #6eb7ff!important;outline-offset:-2px!important}
/* Brand-select styling including focus; no mismatched default white dropdown. */
[data-baseweb="select"]>div,[data-testid="stSelectbox"] [data-baseweb="select"]>div{
background:#112743!important;border-color:#365675!important;border-radius:7px!important;color:#ecf6ff!important;
min-height:43px!important;box-shadow:none!important}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="popover"]{background:#112743!important;color:#e9f4ff!important}
[data-baseweb="select"] svg{fill:#a8c8ed!important}
.stSelectbox label{color:#aec6e1!important;font-size:11px!important;font-weight:750!important}
/* Low-emphasis utility buttons. */
.st-key-bp_hero_actions .stButton>button,.st-key-bp_hero_actions div.stButton>button{
height:53px!important;min-height:53px!important;width:100%!important;box-sizing:border-box!important;
font-size:13px!important;display:flex!important;align-items:center!important;justify-content:center!important}
/* Native st.columns(2) controls 50/50 width. Never mutate Streamlit's
   HorizontalBlock or Column display, flex, grid, or sizing internals here. */
.st-key-bp_hero_actions{margin-top:7px!important;margin-bottom:14px!important}
.st-key-bp_hero_actions div.stButton>button[kind="secondary"]{background:#091d35!important;border:1px solid #3a5d85!important}
/* Matchday strip and uninterrupted stadium hero. */
.bp-toolbar{min-height:37px;margin:0 0 10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
color:#a9c4e0;font-size:10px;font-weight:750}
.bp-toolbar span{background:#0e2641;border:1px solid #2b486b;border-radius:50px;padding:6px 13px}
.bp-toolbar span.bp-toolbar-active{background:#16396b;color:#c8e5ff;border-color:#2866ae}
.bp-hero{position:relative;isolation:isolate;min-height:286px;border:1px solid #2b5182;border-radius:9px;
overflow:hidden;background:linear-gradient(90deg,#061324e3 0%,#091c3588 52%,#07172a45 100%),var(--bp-stadium-image,none) center center/cover no-repeat;
display:grid;grid-template-columns:1fr minmax(310px,440px);gap:24px;align-items:center;padding:28px 30px 26px;
box-shadow:0 14px 36px #0004}
.bp-hero:after{content:"";position:absolute;inset:0;z-index:-1;
background:linear-gradient(0deg,#03101b55,transparent 30%);pointer-events:none}
.bp-eyebrow{font-family:"Manrope",sans-serif;font-size:10px;letter-spacing:.13em;color:#84c2fb;font-weight:900}
.bp-hero h1{font-family:"Sora",sans-serif!important;color:#f4f8ff;font-size:clamp(37px,4.8vw,69px)!important;
line-height:.99!important;letter-spacing:-.078em!important;margin:12px 0 9px!important;font-weight:800!important}
.bp-hero h1 em{color:#68b3ff;font-style:normal}
.bp-hero-content>p{color:#c0d3e8;font-size:12px;line-height:1.55;max-width:390px;margin:0}
.bp-hero-feature{padding:16px 21px;min-width:0;border:1px solid #40638a;border-radius:10px;
background:linear-gradient(140deg,#06172ae9,#142e52db);box-shadow:0 18px 32px #0006;backdrop-filter:blur(10px)}
.bp-small-label{text-transform:uppercase;font-size:10px;letter-spacing:.075em;font-weight:800;color:#91b2d6}
.bp-feature-teams,.bp-focus-score{display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);
text-align:center;align-items:center;gap:8px;padding:15px 0 13px}
.bp-feature-teams>div,.bp-focus-score>div{min-width:0;display:flex;flex-direction:column;align-items:center;gap:8px}
.bp-feature-teams img{height:48px!important;max-height:48px!important;width:48px!important;object-fit:contain!important}
.bp-feature-teams strong,.bp-focus-score strong{font-size:14px;line-height:1.35;overflow-wrap:anywhere;font-weight:800}
.bp-feature-time b,.bp-focus-middle b{font-size:22px;color:#edf5ff;font-variant-numeric:tabular-nums}
.bp-feature-time small,.bp-focus-middle small{font-size:10px;color:#9fbad8}
.bp-prob-triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.bp-prob-triplet>span{display:flex;align-items:center;justify-content:center;gap:6px;min-width:0;
border:1px solid #335879;background:#183554;border-radius:6px;padding:8px 4px;font-variant-numeric:tabular-nums}
.bp-prob-triplet>span:first-child{border-color:#2e75b9;background:#163b69}
.bp-prob-triplet b{font-size:13px;color:#e8f4ff}
.bp-prob-triplet small{font-size:9px;color:#a5c4e4}
.bp-feature-caption{font-size:9px;color:#9ab8d7;margin-top:10px;text-align:center}
/* 3/column data layout: panels align and fill; no irregular floating tile mosaics. */
.st-key-bp_dashboard{margin-top:12px}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]{
align-items:stretch!important;gap:12px!important}
.st-key-bp_dashboard [data-testid="stColumn"]{min-width:0!important}
.st-key-bp_fixtures,.st-key-bp_focus,.st-key-bp_sidebar{
border:1px solid var(--bp-line);border-radius:9px;background:#0a1a2e;padding:0 12px 13px;min-height:580px}
.bp-panel-head{display:flex;justify-content:space-between;align-items:center;gap:8px;
margin:0 -12px;padding:13px 14px;border-bottom:1px solid #25415d}
.bp-panel-head h2{font-family:"Sora",sans-serif!important;font-size:14px!important;letter-spacing:-.025em!important;
margin:0!important;color:#ecf4ff}
.bp-panel-head>span{color:#8eb3da;font-size:10px;font-weight:750;text-align:right}
.bp-list-head{display:flex;justify-content:space-between;padding:8px 2px 7px;
color:#7a9bbb;font-size:10px;letter-spacing:.07em;font-weight:750}
.bp-fixture-row{display:grid;grid-template-columns:42px minmax(0,1fr) 50px;gap:7px;
align-items:center;min-height:60px;padding:6px 1px;border-bottom:1px solid #1d344e}
.bp-fixture-row.selected{background:#153557;border-left:3px solid #4a9bff;padding-left:6px}
.bp-fixture-time{font-size:11px;color:#b5d5f2;font-variant-numeric:tabular-nums;font-weight:750}
.bp-fixture-clubs{min-width:0;display:flex;flex-direction:column;gap:3px}
.bp-fixture-clubs>div{display:flex;gap:6px;align-items:center;min-width:0}
.bp-fixture-clubs span{overflow:hidden;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;line-clamp:2;line-height:1.3;font-size:12px;font-weight:750;white-space:normal}
.bp-fixture-clubs img,.bp-side-crests img{height:17px!important;width:17px!important;max-height:17px!important;object-fit:contain!important}
.bp-fixture-prob{font-size:12px;color:#91caff;font-variant-numeric:tabular-nums;text-align:right;font-weight:850}
.st-key-bp_fixtures [data-testid="stHorizontalBlock"]{align-items:center!important;gap:4px!important;
margin-bottom:1px!important}
.st-key-bp_fixtures [data-testid="stColumn"]:first-child{flex-grow:1!important;min-width:0!important}
/* A real labelled action, not a narrow floating arrow. */
.st-key-bp_fixtures div.stButton>button{
height:38px!important;min-height:38px!important;max-height:38px!important;
background:#15375b!important;color:#bde1ff!important;
border:1px solid #35638e!important;border-radius:6px!important;
font-size:10.5px!important;font-weight:800!important;letter-spacing:0!important;
padding:0 4px!important;white-space:nowrap!important;box-shadow:none!important;
transform:none!important;line-height:1.1!important}
.st-key-bp_fixtures div.stButton>button:hover{
background:#20518a!important;border-color:#65aaff!important;color:#ffffff!important}
.st-key-bp_fixtures div.stButton>button:focus-visible{
outline:2px solid #72b8ff!important;outline-offset:2px!important}
.bp-focus-panel .bp-panel-head{margin:0 -12px}
.bp-focus-score{padding:16px 0 15px;grid-template-columns:minmax(0,1fr) 85px minmax(0,1fr)}
.bp-focus-score img{height:59px!important;width:59px!important;object-fit:contain!important;max-height:59px!important}
.bp-focus-score strong{font-size:15px}
.st-key-bp_feature_tabs{margin:0 -12px 7px!important;padding:0 12px!important;
border-bottom:1px solid #27405c!important}
.st-key-bp_feature_tabs [data-testid="stHorizontalBlock"]{gap:4px!important;flex-wrap:nowrap!important}
.st-key-bp_feature_tabs [data-testid="stColumn"]{min-width:0!important;flex:1 1 0!important;width:0!important;max-width:none!important}
.st-key-bp_feature_tabs div.stButton>button{background:transparent!important;border:0!important;
border-bottom:2px solid transparent!important;color:#96abc7!important;border-radius:0!important;
height:42px!important;min-height:42px!important;padding:3px 4px!important;
font-size:12px!important;font-weight:750!important;box-shadow:none!important;transform:none!important}
.st-key-bp_feature_tabs div.stButton>button[kind="primary"]{background:transparent!important;
color:#e6f5ff!important;border-bottom:2px solid #58adff!important;box-shadow:none!important}
.st-key-bp_feature_tabs div.stButton>button:hover{color:#fff!important;background:#173556!important}
.bp-feature-tab-body{padding-top:14px;min-height:260px}
.bp-feature-footnote{margin-top:12px!important;font-size:10px!important;line-height:1.55!important;color:#98b3d1!important}
.bp-spotlight-line{display:flex;justify-content:space-between;align-items:center;gap:12px;
min-height:41px;padding:8px 0;border-bottom:1px solid #274261;color:#a5c1db;font-size:12px}
.bp-spotlight-line b{font-size:13px;color:#e9f7ff;text-align:right;overflow-wrap:anywhere}

.bp-focus-photo{height:106px;margin:12px 0 13px;border:1px solid #365b82;border-radius:6px;
background:linear-gradient(0deg,#0a1d33e6,transparent 90%),var(--bp-stadium-image,none) center 54%/cover no-repeat;
display:flex;align-items:end;padding:10px 12px;color:#e4f4ff;font-size:10px;font-weight:800;letter-spacing:.08em}
.bp-focus-body h3{font-family:"Sora",sans-serif!important;font-size:19px;margin:5px 0 6px;color:#f6faff;line-height:1.3}
.bp-focus-body p{font-size:11px;color:#9eb8d4;line-height:1.55;margin:0 0 15px}
.bp-projected{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;
margin-top:13px;border-top:1px solid #26425b;padding-top:10px;color:#9dbad8}
.bp-projected strong{color:#f3f9ff;font-size:18px}
.bp-projected span{display:none}
.bp-side-panel{border-bottom:1px solid #2a4664;padding-bottom:10px}
.bp-side-fixture{display:grid;grid-template-columns:36px minmax(0,1fr) 40px;gap:7px;
align-items:center;padding:10px 1px;border-bottom:1px solid #1f3952;min-height:58px}
.bp-side-fixture:last-child{border-bottom:0}
.bp-side-crests{display:flex;flex-direction:column;gap:2px}
.bp-side-team{display:flex;flex-direction:column;gap:3px;min-width:0}
.bp-side-team b{font-size:12px;font-weight:750;line-height:1.3;white-space:normal;display:-webkit-box;-webkit-line-clamp:2;line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.bp-side-fixture time{font-size:12px;text-align:right;color:#b2cce7;font-variant-numeric:tabular-nums}
.bp-strongest .bp-panel-head{margin-top:8px}.bp-strongest>p{color:#91aac4;font-size:10px;margin:10px 0}
.bp-strength-row{display:grid;grid-template-columns:18px minmax(0,1fr) 44px;gap:8px;align-items:center;
border-bottom:1px solid #223b55;padding:9px 1px}
.bp-strength-row>span{color:#7198bd;font-size:10px}.bp-strength-row div{min-width:0;display:flex;flex-direction:column;gap:3px}
.bp-strength-row b{font-size:12px;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}
.bp-strength-row small{font-size:9px;color:#92a9c4;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}
.bp-strength-row>strong{text-align:right;color:#72b9ff;font-size:12px;font-variant-numeric:tabular-nums}
/* Equal Home actions: equal native Streamlit columns, identical buttons.
   No layout overrides of HorizontalBlock, stColumn or stButton wrappers. */
.st-key-bp_hero_actions div.stButton>button{
 width:100%!important;height:52px!important;min-height:52px!important;max-height:52px!important;
 box-sizing:border-box!important;margin:0!important;padding:0 10px!important;
 border-radius:7px!important;display:flex!important;align-items:center!important;
 justify-content:center!important;line-height:1.1!important;font-size:13px!important;
 font-weight:800!important;box-shadow:none!important;transform:none!important;white-space:nowrap!important;
}
/* Maintain a stable compact fixture layout with room for text action. */
.st-key-bp_fixtures [data-testid="stHorizontalBlock"]{gap:7px!important}
.st-key-bp_fixtures [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{min-width:0!important}
.st-key-bp_fixtures [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{min-width:105px!important}
.bp-fixture-row{grid-template-columns:38px minmax(0,1fr) 45px;gap:6px;min-height:63px}
.bp-fixture-clubs span{font-size:12.5px;font-weight:750;line-height:1.25}
/* Future fixtures: a disciplined scoreboard table, not free-height cards. */
.bp-future-panel{padding-bottom:10px}
.bp-future-date{color:#89b8e8;font-size:10px;font-weight:800;letter-spacing:.05em;
 margin:11px 0 8px;text-transform:uppercase}
.bp-future-list{max-height:360px;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;
 scrollbar-width:thin;scrollbar-color:#365a7d transparent;min-width:0}
.bp-future-table{table-layout:fixed;width:100%;border-collapse:collapse;border-spacing:0!important;
 background:transparent!important;margin:0!important}
.bp-future-table tbody,.bp-future-table tr{background:transparent!important}
.bp-future-table td{background:transparent!important;border:0!important;
 border-bottom:1px solid #203b58!important;padding:8px 3px!important;
 vertical-align:middle!important;box-sizing:border-box!important}
.bp-future-table tr:last-child td{border-bottom:0!important}
.bp-future-match-clubs{width:auto!important;min-width:0!important;text-align:left!important}
.bp-future-match-meta{width:108px!important;min-width:108px!important;text-align:right!important;
 font-variant-numeric:tabular-nums}
.bp-future-club{display:grid!important;grid-template-columns:24px minmax(0,1fr)!important;
 align-items:center!important;column-gap:9px!important;min-height:25px!important;
 margin:0!important;padding:0!important}
.bp-future-club+.bp-future-club{margin-top:3px!important}
.bp-future-badge{display:flex!important;align-items:center!important;justify-content:center!important;
 width:22px!important;height:22px!important;max-width:22px!important;max-height:22px!important;
 min-width:22px!important;min-height:22px!important;overflow:hidden!important}
.bp-future-badge img,.bp-future-badge svg{display:block!important;width:20px!important;height:20px!important;
 max-width:20px!important;max-height:20px!important;object-fit:contain!important;
 margin:0 auto!important;flex:none!important}
.bp-future-name{font-size:12px!important;font-weight:750!important;line-height:1.25!important;
 min-width:0!important;color:#e9f5ff!important;overflow-wrap:break-word}
.bp-future-match-meta time{display:block!important;font-size:12px!important;font-weight:800!important;
 line-height:1.3!important;color:#e9f5ff!important;white-space:nowrap}
.bp-future-match-meta small{display:block!important;font-size:9px!important;font-weight:700!important;
 line-height:1.4!important;margin-top:4px!important;color:#91aed0!important;
 letter-spacing:.005em!important;white-space:nowrap}
.bp-future-footnote{margin:10px 0 1px!important;color:#7c9bbf!important;
 font-size:10px!important;line-height:1.5!important}
.bp-future-empty{padding:15px 0;color:#9eb7d1;font-size:12px;line-height:1.5}
@media(max-width:1190px){.bp-future-match-meta{width:94px!important;min-width:94px!important}
 .bp-future-club{grid-template-columns:20px minmax(0,1fr)!important;column-gap:6px!important}
 .bp-future-badge{width:19px!important;height:19px!important;max-width:19px!important;max-height:19px!important;
  min-width:19px!important;min-height:19px!important}
 .bp-future-badge img,.bp-future-badge svg{width:18px!important;height:18px!important}}
/* Match Centre remains a real provider-independent preview; no fake LIVE claim. */
.md-scoreboard{border-radius:9px!important;background:linear-gradient(145deg,#12345b,#07182f)!important;
box-shadow:none!important}.md-panel{border-radius:9px!important;background:#0b1e34!important;box-shadow:none!important}
.md-panel h3{font-size:17px!important}.md-scoreboard-top{padding:12px 18px!important}
.md-scoreboard-bottom{font-size:10px!important;padding:8px 12px!important}
.md-score-feed.unavailable{background:transparent!important;border:0!important;color:#b1cbe4!important}
/* Legible club names across every page, without losing the compact grid. */
.md-fixture-club{font-size:13px!important;line-height:1.3!important}
.md-score-team strong{font-size:clamp(16px,2.1vw,21px)!important}
.md-display-teams{font-size:16px!important}
.fixture-choice-name{font-size:clamp(12px,1.2vw,15px)!important;line-height:1.35!important}
/* Other pages inherit the same broad professional typography and tighter shapes. */
.fixture-choice-card,.vault-filter-card,.evidence-vault-card,.v2-page-shell,.print-stage,.terminal-shell{
border-radius:10px!important}
.interaction-title{font-weight:800!important;letter-spacing:-.045em!important}
/* Theme control is small, not a second navigation row. */
[data-testid="stAppViewContainer"] .st-key-toggle_ui_theme div.stButton>button{min-height:34px!important}
@media(max-width:1100px){.block-container{padding-left:1rem!important;padding-right:1rem!important}
.bp-hero{grid-template-columns:1fr minmax(280px,390px);gap:13px;padding:21px}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]{gap:8px!important}
.bp-fixture-row{grid-template-columns:35px minmax(0,1fr) 40px}}
@media(max-width:780px){.bp-hero{grid-template-columns:1fr;min-height:0;gap:16px;padding:19px}
.bp-hero h1{font-size:clamp(42px,10vw,60px)!important}
.bp-brand-caption{display:none}.bp-brand-name{font-size:16px}
.bp-hero-feature{padding:10px 13px}.bp-feature-teams{padding:9px 0 10px}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{
min-width:calc(50% - 6px)!important;max-width:calc(50% - 6px)!important;flex-basis:calc(50% - 6px)!important}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:nth-child(2){order:0;min-width:100%!important;max-width:100%!important;flex-basis:100%!important}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{order:1}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{order:2}
.st-key-bp_fixtures,.st-key-bp_focus,.st-key-bp_sidebar{min-height:0}}
@media(max-width:540px){.block-container{padding:.6rem .66rem 3.3rem!important}
.bp-brand-name{font-size:14px}.bp-brand-version{font-size:8px;padding:3px 5px}
.bp-toolbar{gap:4px}.bp-toolbar span{font-size:9px;padding:5px 7px}
.bp-hero{padding:14px}.bp-hero h1{font-size:40px!important}
.bp-hero-content>p{font-size:11px}
.bp-feature-teams strong{font-size:13px}
.st-key-bp_dashboard>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{
min-width:100%!important;max-width:100%!important;flex-basis:100%!important}
.st-key-bp_hero_actions div.stButton>button{font-size:10px!important;padding:0 3px!important;height:52px!important;min-height:52px!important}
.st-key-matchday_top_nav [data-testid="stColumn"]{min-width:0!important;width:17.4%!important;max-width:17.4%!important;flex-basis:17.4%!important}
.st-key-matchday_top_nav [data-testid="stColumn"]:nth-child(4){display:none!important}
.st-key-matchday_top_nav [data-testid="stColumn"]:not(:nth-child(4)){min-width:0!important;width:20%!important;max-width:20%!important;flex-basis:20%!important}
.st-key-matchday_top_nav div.stButton>button{font-size:9px!important;letter-spacing:-.025em!important;padding:4px 1px!important}
.bp-focus-photo{height:114px}}
/* Loader uses same logo and exact background asset as Home, like one coherent product. */
.ucl-startup-overlay{background:linear-gradient(0deg,#041222ed 0%,#06162ae0 32%,#05132995 72%),
var(--bp-stadium-image,none) center center/cover no-repeat,#071329!important}
.ucl-startup-overlay:before{background:radial-gradient(circle at 50% 54%,#2a77db55,transparent 53%)!important}
.ucl-startup-inner{display:flex!important;align-items:center!important;justify-content:center!important;flex-direction:column!important}
.bp-startup-brand{display:flex;align-items:center;gap:14px;font-family:"Sora",sans-serif;
font-size:clamp(22px,4vw,37px);font-weight:800;letter-spacing:-.06em;color:#eff8ff}
.bp-startup-brand>span strong{color:#ffffff}.bp-startup-brand>span strong em{color:#63b3fa;font-style:normal}
.bp-startup-brand svg{height:55px;width:55px}
.bp-startup-tagline{font-family:"Manrope",sans-serif;font-size:12px;letter-spacing:.18em;
font-weight:800;margin-top:12px;color:#a9cfed;text-align:center}
.nf-splash-eyebrow{display:none!important}
.nf-splash-ball-area{width:190px!important;height:140px!important;margin:13px 0 4px!important}
.nf-splash-ball{height:112px!important;width:112px!important}
.ucl-startup-fallback{display:none!important}
.ucl-startup-copy{display:none!important}
.ucl-startup-system{font-size:10px!important;font-weight:800!important;letter-spacing:.07em!important}
.ucl-startup-progress{width:min(300px,75vw)!important;height:4px!important}
@media(prefers-reduced-motion:reduce){.bp-hero *{transition:none!important;animation:none!important}}
"""

BROADCAST_LIGHT_CSS = r"""
:root{--bp-bg:#edf4fb;--bp-surface:#f6faff;--bp-surface-2:#eaf2fd;--bp-line:#b9cddd;
--bp-text:#16283e;--bp-subtle:#57718b}
[data-testid="stAppViewContainer"],[data-testid="stMain"],html,body{background:linear-gradient(#eef5fd,#ddeafa)!important;color:#172c46!important}
.bp-brand-header{border-color:#b4cce4}.bp-brand-name{color:#132e4d}.bp-brand-caption{color:#526d89}
.bp-toolbar span,.st-key-bp_fixtures,.st-key-bp_focus,.st-key-bp_sidebar{background:#f4f9ff!important;border-color:#afc8e0!important;color:#213751!important}
.bp-panel-head{border-color:#bed4eb}.bp-panel-head h2,.bp-focus-body h3{color:#122c4a!important}
.bp-panel-head span,.bp-list-head,.bp-projected,.bp-strongest>p{color:#607c99!important}
.bp-fixture-row{border-color:#c5d8eb}.bp-fixture-row.selected{background:#dbe9fa}
.bp-fixture-clubs span,.bp-side-team b,.bp-strength-row b{color:#243e5b!important}
.bp-prob-triplet>span{background:#dce9f7;border-color:#b0cbe7}.bp-prob-triplet b{color:#1b3c5d}.bp-prob-triplet small{color:#517397}
.bp-focus-body p,.bp-strength-row small{color:#526e88}
.bp-side-panel,.bp-side-fixture,.bp-strength-row{border-color:#c2d7ea}
.bp-focus-photo{background:linear-gradient(0deg,#0d2d5ac0,transparent),var(--bp-stadium-image,none) center/cover no-repeat}
.st-key-matchday_top_nav div.stButton>button{color:#385777!important}
.st-key-matchday_top_nav div.stButton>button[kind="primary"]{background:#d9e9fc!important;color:#0c3f74!important}
.st-key-bp_feature_tabs div.stButton>button{color:#496788!important}
.st-key-bp_feature_tabs div.stButton>button[kind="primary"]{color:#19528c!important;border-bottom-color:#3079d5!important}
.bp-spotlight-line{color:#526e88;border-bottom-color:#bad0e4}
.bp-spotlight-line b{color:#203e5f}
.bp-future-table td{border-bottom-color:#c5d8eb!important}
.bp-future-name{color:#243e5b!important}
.bp-future-date,.bp-future-match-meta time{color:#2b5f92!important}
.bp-future-match-meta small,.bp-future-footnote,.bp-future-empty{color:#5d7b9a!important}
.st-key-bp_fixtures div.stButton>button{background:#e1edfa!important;color:#1e537e!important;border-color:#b0c9e5!important}
.st-key-bp_fixtures div.stButton>button:hover{background:#c6e1fa!important;color:#164673!important}
/* Loading and stadium hero intentionally stay night-blue in both themes. */
"""


def apply_broadcast_styles(render_html, theme: str = 'dark') -> None:
    css = BROADCAST_CSS + (BROADCAST_LIGHT_CSS if theme == 'light' else '')
    render_html('<style>' + css + '</style>')
