"""UCL Predictor Nightfall visual identity, layered over stable V2 styles.

No model operations and no data access. Styles intentionally leave the thermal
paper and the receipt probability/goal calculations alone.
"""
from __future__ import annotations

NIGHTFALL_CSS = r"""
/* Nightfall — licensed web-font CSS; fallback fonts remain readable offline. */
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@400;500;600;700;800&display=swap');
:root {
  --nf-night: #071329;
  --nf-deep: #0c1c38;
  --nf-card: #102442;
  --nf-card-hi: #17345b;
  --nf-line: #294a72;
  --nf-accent: #58ceff;
  --nf-blue: #397bea;
  --nf-white: #f0f6ff;
  --nf-muted: #a4bddb;
  --nf-radius: 16px;
  --green: #72debc; /* Keep good-health state green; never turn it into a probability. */
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  color:var(--nf-white)!important;
  background:radial-gradient(ellipse 65% 40% at 80% -8%,rgba(37,92,173,.26),transparent 95%),
    radial-gradient(circle 2px at 12% 16%,rgba(167,218,255,.21) 0 1px,transparent 2px),
    radial-gradient(circle 1px at 69% 12%,rgba(167,218,255,.22) 0 1px,transparent 2px),
    linear-gradient(160deg,#071329,#091b34 48%,#081329) !important;
  font-family:"Manrope",Inter,system-ui,sans-serif!important;
}
[data-testid="stHeader"],header[data-testid="stHeader"] {background:transparent!important;}
.block-container{max-width:1280px!important;padding-top:1rem!important;padding-bottom:4.2rem!important;}
/* Everything outside the physical receipt is a sports UI, not a terminal. */
.stApp p,.stApp label,.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],.app-kicker,.status-pill,.fixture-meta,
.fixture-choice-topline,.interaction-kicker,.interaction-copy,
.machine-idle-screen,.md-hero,.md-panel,.md-fixture-card,.md-scoreboard,
.md-section-heading,.md-feed-notice,.app-footer {
  font-family:"Manrope",Inter,system-ui,sans-serif!important;
}
.stApp h1,.stApp h2,.stApp h3,.interaction-title,.md-hero h1,
.md-section-heading h2,.md-panel h3,.md-score-value,.md-score-team strong,
.printer-page-intro .interaction-title,.evidence-page-intro .interaction-title {
  font-family:"Sora","Manrope",system-ui,sans-serif!important;
  letter-spacing:-.05em!important;
}
/* New small brand works on desktop and mobile without pushing CTAs below fold. */
.nf-app-header {display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:7px 0 12px;border-bottom:1px solid rgba(119,174,237,.24);color:#f4f8ff;
  font-family:"Sora",system-ui,sans-serif}
.nf-brand-lockup{display:flex;align-items:center;gap:11px;min-width:0;}
.nf-brand-emblem{width:35px;height:35px;display:grid;place-items:center;background:linear-gradient(140deg,#204c9e,#4396dd);
 border:1px solid #71b9ff85;border-radius:11px;color:#ecfaff;font-size:18px;box-shadow:0 0 18px #4199ed35;}
.nf-wordmark{font-size:clamp(15px,2.5vw,23px);letter-spacing:-.075em;font-weight:800;white-space:nowrap;}
.nf-wordmark>span{font-weight:600;letter-spacing:-.065em;}.nf-period{color:#5bd3ff;font-weight:800;}
.nf-header-right{display:flex;align-items:center;gap:10px;}
.nf-header-category{font-family:"Manrope",sans-serif;font-size:9px;letter-spacing:.13em;color:#8eacd5;font-weight:800;}
.nf-brand-edition{font-size:9px;font-weight:800;border:1px solid #426b9b;border-radius:6px;padding:5px 7px;color:#9edcff;}
.main-app-brand,.main-app-kicker,.main-app-subtitle,.top-rule{display:none!important;}
/* Tighter status line and new legibility. */
.consumer-status-row{justify-content:flex-start!important;margin:11px 0 11px!important;gap:6px!important;}
.status-pill{font-size:10px!important;letter-spacing:.03em!important;text-transform:none!important;
 background:#11284a!important;color:#b7cce6!important;border:1px solid #355174!important;
 border-radius:50px!important;padding:7px 11px!important;font-weight:750;}
.status-pill.green{color:#90efc9!important;border-color:#338d82!important;background:#103a3c!important;}
.status-pill.amber{color:#f7d593!important;border-color:#876d4a!important;background:#302c28!important;}
.status-pill.red{color:#ffc0ba!important;border-color:#8c514f!important;background:#372628!important;}
.creator-status-pill{border-style:dashed!important;}
/* Navigation: one deliberate app bar, never a row of industrial blocks. */
.st-key-matchday_top_nav{margin:8px 0 13px!important;background:#0e2442;border:1px solid #2c4e79;
 border-radius:14px;padding:6px!important;box-shadow:0 12px 30px #0002;}
.st-key-matchday_top_nav [data-testid="stHorizontalBlock"]{gap:5px!important;}
.st-key-matchday_top_nav div.stButton>button{background:transparent!important;color:#a9bfdc!important;
 border:1px solid transparent!important;border-radius:9px!important;min-height:40px!important;
 font-size:11px!important;font-weight:800!important;letter-spacing:.01em!important;}
.st-key-matchday_top_nav div.stButton>button:hover{background:#1c3a63!important;color:#fff!important;}
.st-key-matchday_top_nav div.stButton>button[kind="primary"]{background:linear-gradient(110deg,#3171d8,#438cf3)!important;
 border:1px solid #669ff8!important;color:#fff!important;box-shadow:0 4px 16px #2c7cdb48!important;}
.st-key-matchday_top_nav div.stButton>button:disabled{opacity:.48!important;}
/* Common control system, scoped to all Streamlit actions except thermal paper. */
div.stButton>button, button[kind="primary"]{
 font-family:"Manrope",sans-serif!important;font-weight:800!important;letter-spacing:.005em!important;
 border-radius:11px!important;min-height:44px!important;font-size:12px!important;
 background:#193457!important;border:1px solid #3b638f!important;color:#d7e8fc!important;
 box-shadow:none!important;text-transform:none!important;
 transition:background .18s ease,border-color .18s ease,transform .18s ease!important;
}
div.stButton>button[kind="primary"],button[kind="primary"]{
 background:linear-gradient(115deg,#3677df,#428ded)!important;border-color:#71aeff!important;color:#fff!important;
 box-shadow:0 7px 18px #2074df30!important;}
div.stButton>button:hover{background:#255286!important;color:#fff!important;border-color:#79beff!important;transform:translateY(-1px);}
div.stButton>button:focus-visible{outline:3px solid #78cfff!important;outline-offset:2px!important;}
div.stButton>button:disabled{opacity:.43!important;transform:none!important;}
/* Two *working* hero CTAs. They are visible immediately after hero. */
.st-key-nightfall_home_actions{margin-top:-6px!important;margin-bottom:12px!important;}
.st-key-nightfall_home_actions [data-testid="stHorizontalBlock"]{display:flex!important;flex-wrap:nowrap!important;gap:9px!important;}
.st-key-nightfall_home_actions [data-testid="stColumn"]{min-width:0!important;flex:1!important;}
.st-key-nightfall_home_actions div.stButton>button{min-height:57px!important;font-size:clamp(10px,1.8vw,14px)!important;
 font-weight:850!important;border-radius:11px!important;white-space:normal!important;}
.st-key-nightfall_home_actions [data-testid="stColumn"]:nth-child(2) button{background:#17365f!important;border:1px solid #4a85bf!important;}
/* Matchday hero: stadium architecture not a generic black gradient. */
.md-hero{border-radius:20px!important;border:1px solid #365d91!important;
 padding:clamp(21px,3vw,32px)!important;margin:6px 0 12px!important;
 background:radial-gradient(ellipse at 80% 18%,#2f68b650,transparent 43%),
  radial-gradient(ellipse at 55% 110%,#0c6c8866,transparent 50%),
  linear-gradient(135deg,#183a6d,#0b1f41 55%,#08152d)!important;
 min-height:244px;isolation:isolate;box-shadow:0 18px 40px #0004!important;}
.md-hero::before{content:"";position:absolute;inset:-30% 0 auto 0;height:180%;z-index:-1;
 background:repeating-radial-gradient(ellipse at 52% 107%,transparent 0 66px,#8abfff1a 67px 68px,transparent 69px 102px);
 pointer-events:none;transform:scaleY(.45);}
.md-hero::after{content:"";position:absolute;bottom:0;left:0;right:0;height:3px;
 background:linear-gradient(90deg,transparent,#80d9ff,#3288ec,transparent);opacity:.65;}
.md-hero-main{gap:20px!important;grid-template-columns:1.25fr .8fr!important;}
.md-eyebrow{font-size:10px!important;font-family:"Manrope",sans-serif!important;letter-spacing:.12em!important;color:#9bdcff!important;}
.md-date{margin:11px 0 7px!important;font-weight:750;color:#b8cbe5!important;}
.md-hero h1{font-size:clamp(36px,5.1vw,64px)!important;line-height:1.04!important;
 letter-spacing:-.065em!important;font-weight:800!important;margin-bottom:9px!important;}
.md-hero h1 em{color:#73d7ff!important;text-shadow:0 0 34px #53c6f93a;}
.md-hero p{font-size:12px!important;line-height:1.5!important;margin-bottom:12px!important;max-width:410px;}
.md-hero-tags{gap:6px!important;}.md-hero-tags span{border-radius:20px!important;padding:6px 9px!important;
 font-size:9px!important;background:#2b4d7880!important;color:#d2e8ff!important;}
.md-hero-display{border-radius:15px!important;padding:17px!important;background:#102849bc!important;
 border:1px solid #5074a4!important;backdrop-filter:blur(5px);}
.md-display-crests{margin:10px 0!important;}.md-display-crests img{width:56px!important;height:56px!important;}
.md-display-caption{font-size:9px!important;margin-top:8px!important;color:#92b1d5!important;}
.md-section-heading h2{color:#f2f8ff!important;font-size:clamp(23px,3vw,30px)!important;}
.md-section-heading p{color:#a4b9d5!important;font-size:12px!important;}
.md-fixture-card{border-radius:15px!important;background:linear-gradient(145deg,#152d50,#0b1d37)!important;
 border-color:#315682!important;box-shadow:0 10px 24px #0003!important;}
.md-fixture-card:hover{border-color:#69a9e9!important;}
.md-scoreboard{background:radial-gradient(ellipse at 50% 0%,#28549a8c,transparent 60%),linear-gradient(140deg,#16325e,#0b1e3f)!important;}
.md-score-value{font-variant-numeric:tabular-nums!important;font-weight:800!important;}
.md-panel,.md-feed-notice{border-radius:15px!important;}
.md-feed-notice{padding:10px 14px!important;font-size:11px!important;background:#10213ac9!important;}
/* Existing fixtures, Vault and printer surroundings join the new UI. */
.v2-page-shell{margin:12px 0!important;}
.interaction-kicker{color:#75d9ff!important;letter-spacing:.12em!important;font-size:10px!important;}
.interaction-title{color:var(--nf-white)!important;font-weight:850!important;
 font-size:clamp(24px,4vw,39px)!important;line-height:1.13!important;}
.interaction-copy,.machine-idle-note{color:#a9c1de!important;}
.fixture-choice-card{background:linear-gradient(155deg,#152a49,#0d203c)!important;
 border:1px solid #31517a!important;border-radius:14px!important;}
.fixture-choice-card.selected{background:#1b3762!important;border-color:#6eb5fb!important;box-shadow:0 0 0 1px #71b3fa38!important;}
.fixture-choice-card.selected .fixture-choice-topline{color:#80dbff!important;}
.fixture-choice-topline{color:#a7c1df!important;font-size:10px!important;letter-spacing:.02em!important;}
.fixture-choice-name{color:#f1f5fc!important;font-weight:800!important;}
.fixture-choice-vs{color:#74addc!important;font-family:"Sora",sans-serif!important;}
.machine-idle-screen{border:1px solid #2c507c!important;background:#10233f!important;border-radius:13px!important;}
.machine-idle-title{color:#e6f3ff!important;font-family:"Sora",sans-serif!important;}
.machine-idle-kicker{color:#80ccff!important;font-family:"Manrope",sans-serif!important;}
/* Select dropdowns: Streamlit BaseWeb control + popup are both themed. */
[data-testid="stSelectbox"] label{color:#b7d3f0!important;font-family:"Manrope",sans-serif!important;font-size:11px!important;letter-spacing:.02em!important;font-weight:800!important;}
div[data-baseweb="select"]>div{background:#142e54!important;border:1px solid #426c9b!important;
 border-radius:12px!important;min-height:50px!important;box-shadow:inset 0 1px #ffffff0b!important;}
div[data-baseweb="select"] span{color:#f0f7ff!important;font-family:"Manrope",sans-serif!important;font-size:12px!important;}
div[data-baseweb="popover"]{border-radius:12px!important;overflow:hidden!important;}
[data-baseweb="menu"]{background:#122b4d!important;border:1px solid #3b6696!important;color:#ebf5ff!important;}
[data-baseweb="menu"] li{font-family:"Manrope",sans-serif!important;font-size:12px!important;color:#e7f3ff!important;}
[data-baseweb="menu"] li:hover{background:#254f82!important;}
input:focus-visible,[data-baseweb="select"]:focus-within>div{outline:2px solid #6ad0ff!important;outline-offset:2px!important;}
.app-footer{border-top:1px solid #315378!important;color:#89a9ce!important;font-size:10px!important;
 line-height:1.8!important;text-transform:none!important;letter-spacing:.015em!important;}
/* Original onboarding loader redesigned without extra media requests. */
.ucl-startup-overlay{background:radial-gradient(ellipse at center,#173b72 0%,#0c2148 40%,#06132a 84%)!important;
 animation:uclStartupOverlayExit .55s ease 2.35s forwards!important;}
.ucl-startup-overlay:before{content:"";position:absolute;inset:0;opacity:.5;pointer-events:none;
 background:repeating-radial-gradient(ellipse at 50% 120%,transparent 0 45px,#88cfff22 46px 47px,transparent 48px 78px);}
.ucl-startup-scan{display:none!important;}
.ucl-startup-fallback{font-family:"Sora",system-ui,sans-serif!important;font-size:clamp(27px,6vw,54px)!important;
 letter-spacing:-.07em!important;font-weight:850!important;color:#f6faff!important;}
.ucl-startup-fallback span{font-family:"Sora",system-ui,sans-serif!important;font-size:.5em!important;
 color:#68d7ff!important;letter-spacing:-.03em!important;margin-top:3px!important;}
.ucl-startup-copy{font-family:"Manrope",sans-serif!important;font-weight:750!important;
 letter-spacing:.14em!important;color:#a7d8ff!important;}
.ucl-startup-system{font-family:"Manrope",sans-serif!important;color:#8caed7!important;letter-spacing:.06em!important;}
.ucl-startup-progress{height:4px!important;background:#335784!important;border-radius:4px!important;}
.ucl-startup-progress-fill{background:linear-gradient(90deg,#327cea,#76e2ff)!important;}
.nf-splash-ball-area{position:relative;height:164px;width:270px;margin-bottom:8px;
 display:flex;align-items:center;justify-content:center;}
.nf-ball-trail{position:absolute;inset:65% 0 0;border-radius:50%;background:radial-gradient(ellipse,#69cfff55,transparent 70%);
 filter:blur(13px);animation:nf-trail-glow 2s ease-in-out both;}
.nf-splash-ball{display:block;width:128px;height:128px;filter:drop-shadow(0 0 22px #6ed9ff55);
 animation:nf-ball-entry 1.8s cubic-bezier(.14,.72,.2,1) both;transform-origin:center;}
.nf-splash-ball svg{display:block;height:100%;width:100%;}
.nf-splash-eyebrow{color:#9ac8f0;font-size:10px;font-weight:800;letter-spacing:.18em;margin-bottom:10px;}
@keyframes nf-ball-entry{from{opacity:0;transform:translateX(-150px) rotate(-220deg) scale(.72)}
 65%{opacity:1;transform:translateX(18px) rotate(30deg) scale(1.04)}to{opacity:1;transform:translateX(0) rotate(0deg) scale(1)}}
@keyframes nf-trail-glow{from{opacity:0;transform:scaleX(.4)}to{opacity:1;transform:scaleX(1)}}
/* Compact phone navigation: printer is contextually accessible from home and fixture. */
@media(max-width:760px){
 .block-container{padding-top:.45rem!important;padding-left:.75rem!important;padding-right:.75rem!important;}
 .nf-header-category{display:none!important;}.nf-brand-emblem{width:30px;height:30px;}
 .consumer-status-row{margin:7px 0!important;}
 .status-pill{font-size:9px!important;padding:5px 8px!important;}
 .st-key-matchday_top_nav{margin:6px 0 12px!important;}
 .st-key-matchday_top_nav [data-testid="stHorizontalBlock"]{display:flex!important;flex-wrap:nowrap!important;gap:3px!important;}
 .st-key-matchday_top_nav [data-testid="stColumn"]{flex:1 1 25%!important;width:25%!important;min-width:0!important;max-width:25%!important;}
 .st-key-matchday_top_nav [data-testid="stColumn"]:nth-child(4){display:none!important;}
 .st-key-matchday_top_nav div.stButton>button{padding:5px 2px!important;font-size:clamp(9px,2vw,11px)!important;min-height:40px!important;}
 .md-hero{padding:20px!important;min-height:0!important;}
 .md-hero-main{grid-template-columns:1fr!important;gap:12px!important;}
 .md-hero h1{font-size:clamp(39px,9vw,59px)!important;}
 .md-hero-display{padding:12px!important;}
 .md-display-crests{margin:8px 0!important;}
 .md-display-crests .club-crest-wrap{width:54px!important;height:54px!important;}
 .md-display-crests img{width:49px!important;height:49px!important;}
 .md-section-heading{margin-top:16px!important;}
 .st-key-nightfall_home_actions div.stButton>button{min-height:55px!important;padding:8px 5px!important;}
}
@media(max-width:390px){.nf-wordmark{font-size:14px!important;}
 .nf-brand-emblem{width:27px;height:27px;}.nf-brand-lockup{gap:6px;}
 .st-key-matchday_top_nav div.stButton>button{font-size:9px!important;}
 .status-pill{font-size:8px!important;}
}
@media(prefers-reduced-motion:reduce){
 .nf-splash-ball,.nf-ball-trail{animation:none!important;transform:none!important;opacity:1!important;}
 .ucl-startup-overlay{animation:uclStartupOverlayExit .01s linear .5s forwards!important;}
 div.stButton>button,.md-fixture-card{transition:none!important;transform:none!important;}
}
"""

NIGHTFALL_LIGHT_CSS = r"""
:root{--nf-night:#ebf2fb;--nf-deep:#f1f6fc;--nf-card:#eaf3ff;--nf-card-hi:#d4e6fc;
--nf-line:#94b7da;--nf-white:#10274b;--nf-muted:#4c6688;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
 background:radial-gradient(ellipse at 60% -10%,#bbd9ff 0%,transparent 60%),linear-gradient(155deg,#f3f8ff,#e8f1fc)!important;
 color:#183157!important;}
.nf-app-header{color:#122b51!important;border-bottom-color:#b4c9e3!important;}
.nf-brand-edition{border-color:#a0bfe1!important;color:#3a668b!important;}
.nf-header-category{color:#4a6383!important;}
.consumer-status-row .status-pill{background:#e6effb!important;color:#2b527e!important;border-color:#adc3df!important;}
.st-key-matchday_top_nav{background:#ddeafa!important;border-color:#b3c9e5!important;}
.st-key-matchday_top_nav div.stButton>button{color:#355479!important;}
.st-key-matchday_top_nav div.stButton>button:hover{background:#bfd8f6!important;color:#173b6f!important;}
.md-section-heading h2,.interaction-title{color:#193858!important;}
.md-section-heading p,.interaction-copy{color:#4c6383!important;}
.fixture-choice-card{background:linear-gradient(145deg,#f6faff,#e6f0fc)!important;border-color:#9bb9da!important;}
.fixture-choice-name{color:#1e3b60!important;}.fixture-choice-topline{color:#557399!important;}
.fixture-choice-card.selected{background:#d8e9fd!important;border-color:#2d7cca!important;}
.md-feed-notice{background:#dceafa!important;color:#355675!important;border-color:#aac7e5!important;}
.md-feed-notice strong{color:#17416f!important;}
.md-fixture-card,.md-panel,.md-scoreboard,.md-hero{color:#f0f6ff!important;} /* Night cards stay dark on light surrounds. */
[data-testid="stSelectbox"] label{color:#345a82!important;}
div[data-baseweb="select"]>div{background:#e8f3ff!important;border-color:#8bafd4!important;}
div[data-baseweb="select"] span{color:#173d65!important;}
div.stButton>button{background:#dfedfc!important;color:#24527d!important;border-color:#9ab9dc!important;}
div.stButton>button[kind="primary"]{background:linear-gradient(115deg,#3677df,#428ded)!important;color:#fff!important;}
.st-key-nightfall_home_actions [data-testid="stColumn"]:nth-child(2) button{background:#d8eafb!important;color:#214e81!important;border-color:#84add5!important;}
.app-footer{color:#597498!important;border-top-color:#aac2df!important;}
"""


def apply_nightfall_styles(render_html, *, theme: str = "dark") -> None:
    normalized = str(theme).strip().lower()
    css = NIGHTFALL_CSS
    if normalized == "light":
        css += "\n" + NIGHTFALL_LIGHT_CSS
    render_html("<style>" + css + "</style>")
