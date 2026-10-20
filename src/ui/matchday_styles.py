"""Stage 1 football-inspired Matchday styling. Scoped away from the receipt printer."""
from __future__ import annotations

MATCHDAY_CSS = r"""
/* MATCHDAY / LIVE SCREENS — deliberately scoped; do not alter receipt paper. */
.md-hero,.md-fixture-card,.md-scoreboard,.md-panel,.md-feed-notice{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;box-sizing:border-box}
.md-hero *, .md-fixture-card *, .md-scoreboard *, .md-panel *, .md-feed-notice *{box-sizing:border-box}
.md-hero{position:relative;overflow:hidden;background:radial-gradient(ellipse at 84% 18%,rgba(48,119,231,.35),transparent 42%),linear-gradient(134deg,#111e40,#081226 62%,#060e20);border:1px solid #293f67;border-radius:20px;padding:clamp(22px,4vw,44px);color:#eef5ff;margin:12px 0 25px;box-shadow:0 22px 52px rgba(0,0,0,.2)}
.md-hero-glow{position:absolute;inset:-12% -15% auto auto;width:38%;height:110%;background:radial-gradient(circle,rgba(46,126,241,.18),transparent 68%);pointer-events:none}
.md-eyebrow{font-size:10px;font-weight:850;letter-spacing:.19em;color:#83baff;text-transform:uppercase}
.md-star{color:#c1d7ff;margin-right:6px}
.md-hero-main{display:grid;grid-template-columns:1.18fr 1fr;gap:28px;align-items:center;position:relative;z-index:1}
.md-date{margin:20px 0 10px;color:#9fbfe9;font-size:11px;font-weight:700;letter-spacing:.06em}
.md-hero h1{font-family:Impact,"Arial Black",sans-serif;letter-spacing:.015em;font-size:clamp(46px,7vw,88px);line-height:.96;margin:0 0 17px;color:#f5f9ff}
.md-hero h1 em{font-style:normal;color:#81caff;text-shadow:0 0 38px rgba(79,170,255,.22)}
.md-hero p{color:#b8cae5;font-size:13px;line-height:1.7;margin:0 0 22px}
.md-hero-tags{display:flex;flex-wrap:wrap;gap:8px}
.md-hero-tags span{border:1px solid #43608a;background:rgba(31,64,112,.4);border-radius:5px;padding:9px 11px;color:#d7e8ff;font-size:10px;font-weight:800;letter-spacing:.05em}
.md-hero-display{padding:25px 18px;border-radius:15px;border:1px solid #375783;background:linear-gradient(130deg,rgba(54,94,155,.34),rgba(14,29,56,.75));text-align:center;box-shadow:inset 0 1px 0 rgba(255,255,255,.06)}
.md-display-title{font-weight:800;font-size:10px;letter-spacing:.12em;color:#a6c9f6}
.md-display-crests{display:flex;align-items:center;justify-content:space-around;gap:12px;margin:20px 0 15px}
.md-display-crests img{width:66px!important;height:66px!important;object-fit:contain!important;max-height:66px!important}
.md-display-crests b{color:#6487b5;font-size:19px;font-family:Impact,sans-serif}
.md-display-teams{font-size:15px;font-weight:850;line-height:1.5;color:#f0f5ff}
.md-display-teams span{color:#6fa8e8;padding:0 4px}
.md-display-caption{font-size:9px;letter-spacing:.045em;line-height:1.6;color:#839ec4;margin-top:15px}
.md-section-heading{margin:21px 0 15px;color:#f1f6ff}
.md-section-heading h2{font-size:clamp(23px,3vw,32px);letter-spacing:-.035em;margin:7px 0 5px;font-weight:900}
.md-section-heading p{color:#91a7ca;margin:0;font-size:12px;line-height:1.5}
.md-fixture-card{border:1px solid #294467;background:linear-gradient(165deg,#132849,#0d1a32);border-radius:13px;min-height:185px;padding:15px;color:#f4f7ff;margin:6px 0 8px;box-shadow:0 12px 22px rgba(0,0,0,.14);transition:border-color .18s,transform .18s}
.md-fixture-card:hover{border-color:#5f9feb;transform:translateY(-2px)}
.md-fixture-top,.md-fixture-foot{display:flex;justify-content:space-between;gap:10px;align-items:center;color:#9bb4d7;font-size:10px;font-weight:850;letter-spacing:.06em}
.md-fixture-state{color:#a6d1f7;background:#19375c;border-radius:4px;padding:5px 7px;font-size:9px}
.md-fixture-identity{display:grid;grid-template-columns:minmax(0,1fr) 25px minmax(0,1fr);gap:6px;align-items:center;margin:17px 0 20px}
.md-fixture-club{display:flex;flex-direction:column;gap:8px;align-items:center;text-align:center;min-width:0;font-size:12px;font-weight:800;line-height:1.25;overflow-wrap:anywhere}
.md-fixture-club img{width:41px!important;height:41px!important;object-fit:contain!important;max-height:41px!important}
.md-versus{text-align:center;font-weight:900;font-size:12px;color:#5b7ea8}
.md-fixture-foot{border-top:1px solid #29405d;padding-top:11px;color:#8399bb;font-size:9px}
.md-fixture-foot span:last-child{color:#72bdff;font-size:18px}
.md-feed-notice{display:flex;gap:11px;align-items:flex-start;border:1px solid #3d5276;border-radius:10px;background:#13233f;color:#c9dafa;padding:14px 17px;margin:14px 0 18px;font-size:11px;line-height:1.6}
.md-feed-notice strong{color:#e9f3ff;letter-spacing:.06em;font-size:11px}
.md-feed-notice .md-feed-dot{display:inline-block;flex-shrink:0;width:9px;height:9px;border-radius:50%;background:#efbe63;margin-top:5px;box-shadow:0 0 10px rgba(239,190,99,.28)}
.md-scoreboard{overflow:hidden;border:1px solid #376195;border-radius:17px;background:radial-gradient(ellipse at 50% 10%,rgba(64,124,215,.4),transparent 64%),linear-gradient(145deg,#172b52,#0a1630);color:#f3f7ff;margin-bottom:14px;box-shadow:0 20px 45px rgba(0,0,0,.22)}
.md-scoreboard-top{display:flex;justify-content:space-between;align-items:center;gap:9px;padding:16px 20px;border-bottom:1px solid #37537a;color:#91b5e7;font-size:10px;font-weight:850;letter-spacing:.075em;flex-wrap:wrap}
.md-score-feed{font-size:10px;border-radius:5px;padding:6px 9px;background:#223854;color:#c3d2e6}
.md-score-feed.connected{color:#b4fae4;background:#145d53}
.md-scoreboard-core{display:grid;grid-template-columns:minmax(0,1fr) minmax(90px,1.05fr) minmax(0,1fr);gap:7px;align-items:center;min-height:215px;padding:18px}
.md-score-team{text-align:center;display:flex;flex-direction:column;align-items:center;gap:9px;min-width:0}
.md-score-team img{width:67px!important;height:67px!important;object-fit:contain!important;max-height:67px!important}
.md-score-team strong{font-size:clamp(13px,2vw,19px);line-height:1.3;overflow-wrap:anywhere}
.md-score-team span{color:#7d9bc3;font-weight:750;letter-spacing:.12em;font-size:9px}
.md-score-centre{text-align:center;white-space:nowrap}
.md-score-value{font-family:Impact,"Arial Black",sans-serif;font-size:clamp(42px,6vw,82px);font-variant-numeric:tabular-nums;letter-spacing:.035em;line-height:1.12}
.md-score-caption{color:#9ecaff;font-size:11px;font-weight:800;margin-top:12px}
.md-scoreboard-bottom{padding:13px 20px;border-top:1px solid #37537a;text-align:center;color:#9bb4d8;font-size:11px;line-height:1.55}
.md-panel{background:linear-gradient(155deg,#101f3d,#0a1730);border:1px solid #2c4266;border-radius:13px;padding:20px;color:#f0f5ff;margin:9px 0 18px}
.md-panel-overline{font-size:10px;color:#78b7f3;font-weight:850;letter-spacing:.13em}
.md-panel h3{font-size:21px;margin:9px 0 20px;font-weight:900;letter-spacing:-.03em}
.md-panel h3 span{color:#7da8d6}
.md-panel p{font-size:12px;color:#a9bbd4;line-height:1.65}
.md-forecast-row{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #2c4365;padding:12px 0;align-items:center}
.md-forecast-row span{font-size:10px;font-weight:800;color:#8fa8cb;letter-spacing:.04em}
.md-forecast-row strong{font-size:12px;text-align:right;overflow-wrap:anywhere}
.md-chance-title{font-size:10px;letter-spacing:.10em;color:#b5d4fa;margin-top:19px;margin-bottom:9px;font-weight:850}
.md-prob-track{display:flex;width:100%;height:9px;gap:3px;border-radius:6px;overflow:hidden}
.md-prob-track span:nth-child(1){background:#4aabff}.md-prob-track span:nth-child(2){background:#6d7fa2}.md-prob-track span:nth-child(3){background:#e8bc6c}
.md-prob-legend{display:flex;justify-content:space-between;gap:5px;color:#a8c5e6;font-size:10px;font-weight:850;margin-top:8px}
.md-panel-disclaimer{color:#88a3c3!important;font-size:11px!important;margin-bottom:0!important}
.md-pitch{width:100%;display:block;border-radius:8px;margin-top:10px;border:1px solid #285f57}
@media(max-width:750px){.md-hero-main{grid-template-columns:1fr}.md-hero-display{padding:20px 12px}.md-hero h1{font-size:clamp(43px,12vw,67px)}.md-scoreboard-core{min-height:180px;padding:12px 7px}.md-score-team img{width:45px!important;height:45px!important}.md-score-value{font-size:clamp(32px,9vw,50px)}.md-score-team strong{font-size:12px}.md-scoreboard-top{padding:10px;font-size:9px}}
@media(prefers-reduced-motion:reduce){.md-fixture-card{transition:none}.md-fixture-card:hover{transform:none}}

/* Existing crest_html returns branded crest wrappers with grayscale globally. */
.md-display-crests .club-crest-wrap{width:70px!important;height:70px!important;display:flex;align-items:center;justify-content:center}
.md-fixture-club .club-crest-wrap{width:44px!important;height:44px!important;display:flex;align-items:center;justify-content:center}
.md-score-team .club-crest-wrap{width:70px!important;height:70px!important;display:flex;align-items:center;justify-content:center}
.md-display-crests .club-crest,.md-fixture-club .club-crest,.md-score-team .club-crest{filter:none!important;opacity:1!important}
@media(max-width:650px){
.md-fixture-card{min-height:176px;padding:11px 9px}
.md-fixture-club{font-size:10px}
.md-fixture-top{font-size:9px;flex-wrap:wrap}
.md-fixture-foot{font-size:8px}
.md-score-team .club-crest-wrap{width:45px!important;height:45px!important}
}

/* Five tap-friendly navigation destinations, including narrow phone widths. */
@media(max-width:650px){
.st-key-matchday_top_nav [data-testid="stHorizontalBlock"]{display:flex!important;flex-wrap:nowrap!important;gap:4px!important}
.st-key-matchday_top_nav [data-testid="stColumn"]{flex:1 1 0%!important;width:20%!important;min-width:0!important;max-width:20%!important}
.st-key-matchday_top_nav div.stButton>button{padding:.26rem .04rem!important;font-size:.55rem!important;min-height:40px!important;letter-spacing:-.015em!important;white-space:nowrap!important}
}
body:has(.md-hero) .st-key-matchday_top_nav button[kind="primary"],
body:has(.md-scoreboard) .st-key-matchday_top_nav button[kind="primary"]{background:#2e71d7!important;border-color:#4283e6!important;color:#fff!important}
/* Light-mode remains consistent without affecting the existing receipt printer. */
body:has(.md-hero) .main-app-wordmark-predictor,body:has(.md-scoreboard) .main-app-wordmark-predictor{color:#6dc2ff}
"""


def apply_matchday_styles(render_html) -> None:
    render_html("<style>" + MATCHDAY_CSS + "</style>")
