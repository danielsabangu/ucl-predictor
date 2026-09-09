from __future__ import annotations

import io
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
)

from src.config import (
    BASE_DIR,
    PREDICTIONS_DIR,
)

from src.production.club_crests import (
    get_crest_path,
    team_initials,
)


# ============================================================
# SETTINGS
# ============================================================

LOCAL_TZ = ZoneInfo(
    "Africa/Johannesburg"
)

PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_complete.csv"
)

EXPORT_DIR = (
    BASE_DIR
    / "reports"
    / "exports"
)

RECEIPT_WIDTH = 900
RECEIPT_HEIGHT = 1650

PAPER_RGB = (
    238,
    233,
    220,
)

INK_RGB = (
    18,
    18,
    18,
)

MUTED_RGB = (
    78,
    75,
    67,
)

LIGHT_RGB = (
    205,
    199,
    184,
)


# ============================================================
# FONT DISCOVERY
# ============================================================


def existing_path(
    candidates: list[str],
) -> Path | None:

    for candidate in candidates:

        path = Path(
            candidate
        )

        if path.exists():
            return path

    return None


def regular_font_path() -> Path | None:

    return existing_path(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )


def bold_font_path() -> Path | None:

    return existing_path(
        [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    )


def mono_font_path() -> Path | None:

    return existing_path(
        [
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\cour.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ]
    )


def load_font(
    *,
    size: int,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.ImageFont:

    if mono:

        path = mono_font_path()

    elif bold:

        path = bold_font_path()

    else:

        path = regular_font_path()

    if path is not None:

        return ImageFont.truetype(
            str(path),
            size=size,
        )

    return ImageFont.load_default()


# ============================================================
# TEXT HELPERS
# ============================================================


def safe_filename_component(
    value: str,
) -> str:

    value = str(
        value
    ).strip()

    value = re.sub(
        r'[<>:"/\\|?*]+',
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    return (
        value
        or
        "team"
    )


def percentage_probability(
    value,
) -> float:

    return (
        float(value)
        *
        100.0
    )


def margin_pp(
    value,
) -> float:

    number = float(
        value
    )

    # Some pipeline versions may store the margin as
    # 0.4745 while others may already store 47.45 pp.
    if abs(number) <= 1.0:

        return (
            number
            *
            100.0
        )

    return number


def center_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy_y: int,
    text: str,
    font,
    fill=INK_RGB,
    width: int = RECEIPT_WIDTH,
) -> None:

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    text_width = (
        bbox[2]
        -
        bbox[0]
    )

    x = int(
        (
            width
            -
            text_width
        )
        /
        2
    )

    draw.text(
        (
            x,
            xy_y,
        ),
        text,
        font=font,
        fill=fill,
    )


def fit_font(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    max_width: int,
    start_size: int,
    minimum_size: int,
    bold: bool = True,
) -> ImageFont.ImageFont:

    size = int(
        start_size
    )

    while size >= minimum_size:

        font = load_font(
            size=size,
            bold=bold,
        )

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        width = (
            bbox[2]
            -
            bbox[0]
        )

        if width <= max_width:

            return font

        size -= 2

    return load_font(
        size=minimum_size,
        bold=bold,
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    left: int = 70,
    right: int = RECEIPT_WIDTH - 70,
    dash: int = 12,
    gap: int = 9,
) -> None:

    x = left

    while x < right:

        draw.line(
            (
                x,
                y,
                min(
                    x + dash,
                    right,
                ),
                y,
            ),
            fill=MUTED_RGB,
            width=2,
        )

        x += (
            dash
            +
            gap
        )


def draw_row(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    value: str,
    font,
    value_font=None,
) -> None:

    if value_font is None:

        value_font = font

    draw.text(
        (
            80,
            y,
        ),
        label,
        font=font,
        fill=MUTED_RGB,
    )

    bbox = draw.textbbox(
        (0, 0),
        value,
        font=value_font,
    )

    width = (
        bbox[2]
        -
        bbox[0]
    )

    draw.text(
        (
            RECEIPT_WIDTH
            -
            80
            -
            width,
            y,
        ),
        value,
        font=value_font,
        fill=INK_RGB,
    )


# ============================================================
# CRESTS
# ============================================================


def load_crest_image(
    team: str,
    *,
    size: int = 125,
) -> Image.Image | None:

    path = get_crest_path(
        team
    )

    if path is None:

        return None

    try:

        image = Image.open(
            path
        )

        image = image.convert(
            "RGBA"
        )

    except Exception:

        # SVG or unsupported file types safely fall back
        # to the initials badge.
        return None

    image.thumbnail(
        (
            size,
            size,
        ),
        Image.Resampling.LANCZOS,
    )

    grayscale = ImageOps.grayscale(
        image
    )

    grayscale = ImageOps.autocontrast(
        grayscale
    )

    alpha = image.getchannel(
        "A"
    )

    result = Image.merge(
        "RGBA",
        (
            grayscale,
            grayscale,
            grayscale,
            alpha,
        ),
    )

    return result


def draw_initials_badge(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    team: str,
    center_x: int,
    center_y: int,
) -> None:

    radius = 52

    box = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )

    draw.ellipse(
        box,
        outline=INK_RGB,
        width=4,
    )

    initials = team_initials(
        team
    )

    font = load_font(
        size=28,
        bold=True,
        mono=True,
    )

    bbox = draw.textbbox(
        (0, 0),
        initials,
        font=font,
    )

    text_width = (
        bbox[2]
        -
        bbox[0]
    )

    text_height = (
        bbox[3]
        -
        bbox[1]
    )

    draw.text(
        (
            center_x
            -
            text_width
            /
            2,
            center_y
            -
            text_height
            /
            2
            -
            3,
        ),
        initials,
        font=font,
        fill=INK_RGB,
    )


def draw_team_crest(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    team: str,
    center_x: int,
    center_y: int,
) -> None:

    crest = load_crest_image(
        team
    )

    if crest is None:

        draw_initials_badge(
            image,
            draw,
            team=team,
            center_x=center_x,
            center_y=center_y,
        )

        return

    x = int(
        center_x
        -
        crest.width
        /
        2
    )

    y = int(
        center_y
        -
        crest.height
        /
        2
    )

    image.alpha_composite(
        crest,
        (
            x,
            y,
        ),
    )


# ============================================================
# BARCODE
# ============================================================


def draw_fake_barcode(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    identifier: str,
) -> None:

    left = 185
    right = RECEIPT_WIDTH - 185

    seed = sum(
        ord(
            character
        )
        for character in identifier
    )

    x = left
    index = 0

    while x < right:

        value = (
            seed
            +
            index * 17
        )

        width = (
            2
            +
            value % 5
        )

        gap = (
            2
            +
            value % 4
        )

        draw.rectangle(
            (
                x,
                y,
                min(
                    x + width,
                    right,
                ),
                y + 55,
            ),
            fill=INK_RGB,
        )

        x += (
            width
            +
            gap
        )

        index += 1


# ============================================================
# RECEIPT
# ============================================================


def build_receipt_image(
    row: pd.Series,
) -> Image.Image:

    image = Image.new(
        "RGBA",
        (
            RECEIPT_WIDTH,
            RECEIPT_HEIGHT,
        ),
        (
            PAPER_RGB[0],
            PAPER_RGB[1],
            PAPER_RGB[2],
            255,
        ),
    )

    draw = ImageDraw.Draw(
        image
    )

    mono_small = load_font(
        size=20,
        mono=True,
    )

    mono_medium = load_font(
        size=24,
        mono=True,
    )

    mono_bold = load_font(
        size=25,
        mono=True,
        bold=True,
    )

    heading_font = load_font(
        size=46,
        bold=True,
    )

    score_font = load_font(
        size=150,
        bold=True,
    )

    pick_font = load_font(
        size=40,
        bold=True,
    )

    probability_font = load_font(
        size=34,
        bold=True,
    )

    kickoff = pd.Timestamp(
        row[
            "kickoff_utc"
        ]
    )

    if kickoff.tzinfo is None:

        kickoff = kickoff.tz_localize(
            "UTC"
        )

    kickoff = kickoff.tz_convert(
        LOCAL_TZ
    )

    home_team = str(
        row[
            "home_team"
        ]
    )

    away_team = str(
        row[
            "away_team"
        ]
    )

    receipt_id = (
        f"{kickoff.strftime('%Y%m%d%H%M')}-"
        f"{home_team[:3].upper()}-"
        f"{away_team[:3].upper()}"
    )

    # ========================================================
    # TOP TEAR
    # ========================================================

    for x in range(
        0,
        RECEIPT_WIDTH,
        24,
    ):

        draw.polygon(
            [
                (
                    x,
                    0,
                ),
                (
                    x + 12,
                    12,
                ),
                (
                    x + 24,
                    0,
                ),
            ],
            fill=(
                235,
                230,
                216,
                255,
            ),
        )

    # ========================================================
    # BRAND
    # ========================================================

    center_text(
        draw,
        xy_y=55,
        text="UCL//FORECAST",
        font=heading_font,
    )

    center_text(
        draw,
        xy_y=112,
        text="PRODUCTION PREDICTION RECEIPT",
        font=mono_small,
        fill=MUTED_RGB,
    )

    center_text(
        draw,
        xy_y=154,
        text=f"TXN {receipt_id}",
        font=mono_small,
        fill=MUTED_RGB,
    )

    draw_dashed_line(
        draw,
        y=205,
    )

    # ========================================================
    # MATCH META
    # ========================================================

    center_text(
        draw,
        xy_y=230,
        text=kickoff.strftime(
            "%A %d %B %Y"
        ).upper(),
        font=mono_small,
    )

    center_text(
        draw,
        xy_y=260,
        text=(
            f"{kickoff.strftime('%H:%M')} SAST "
            "// UEFA CHAMPIONS LEAGUE"
        ),
        font=mono_small,
        fill=MUTED_RGB,
    )

    # ========================================================
    # CRESTS
    # ========================================================

    home_center_x = 260
    away_center_x = 640
    crest_center_y = 385

    draw_team_crest(
        image,
        draw,
        team=home_team,
        center_x=home_center_x,
        center_y=crest_center_y,
    )

    draw_team_crest(
        image,
        draw,
        team=away_team,
        center_x=away_center_x,
        center_y=crest_center_y,
    )

    home_font = fit_font(
        draw,
        text=home_team.upper(),
        max_width=300,
        start_size=38,
        minimum_size=22,
    )

    away_font = fit_font(
        draw,
        text=away_team.upper(),
        max_width=300,
        start_size=38,
        minimum_size=22,
    )

    home_bbox = draw.textbbox(
        (0, 0),
        home_team.upper(),
        font=home_font,
    )

    away_bbox = draw.textbbox(
        (0, 0),
        away_team.upper(),
        font=away_font,
    )

    draw.text(
        (
            home_center_x
            -
            (
                home_bbox[2]
                -
                home_bbox[0]
            )
            /
            2,
            465,
        ),
        home_team.upper(),
        font=home_font,
        fill=INK_RGB,
    )

    draw.text(
        (
            away_center_x
            -
            (
                away_bbox[2]
                -
                away_bbox[0]
            )
            /
            2,
            465,
        ),
        away_team.upper(),
        font=away_font,
        fill=INK_RGB,
    )

    center_text(
        draw,
        xy_y=480,
        text="VS",
        font=mono_bold,
        fill=MUTED_RGB,
    )

    # ========================================================
    # SCORE
    # ========================================================

    center_text(
        draw,
        xy_y=525,
        text=str(
            row[
                "predicted_scoreline"
            ]
        ),
        font=score_font,
    )

    center_text(
        draw,
        xy_y=680,
        text="ESTIMATED SCORELINE // POISSON BRIDGE",
        font=mono_small,
        fill=MUTED_RGB,
    )

    # ========================================================
    # PICK BOX
    # ========================================================

    pick_box = (
        95,
        730,
        RECEIPT_WIDTH - 95,
        840,
    )

    draw.rectangle(
        pick_box,
        outline=INK_RGB,
        width=4,
    )

    center_text(
        draw,
        xy_y=748,
        text="CATBOOST RESULT PICK",
        font=mono_small,
        fill=MUTED_RGB,
    )

    pick = str(
        row[
            "predicted_selection"
        ]
    ).upper()

    selected_pick_font = fit_font(
        draw,
        text=pick,
        max_width=610,
        start_size=40,
        minimum_size=24,
    )

    center_text(
        draw,
        xy_y=785,
        text=pick,
        font=selected_pick_font,
    )

    draw_dashed_line(
        draw,
        y=880,
    )

    # ========================================================
    # H/D/A
    # ========================================================

    probabilities = [
        (
            "HOME",
            percentage_probability(
                row[
                    "home_probability"
                ]
            ),
            "H",
        ),
        (
            "DRAW",
            percentage_probability(
                row[
                    "draw_probability"
                ]
            ),
            "D",
        ),
        (
            "AWAY",
            percentage_probability(
                row[
                    "away_probability"
                ]
            ),
            "A",
        ),
    ]

    predicted_result = str(
        row[
            "predicted_result"
        ]
    )

    cell_lefts = [
        75,
        310,
        545,
    ]

    cell_width = 205

    for (
        label,
        value,
        result_code,
    ), x in zip(
        probabilities,
        cell_lefts,
    ):

        selected = (
            result_code
            ==
            predicted_result
        )

        box = (
            x,
            910,
            x + cell_width,
            1015,
        )

        if selected:

            draw.rectangle(
                box,
                fill=INK_RGB,
            )

            label_fill = PAPER_RGB
            value_fill = PAPER_RGB

        else:

            draw.rectangle(
                box,
                outline=MUTED_RGB,
                width=2,
            )

            label_fill = MUTED_RGB
            value_fill = INK_RGB

        label_font = mono_small

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=label_font,
        )

        draw.text(
            (
                x
                +
                (
                    cell_width
                    -
                    (
                        bbox[2]
                        -
                        bbox[0]
                    )
                )
                /
                2,
                927,
            ),
            label,
            font=label_font,
            fill=label_fill,
        )

        probability_text = (
            f"{value:.2f}%"
        )

        bbox = draw.textbbox(
            (0, 0),
            probability_text,
            font=probability_font,
        )

        draw.text(
            (
                x
                +
                (
                    cell_width
                    -
                    (
                        bbox[2]
                        -
                        bbox[0]
                    )
                )
                /
                2,
                960,
            ),
            probability_text,
            font=probability_font,
            fill=value_fill,
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    draw_dashed_line(
        draw,
        y=1055,
    )

    confidence = float(
        row[
            "confidence_percent"
        ]
    )

    draw_row(
        draw,
        y=1080,
        label="EVIDENCE CONFIDENCE",
        value=f"{confidence:.2f}%",
        font=mono_medium,
        value_font=mono_bold,
    )

    track_left = 80
    track_right = RECEIPT_WIDTH - 80
    track_top = 1125
    track_bottom = 1142

    draw.rectangle(
        (
            track_left,
            track_top,
            track_right,
            track_bottom,
        ),
        fill=LIGHT_RGB,
    )

    filled_width = int(
        (
            track_right
            -
            track_left
        )
        *
        max(
            0.0,
            min(
                100.0,
                confidence,
            ),
        )
        /
        100.0
    )

    draw.rectangle(
        (
            track_left,
            track_top,
            track_left
            +
            filled_width,
            track_bottom,
        ),
        fill=INK_RGB,
    )

    profile = str(
        row[
            "model_decisiveness_profile"
        ]
    ).replace(
        "_",
        " ",
    )

    confidence_label = str(
        row[
            "confidence_label"
        ]
    ).replace(
        "_",
        " ",
    )

    center_text(
        draw,
        xy_y=1157,
        text=(
            f"{confidence_label} // "
            f"{profile} // "
            f"{margin_pp(row['model_top_two_margin']):.2f} PP"
        ),
        font=mono_small,
        fill=MUTED_RGB,
    )

    # ========================================================
    # SCORE DETAILS
    # ========================================================

    draw_dashed_line(
        draw,
        y=1205,
    )

    exact_score_share = (
        float(
            row[
                "predicted_scoreline_probability"
            ]
        )
        *
        100.0
    )

    draw_row(
        draw,
        y=1230,
        label="EXACT SCORE SHARE",
        value=(
            f"{exact_score_share:.2f}%"
        ),
        font=mono_small,
        value_font=mono_bold,
    )

    draw_row(
        draw,
        y=1265,
        label="EXPECTED GOALS",
        value=(
            f"{float(row['score_home_lambda']):.2f}"
            " - "
            f"{float(row['score_away_lambda']):.2f}"
        ),
        font=mono_small,
        value_font=mono_bold,
    )

    draw_row(
        draw,
        y=1300,
        label="BRIDGE QUALITY",
        value=(
            f"{row['score_bridge_quality']} // "
            f"{float(row['score_bridge_max_error_pp']):.3f} PP"
        ),
        font=mono_small,
        value_font=mono_bold,
    )

    # ========================================================
    # FEATURES
    # ========================================================

    draw_dashed_line(
        draw,
        y=1345,
    )

    feature_text = (
        f"ELO {int(row['elo_features_available'])}/6  //  "
        f"UCL {int(row['ucl_form_features_available'])}/81  //  "
        f"DOM {int(row['domestic_features_available'])}/102"
    )

    center_text(
        draw,
        xy_y=1370,
        text=feature_text,
        font=mono_small,
    )

    feature_text_2 = (
        f"SQUAD {int(row['squad_features_available'])}/101  //  "
        f"TOTAL {int(row['overall_features_available'])}/290  "
        f"({float(row['overall_feature_availability']) * 100:.2f}%)"
    )

    center_text(
        draw,
        xy_y=1403,
        text=feature_text_2,
        font=mono_small,
    )

    # ========================================================
    # WARNING
    # ========================================================

    draw.rectangle(
        (
            80,
            1450,
            RECEIPT_WIDTH - 80,
            1457,
        ),
        fill=INK_RGB,
    )

    center_text(
        draw,
        xy_y=1475,
        text="PROBABILITY != CONFIDENCE",
        font=mono_bold,
    )

    center_text(
        draw,
        xy_y=1507,
        text=(
            "SCORELINE IS AN AUXILIARY POISSON ESTIMATE"
        ),
        font=mono_small,
        fill=MUTED_RGB,
    )

    # ========================================================
    # BARCODE
    # ========================================================

    draw_fake_barcode(
        draw,
        y=1550,
        identifier=receipt_id,
    )

    center_text(
        draw,
        xy_y=1615,
        text=receipt_id.replace(
            "-",
            "",
        ),
        font=mono_small,
        fill=MUTED_RGB,
    )

    return image


# ============================================================
# PNG
# ============================================================


def build_receipt_png(
    row: pd.Series,
) -> bytes:

    image = build_receipt_image(
        row
    )

    output = io.BytesIO()

    image.convert(
        "RGB"
    ).save(
        output,
        format="PNG",
        optimize=True,
    )

    output.seek(
        0
    )

    return output.getvalue()


# ============================================================
# PDF
# ============================================================


def build_receipt_pdf(
    row: pd.Series,
) -> bytes:

    try:

        from reportlab.lib.utils import (
            ImageReader,
        )

        from reportlab.pdfgen import (
            canvas,
        )

    except ImportError as error:

        raise RuntimeError(
            "PDF export requires reportlab. "
            "Install it with: "
            "pip install reportlab"
        ) from error

    png_bytes = (
        build_receipt_png(
            row
        )
    )

    output = io.BytesIO()

    # 0.75 converts our pixel dimensions to a compact
    # point-based PDF receipt while preserving the ratio.
    page_width = (
        RECEIPT_WIDTH
        *
        0.75
    )

    page_height = (
        RECEIPT_HEIGHT
        *
        0.75
    )

    pdf = canvas.Canvas(
        output,
        pagesize=(
            page_width,
            page_height,
        ),
    )

    image_reader = ImageReader(
        io.BytesIO(
            png_bytes
        )
    )

    pdf.drawImage(
        image_reader,
        0,
        0,
        width=page_width,
        height=page_height,
        preserveAspectRatio=True,
        mask="auto",
    )

    pdf.showPage()

    pdf.save()

    output.seek(
        0
    )

    return output.getvalue()


# ============================================================
# FILENAMES
# ============================================================


def export_filename(
    row: pd.Series,
    *,
    extension: str,
) -> str:

    home = safe_filename_component(
        row[
            "home_team"
        ]
    )

    away = safe_filename_component(
        row[
            "away_team"
        ]
    )

    extension = (
        extension
        .lower()
        .lstrip(".")
    )

    return (
        f"UCL_Forecast_"
        f"{home}_vs_{away}."
        f"{extension}"
    )


# ============================================================
# CLI TEST
# ============================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "UCL PREDICTOR - RECEIPT EXPORT TEST"
    )

    print(
        "=" * 78
    )

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Prediction artifact not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    dataframe = pd.read_csv(
        PREDICTIONS_FILE,
        low_memory=False,
    )

    if dataframe.empty:

        raise RuntimeError(
            "Prediction artifact is empty."
        )

    dataframe[
        "kickoff_utc"
    ] = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        utc=True,
        errors="raise",
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    upcoming = dataframe[
        dataframe[
            "kickoff_utc"
        ]
        >
        now
    ].copy()

    if upcoming.empty:

        # Useful for local testing after kickoff too.
        row = dataframe.iloc[
            0
        ]

    else:

        upcoming = upcoming.sort_values(
            "kickoff_utc"
        )

        row = upcoming.iloc[
            0
        ]

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        EXPORT_DIR
        /
        export_filename(
            row,
            extension="png",
        )
    )

    pdf_path = (
        EXPORT_DIR
        /
        export_filename(
            row,
            extension="pdf",
        )
    )

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

    png_path.write_bytes(
        png_bytes
    )

    pdf_path.write_bytes(
        pdf_bytes
    )

    print(
        f"\nFixture:"
    )

    print(
        f"{row['home_team']} "
        f"vs "
        f"{row['away_team']}"
    )

    print(
        f"\nPNG:"
    )

    print(
        png_path
    )

    print(
        f"{len(png_bytes):,} bytes"
    )

    print(
        f"\nPDF:"
    )

    print(
        pdf_path
    )

    print(
        f"{len(pdf_bytes):,} bytes"
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RECEIPT EXPORTS CREATED SUCCESSFULLY ✓"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()