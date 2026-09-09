from __future__ import annotations

import os
import sys


# ============================================================
# PRODUCTION PYTHON ENVIRONMENT
# ============================================================
#
# Windows terminals may default to cp1252. The UCL Predictor
# contains legitimate Unicode club names and status symbols:
#
# Atlético Madrid
# Bodø/Glimt
# Fenerbahçe
# ŠK Slovan Bratislava
# ✓
#
# Configure UTF-8 before importing the actual pipeline.
# Child Python processes inherit these values.
# ============================================================

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"


# ============================================================
# CURRENT PROCESS STDOUT / STDERR
# ============================================================
#
# PYTHONUTF8 affects Python when it starts. Because this module
# is already executing, explicitly reconfigure the current
# process as well.
# ============================================================


def configure_stream(
    stream,
):
    """
    Configure a text stream for UTF-8 when supported.

    On platforms/environments where reconfigure() is not
    available, leave the stream untouched rather than failing
    the production pipeline.
    """

    reconfigure = getattr(
        stream,
        "reconfigure",
        None,
    )

    if reconfigure is None:
        return

    try:
        reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:
        # Encoding setup must never become a reason for the
        # production pipeline itself to fail.
        pass


configure_stream(
    sys.stdout
)

configure_stream(
    sys.stderr
)


# ============================================================
# PIPELINE
# ============================================================
#
# Import only after environment/terminal configuration.
# argparse continues to work normally because run_pipeline.main
# reads the same sys.argv passed to this launcher.
# ============================================================

from src.production.run_pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()