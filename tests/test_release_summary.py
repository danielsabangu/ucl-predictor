import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.production.release import (
    print_release_summary,
)


class ReleaseSummaryTests(unittest.TestCase):
    def test_01_no_stage_does_not_claim_identical(
        self,
    ):
        def fake_git_output(*args):
            if args[:2] == (
                "status",
                "--short",
            ):
                return (
                    " M data/app/current_predictions.csv\n"
                    " M data/app/pipeline_status.json\n"
                    " M data/app/snapshot_metadata.json"
                )

            if args[:2] == (
                "branch",
                "--show-current",
            ):
                return "v2-product-experience"

            return ""

        output = io.StringIO()

        with patch(
            "src.production.release.git_output",
            side_effect=fake_git_output,
        ):
            with redirect_stdout(output):
                print_release_summary(
                    [],
                    stage=False,
                )

        rendered = output.getvalue()

        self.assertIn(
            "Dry run completed successfully.",
            rendered,
        )

        self.assertIn(
            "--no-stage",
            rendered,
        )

        self.assertIn(
            "working tree contains unstaged changes",
            rendered,
        )

        self.assertNotIn(
            "snapshot is identical",
            rendered,
        )

        self.assertNotIn(
            "nothing new to commit or push",
            rendered,
        )

    def test_02_staged_release_shows_commit_commands(
        self,
    ):
        def fake_git_output(*args):
            if args[:2] == (
                "status",
                "--short",
            ):
                return (
                    "M  data/app/current_predictions.csv"
                )

            if args[:2] == (
                "branch",
                "--show-current",
            ):
                return "v2-product-experience"

            return ""

        output = io.StringIO()

        with patch(
            "src.production.release.git_output",
            side_effect=fake_git_output,
        ):
            with redirect_stdout(output):
                print_release_summary(
                    [
                        "data/app/current_predictions.csv",
                    ],
                    stage=True,
                )

        rendered = output.getvalue()

        self.assertIn(
            "The following files are staged:",
            rendered,
        )

        self.assertIn(
            'git commit -m "Refresh UCL predictions',
            rendered,
        )

        self.assertIn(
            "git push origin v2-product-experience",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
