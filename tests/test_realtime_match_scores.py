"""Score adapter regressions: no guessed live scores, auth header only on server."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ui.realtime_match_scores import (  # noqa: E402
    EVENTS_ENDPOINT, ProviderScore, _canonical_team,
    fetch_champions_league_events, match_fixture_score,
)


FIXTURE = {
    'home_team': 'Fenerbahçe',
    'away_team': 'Slavia Prague',
    'kickoff_utc': datetime(2026, 10, 20, 16, 45, tzinfo=timezone.utc),
}


def event(state='in', completed=False, home=1, away=0, home_name='Fenerbahce',
          away_name='Slavia Prague', kickoff='2026-10-20T16:45:00Z'):
    return {
        'id': 'e1', 'date': kickoff,
        'homeTeam': {'name': home_name, 'score': home},
        'awayTeam': {'name': away_name, 'score': away},
        'status': {'state': state, 'completed': completed, 'clock': '54:20',
                   'detail': '54th minute'},
    }


def test_expected_club_normalization_is_explicit():
    assert _canonical_team('Fenerbahçe') == _canonical_team('Fenerbahce')
    assert _canonical_team('Roma') == _canonical_team('AS Roma')
    assert _canonical_team('Porto') == _canonical_team('FC Porto')
    assert _canonical_team('ŠK Slovan Bratislava') == _canonical_team('Slovan Bratislava')
    assert _canonical_team('Bodø/Glimt') == _canonical_team('Bodo/Glimt')



def test_explicit_matchday_aliases_cover_provider_variants():
    pairs = [
        ('Como 1907', 'Como'),
        ('Lille', 'Lille OSC'),
        ('Galatasaray', 'Galatasaray Istanbul'),
        ('Inter Milan', 'Inter'),
        ('Shakhtar Donetsk', 'FC Shakhtar Donetsk'),
        ('Sporting CP', 'Sporting Lisbon'),
        ('LASK Linz', 'LASK'),
        ('Real Betis Balompié', 'Real Betis'),
        ('Club Brugge', 'Club Brugge KV'),
        ('Lens', 'Racing Club de Lens'),
        ('Bayern Munich', 'FC Bayern München'),
    ]
    for predicted, provider in pairs:
        assert _canonical_team(predicted) == _canonical_team(provider)


def test_known_team_alias_does_not_bypass_kickoff_or_home_away():
    fixture = {
        'home_team': 'Como 1907', 'away_team': 'Manchester United',
        'kickoff_utc': datetime(2026, 10, 21, 16, 45, tzinfo=timezone.utc),
    }
    result = event(home_name='Como', away_name='Manchester United',
                   kickoff='2026-10-21T16:45:00Z')
    assert match_fixture_score(fixture, [result]) == ProviderScore(1, 0, 'LIVE', 54)
    assert match_fixture_score(fixture, [
        event(home_name='Manchester United', away_name='Como',
              kickoff='2026-10-21T16:45:00Z')
    ]) is None
    assert match_fixture_score(fixture, [event(
        home_name='Como', away_name='Manchester United',
        kickoff='2026-10-22T16:45:00Z')]) is None


def test_pitch_is_always_renderable_as_svg_without_tracking():
    from src.ui.matchday import pitch_svg_bytes
    markup = pitch_svg_bytes()
    assert markup.startswith(b'<svg')
    assert b'viewBox="0 0 600 310"' in markup
    assert b'<rect' in markup and b'</svg>' in markup
    assert b'player-position' not in markup



def test_streamlit_presents_pitch_as_svg_html_not_pillow_image():
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'pitch_svg_bytes().decode("utf-8")' in source
    assert 'components.html(' in source
    assert 'st.image(pitch_svg_bytes()' not in source



def test_verified_live_score_and_minute():
    assert match_fixture_score(FIXTURE, [event()]) == ProviderScore(1, 0, 'LIVE', 54)


def test_halftime_and_finished_score_labels():
    half = event()
    half['status']['halftime'] = True
    assert match_fixture_score(FIXTURE, [half]).status == 'HALF_TIME'
    assert match_fixture_score(FIXTURE, [event('post', True, 2, 1)]) == ProviderScore(2, 1, 'FULL_TIME')
    assert match_fixture_score(FIXTURE, [event('post', False, 2, 1)]) is None


def test_scheduled_zero_zero_is_not_a_score():
    assert match_fixture_score(FIXTURE, [event('pre', False, 0, 0)]) is None


def test_wrong_fixture_teams_orientation_or_date_never_match():
    assert match_fixture_score(FIXTURE, [event(home_name='Slavia Prague', away_name='Fenerbahce')]) is None
    assert match_fixture_score(FIXTURE, [event(home_name='Galatasaray')]) is None
    assert match_fixture_score(FIXTURE, [event(kickoff='2026-10-21T16:45:00Z')]) is None
    assert match_fixture_score(FIXTURE, [event(kickoff='2026-10-20T19:00:00Z')]) is None


def test_missing_malformed_and_ambiguous_scores_never_match():
    assert match_fixture_score(FIXTURE, [event(home='NaN')]) is None
    assert match_fixture_score(FIXTURE, [event(home=True)]) is None
    assert match_fixture_score(FIXTURE, [event(), event(home=3)]) is None
    assert match_fixture_score(FIXTURE, [{'homeTeam': {'name': 'Fenerbahce'}}]) is None


def test_provider_request_is_single_authenticated_league_call():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, _): return json.dumps({'success': True, 'data': [event()]}).encode()

    def mock_urlopen(req, timeout):
        assert req.full_url == EVENTS_ENDPOINT
        assert req.get_header('Authorization') == 'Bearer private-key'
        assert timeout == 8.0
        return Response()

    with patch('src.ui.realtime_match_scores.urlopen', side_effect=mock_urlopen) as opener:
        result = fetch_champions_league_events('private-key')
    assert len(result) == 1 and opener.call_count == 1


def test_no_request_without_secret():
    with patch('src.ui.realtime_match_scores.urlopen') as mocked:
        with pytest.raises(ValueError):
            fetch_champions_league_events('')
        mocked.assert_not_called()


def test_runtime_uses_optional_key_and_caches_shared_events():
    app = (ROOT / 'app.py').read_text(encoding='utf-8')
    source = ast.parse(app)
    assert 'ttl=600' in app
    assert 'run_every="10m"' in app
    assert 'render_selected_match_score(' in app
    assert 'render_ucl_live_scoreboard()' in app
    assert '_matchday_near_kickoff(centre_rows)' in app
    assert 'score=match_score' in app
    assert 'RSA_API_KEY' in app
    assert 'UCL_CREATOR_MODE' in app
    assert 'from src.ui.realtime_match_scores import' in app
    assert source is not None


def test_provider_matchday_markup_distinguishes_current_and_final_score():
    source = (ROOT / 'src/ui/matchday.py').read_text(encoding='utf-8')
    assert 'CURRENT SCORE' in source
    assert 'FINAL SCORE' in source
    assert 'Live score data is currently unavailable.' in source
    assert 'Scores supplied by Realtime Sports API' in source
