from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# KNOWN DAMAGED STRINGS
# ============================================================

MOJIBAKE_TEAM_FIXES = {
    "MÅ K Å½ilina": "MŠK Žilina",
    "MÅ\u00a0K Å½ilina": "MŠK Žilina",
    "BaÅlul GalaÈ": "Oțelul Galați",
    "OÅ£elul GalaÅ£i": "Oțelul Galați",
    "OÈ›elul GalaÈ›i": "Oțelul Galați",
    "QarabaÄ": "Qarabağ",
}


# ============================================================
# CANONICAL TEAM ALIASES
# ============================================================

TEAM_ALIASES = {
    # England
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",

    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",

    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",

    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "leicester city fc": "Leicester City",

    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",

    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",

    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united": "Manchester United",
    "manchester united fc": "Manchester United",

    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "newcastle united fc": "Newcastle United",

    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "tottenham hotspur fc": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",

    # Spain
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",

    "real madrid": "Real Madrid",
    "real madrid cf": "Real Madrid",

    "atletico madrid": "Atlético Madrid",
    "ath madrid": "Atlético Madrid",
    "club atletico de madrid": "Atlético Madrid",

    "athletic bilbao": "Athletic Bilbao",
    "athletic club": "Athletic Bilbao",
    "ath bilbao": "Athletic Bilbao",

    "real sociedad": "Real Sociedad",
    "real sociedad de futbol": "Real Sociedad",
    "sociedad": "Real Sociedad",

    "sevilla": "Sevilla",
    "sevilla fc": "Sevilla",

    "valencia": "Valencia",
    "valencia cf": "Valencia",

    "villarreal": "Villarreal",
    "villarreal cf": "Villarreal",

    "malaga": "Málaga",
    "malaga cf": "Málaga",

    "girona": "Girona",
    "girona fc": "Girona",

    # Germany
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern munich": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",

    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",

    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",

    "bayer leverkusen": "Bayer Leverkusen",
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",

    "schalke": "Schalke 04",
    "schalke 04": "Schalke 04",
    "fc schalke 04": "Schalke 04",

    "borussia monchengladbach": "Borussia Mönchengladbach",
    "monchengladbach": "Borussia Mönchengladbach",
    "m'gladbach": "Borussia Mönchengladbach",

    "stuttgart": "VfB Stuttgart",
    "vfb stuttgart": "VfB Stuttgart",

    "werder bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen",

    "wolfsburg": "Wolfsburg",
    "vfl wolfsburg": "Wolfsburg",

    "hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim",

    "ein frankfurt": "Eintracht Frankfurt",
    "eintracht frankfurt": "Eintracht Frankfurt",

    "union berlin": "Union Berlin",
    "1. fc union berlin": "Union Berlin",

    # Italy
    "ac milan": "AC Milan",
    "milan": "AC Milan",

    "inter": "Inter Milan",
    "inter milan": "Inter Milan",
    "internazionale": "Inter Milan",
    "fc internazionale": "Inter Milan",
    "fc internazionale milano": "Inter Milan",

    "juventus": "Juventus",
    "juventus fc": "Juventus",

    "napoli": "Napoli",
    "ssc napoli": "Napoli",

    "roma": "Roma",
    "as roma": "Roma",

    "lazio": "Lazio",
    "ss lazio": "Lazio",

    "atalanta": "Atalanta",
    "atalanta bc": "Atalanta",

    "bologna": "Bologna",
    "bologna fc": "Bologna",
    "bologna fc 1909": "Bologna",

    # France
    "paris saint germain": "Paris Saint-Germain",
    "paris saint-germain": "Paris Saint-Germain",
    "paris saint-germain fc": "Paris Saint-Germain",
    "paris s-g": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",

    "monaco": "Monaco",
    "as monaco": "Monaco",
    "as monaco fc": "Monaco",

    "marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",

    "lyon": "Lyon",
    "olympique lyonnais": "Lyon",

    "lille": "Lille",
    "lille osc": "Lille",

    "auxerre": "Auxerre",
    "aj auxerre": "Auxerre",

    "lens": "Lens",
    "racing club de lens": "Lens",

    "rennes": "Rennes",
    "stade rennais": "Rennes",
    "stade rennais fc": "Rennes",
    "stade rennais fc 1901": "Rennes",

    "brest": "Brest",
    "stade brestois": "Brest",
    "stade brestois 29": "Brest",

    # Portugal
    "benfica": "Benfica",
    "sl benfica": "Benfica",
    "sport lisboa e benfica": "Benfica",

    "porto": "Porto",
    "fc porto": "Porto",

    "sporting": "Sporting CP",
    "sporting cp": "Sporting CP",
    "sporting lisbon": "Sporting CP",
    "sporting clube de portugal": "Sporting CP",

    "braga": "Braga",
    "sc braga": "Braga",
    "sporting clube de braga": "Braga",

    # Netherlands
    "ajax": "Ajax",
    "afc ajax": "Ajax",

    "psv": "PSV Eindhoven",
    "psv eindhoven": "PSV Eindhoven",

    "feyenoord": "Feyenoord",
    "feyenoord rotterdam": "Feyenoord",

    "az": "AZ Alkmaar",
    "az alkmaar": "AZ Alkmaar",

    "twente": "Twente",
    "fc twente": "Twente",
    "fc twente '65": "Twente",

    # Belgium
    "club brugge": "Club Brugge",
    "club brugge kv": "Club Brugge",

    "anderlecht": "Anderlecht",
    "rsc anderlecht": "Anderlecht",

    "genk": "Genk",
    "krc genk": "Genk",

    "gent": "Gent",
    "kaa gent": "Gent",

    "royal antwerp": "Royal Antwerp",
    "royal antwerp fc": "Royal Antwerp",

    "union saint-gilloise": "Union Saint-Gilloise",
    "royale union saint-gilloise": "Union Saint-Gilloise",

    # Scotland
    "celtic": "Celtic",
    "celtic fc": "Celtic",

    "rangers": "Rangers",
    "rangers fc": "Rangers",

    # Austria
    "red bull salzburg": "RB Salzburg",
    "rb salzburg": "RB Salzburg",
    "fc red bull salzburg": "RB Salzburg",
    "salzburg": "RB Salzburg",

    "sturm graz": "Sturm Graz",
    "sk sturm graz": "Sturm Graz",

    # Switzerland
    "young boys": "Young Boys",
    "bsc young boys": "Young Boys",

    "basel": "Basel",
    "fc basel": "Basel",
    "fc basel 1893": "Basel",

    # Ukraine
    "shakhtar": "Shakhtar Donetsk",
    "shakhtar donetsk": "Shakhtar Donetsk",
    "fk shakhtar donetsk": "Shakhtar Donetsk",
    "fc shakhtar donetsk": "Shakhtar Donetsk",

    "dynamo kyiv": "Dynamo Kyiv",
    "dynamo kiev": "Dynamo Kyiv",
    "fc dynamo kyiv": "Dynamo Kyiv",

    # Russia
    "loko moscow": "Lokomotiv Moscow",
    "lokomotiv moscow": "Lokomotiv Moscow",
    "fc lokomotiv moscow": "Lokomotiv Moscow",

    # Czechia
    "plzen": "Viktoria Plzeň",
    "viktoria plzen": "Viktoria Plzeň",
    "fc viktoria plzen": "Viktoria Plzeň",

    "slavia prague": "Slavia Prague",
    "slavia praha": "Slavia Prague",
    "sk slavia praha": "Slavia Prague",

    "sparta prague": "Sparta Prague",
    "sparta praha": "Sparta Prague",
    "ac sparta praha": "Sparta Prague",

    # Croatia
    "dinamo zagreb": "Dinamo Zagreb",
    "gnk dinamo zagreb": "Dinamo Zagreb",

    # Serbia
    "red star": "Red Star Belgrade",
    "red star belgrade": "Red Star Belgrade",
    "crvena zvezda": "Red Star Belgrade",
    "fk crvena zvezda": "Red Star Belgrade",

    # Turkey
    "galatasaray": "Galatasaray",
    "galatasaray sk": "Galatasaray",

    "besiktas": "Beşiktaş",
    "besiktas jk": "Beşiktaş",

    "basaksehir": "İstanbul Başakşehir",
    "istanbul basaksehir": "İstanbul Başakşehir",
    "istanbul basaksehir fk": "İstanbul Başakşehir",

    "fenerbahce": "Fenerbahçe",
    "fenerbahce sk": "Fenerbahçe",

    # Greece
    "olympiacos": "Olympiacos",
    "olympiacos fc": "Olympiacos",
    "olympiakos": "Olympiacos",
    "olympiakos fc": "Olympiacos",
    "pae olympiakos sfp": "Olympiacos",

    # Denmark
    "copenhagen": "Copenhagen",
    "fc copenhagen": "Copenhagen",
    "fc kobenhavn": "Copenhagen",
    "fc københavn": "Copenhagen",

    "midtjylland": "Midtjylland",
    "fc midtjylland": "Midtjylland",

    # Norway
    "bodo/glimt": "Bodø/Glimt",
    "bodø/glimt": "Bodø/Glimt",
    "fk bodo/glimt": "Bodø/Glimt",
    "fk bodø/glimt": "Bodø/Glimt",

    # Israel
    "maccabi haifa": "Maccabi Haifa",
    "maccabi haifa fc": "Maccabi Haifa",

    "maccabi tel aviv": "Maccabi Tel Aviv",
    "maccabi tel-aviv": "Maccabi Tel Aviv",

    # Azerbaijan
    "qarabag": "Qarabağ",
    "qarabag fk": "Qarabağ",
    "qarabag agdam": "Qarabağ",
    "qarabag agdam fk": "Qarabağ",

    # Hungary
    "ferencvaros": "Ferencváros",
    "ferencvarosi tc": "Ferencváros",

    # Bulgaria
    "ludogorets": "Ludogorets",
    "pfc ludogorets razgrad": "Ludogorets",

    # Moldova
    "sheriff tiraspol": "Sheriff Tiraspol",
    "fc sheriff tiraspol": "Sheriff Tiraspol",

    # Kazakhstan
    "astana": "Astana",
    "fc astana": "Astana",

    # Cyprus
    "apoel": "APOEL",
    "apoel fc": "APOEL",

    # Slovakia
    "msk zilina": "MŠK Žilina",
    "zilina": "MŠK Žilina",

    "sk slovan bratislava": "ŠK Slovan Bratislava",
    "slovan bratislava": "ŠK Slovan Bratislava",

    # Romania
    "otelul galati": "Oțelul Galați",
    "fc otelul galati": "Oțelul Galați",
    "sc otelul galati": "Oțelul Galați",
}


# ============================================================
# MOJIBAKE DETECTION
# ============================================================


def contains_mojibake_markers(
    value: str,
) -> bool:
    """
    Detect common UTF-8/Latin-1 decoding damage.
    """

    if not isinstance(
        value,
        str,
    ):
        return False

    markers = (
        "Ã",
        "Â",
        "Å",
        "Ä",
        "È",
        "â",
        "ð",
    )

    if any(
        marker in value
        for marker in markers
    ):

        return True

    for character in value:

        codepoint = ord(
            character
        )

        if (
            0x80
            <= codepoint
            <= 0x9F
        ):

            return True

    return False


# ============================================================
# GENERIC UTF-8 REPAIR
# ============================================================


def repair_utf8_mojibake(
    value: str,
) -> str:
    """
    Repair UTF-8 bytes that were incorrectly decoded as
    Latin-1 or Windows-1252.
    """

    if not isinstance(
        value,
        str,
    ):

        return value

    current = value

    for _ in range(
        2
    ):

        if not contains_mojibake_markers(
            current
        ):

            break

        repaired = None

        for encoding in (
            "latin1",
            "cp1252",
        ):

            try:

                candidate = (
                    current
                    .encode(
                        encoding
                    )
                    .decode(
                        "utf-8"
                    )
                )

            except (
                UnicodeEncodeError,
                UnicodeDecodeError,
            ):

                continue

            repaired = candidate
            break

        if (
            repaired is None
            or repaired == current
        ):

            break

        current = repaired

    return current


# ============================================================
# ACCENT-FREE COMPARISON TEXT
# ============================================================


def alias_text(
    value: str,
) -> str:
    """
    Build lower-case, accent-free comparison text.
    """

    if not isinstance(
        value,
        str,
    ):

        return value

    normalized = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )

    return (
        value
        .casefold()
        .strip()
    )


# ============================================================
# KNOWN MOJIBAKE REPAIR
# ============================================================


def repair_known_mojibake(
    value: str,
) -> str:
    """
    Run generic decoding repair and historical fallbacks.
    """

    if not isinstance(
        value,
        str,
    ):

        return value

    value = value.replace(
        "\u00a0",
        " ",
    ).strip()

    value = repair_utf8_mojibake(
        value
    )

    if value in MOJIBAKE_TEAM_FIXES:

        return MOJIBAKE_TEAM_FIXES[
            value
        ]

    lowered = value.casefold()
    comparison = alias_text(
        value
    )

    # Žilina fallback
    if (
        comparison == "zilina"
        or comparison == "msk zilina"
    ):

        return "MŠK Žilina"

    # Oțelul Galați fallback
    if (
        lowered == "baå"
        or lowered.startswith(
            "baål"
        )
        or (
            "gala" in lowered
            and (
                "otel" in comparison
                or "oå" in lowered
                or "oè" in lowered
                or "baå" in lowered
            )
        )
    ):

        return "Oțelul Galați"

    # Qarabağ fallback
    if lowered.startswith(
        "qaraba"
    ):

        if (
            "ä" in lowered
            or "agdam" in comparison
            or comparison in {
                "qarabag",
                "qarabag fk",
            }
        ):

            return "Qarabağ"

    return value


# ============================================================
# BASIC TEAM TEXT CLEANING
# ============================================================


def clean_team_text(
    value: str,
) -> str:
    """
    Standardize typography before alias lookup.
    """

    if not isinstance(
        value,
        str,
    ):

        return value

    value = repair_known_mojibake(
        value
    )

    value = unicodedata.normalize(
        "NFC",
        value,
    )

    replacements = {
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():

        value = value.replace(
            old,
            new,
        )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# ALIAS KEY
# ============================================================


def alias_lookup_key(
    value: str,
) -> str:
    """
    Build the key used against TEAM_ALIASES.
    """

    value = clean_team_text(
        value
    )

    value = alias_text(
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# NORMALIZE SINGLE TEAM
# ============================================================


def normalize_team_name(
    team_name: str,
) -> str:
    """
    Convert provider-specific names into canonical club names.
    """

    if pd.isna(
        team_name
    ):

        return team_name

    clean_name = clean_team_text(
        str(
            team_name
        )
    )

    lookup = alias_lookup_key(
        clean_name
    )

    canonical = TEAM_ALIASES.get(
        lookup
    )

    if canonical is not None:

        return canonical

    return clean_name


# ============================================================
# NORMALIZE TEAM COLUMNS
# ============================================================


def normalize_team_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize home_team and away_team.
    """

    df = dataframe.copy()

    required = {
        "home_team",
        "away_team",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing team columns: "
            f"{sorted(missing)}"
        )

    df[
        "home_team"
    ] = df[
        "home_team"
    ].map(
        normalize_team_name
    )

    df[
        "away_team"
    ] = df[
        "away_team"
    ].map(
        normalize_team_name
    )

    return df


# ============================================================
# TEAM SET
# ============================================================


def get_team_names(
    dataframe: pd.DataFrame,
) -> set[str]:
    """
    Return all unique teams from home/away columns.
    """

    home = set(
        dataframe[
            "home_team"
        ].dropna()
    )

    away = set(
        dataframe[
            "away_team"
        ].dropna()
    )

    return home.union(
        away
    )


# ============================================================
# SINGLE DATASET AUDIT
# ============================================================


def audit_dataframe(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Display normalization changes.
    """

    original_names = sorted(
        get_team_names(
            dataframe
        )
    )

    mappings = []

    for original in original_names:

        canonical = normalize_team_name(
            original
        )

        if canonical != original:

            mappings.append(
                (
                    original,
                    canonical,
                )
            )

    normalized = normalize_team_columns(
        dataframe
    )

    canonical_names = sorted(
        get_team_names(
            normalized
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        name
    )

    print(
        "=" * 75
    )

    print(
        f"\nOriginal unique team names: "
        f"{len(original_names):,}"
    )

    print(
        f"Canonical unique team names: "
        f"{len(canonical_names):,}"
    )

    print(
        f"Names changed: "
        f"{len(mappings):,}"
    )

    if mappings:

        print(
            "\nNORMALIZED NAMES"
        )

        print(
            "-" * 75
        )

        for original, canonical in mappings:

            print(
                f"{original:<40} "
                f"-> "
                f"{canonical}"
            )


# ============================================================
# CROSS-SOURCE AUDIT
# ============================================================


def audit_cross_source_names(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Display source-specific spellings that collapse to the
    same canonical club.
    """

    canonical_sources = {}

    for (
        source_name,
        dataframe,
    ) in datasets.items():

        for team in get_team_names(
            dataframe
        ):

            canonical = normalize_team_name(
                team
            )

            canonical_sources.setdefault(
                canonical,
                [],
            ).append(
                (
                    source_name,
                    team,
                )
            )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CROSS-SOURCE TEAM IDENTITY AUDIT"
    )

    print(
        "=" * 75
    )

    multi_source_count = 0

    for canonical, entries in sorted(
        canonical_sources.items()
    ):

        providers = {
            source
            for source, _ in entries
        }

        if len(
            providers
        ) <= 1:

            continue

        multi_source_count += 1

        raw_names = {
            raw_name
            for _, raw_name in entries
        }

        if len(
            raw_names
        ) <= 1:

            continue

        print(
            f"\n{canonical}"
        )

        for (
            source,
            raw_name,
        ) in sorted(
            set(
                entries
            )
        ):

            print(
                f"  "
                f"{source:<34} "
                f"{raw_name}"
            )

    print(
        f"\nCanonical clubs appearing in "
        f"multiple sources: "
        f"{multi_source_count:,}"
    )


# ============================================================
# ENCODING AUDIT
# ============================================================


def audit_suspicious_encoding(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Detect remaining encoding corruption.
    """

    suspicious = []

    for (
        source_name,
        dataframe,
    ) in datasets.items():

        normalized = normalize_team_columns(
            dataframe
        )

        for team in get_team_names(
            normalized
        ):

            if contains_mojibake_markers(
                str(
                    team
                )
            ):

                suspicious.append(
                    (
                        source_name,
                        team,
                    )
                )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ENCODING AUDIT"
    )

    print(
        "=" * 75
    )

    if suspicious:

        print(
            "\nWARNING: suspicious encoding remains:"
        )

        for source, team in suspicious:

            print(
                f"{source:<35} "
                f"{team}"
            )

    else:

        print(
            "\nNo obvious mojibake remains "
            "after normalization."
        )


# ============================================================
# PROJECT UCL AUDIT
# ============================================================


def run_project_team_audit() -> None:
    """
    Audit the three UCL match datasets.
    """

    files = {
        "Kaggle 2010/11-2021/22": (
            PROCESSED_DATA_DIR
            / "matches_clean.csv"
        ),
        "API-Football 2022/23": (
            INTERIM_DATA_DIR
            / "matches"
            / "ucl_2022_2023.csv"
        ),
        "football-data.org 2023/24-2025/26": (
            INTERIM_DATA_DIR
            / "matches"
            / "ucl_recent_matches.csv"
        ),
    }

    datasets = {}

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TEAM NAME AUDIT"
    )

    print(
        "=" * 75
    )

    for (
        source_name,
        file_path,
    ) in files.items():

        print(
            f"\nChecking:\n"
            f"{file_path}"
        )

        if not file_path.exists():

            print(
                "STATUS: FILE NOT FOUND"
            )

            continue

        dataframe = pd.read_csv(
            file_path
        )

        if not {
            "home_team",
            "away_team",
        }.issubset(
            dataframe.columns
        ):

            print(
                "STATUS: team columns missing"
            )

            continue

        print(
            f"STATUS: FOUND "
            f"({len(dataframe):,} matches)"
        )

        datasets[
            source_name
        ] = dataframe

    if not datasets:

        raise RuntimeError(
            "No UCL datasets available."
        )

    for (
        source_name,
        dataframe,
    ) in datasets.items():

        audit_dataframe(
            source_name,
            dataframe,
        )

    if len(
        datasets
    ) > 1:

        audit_cross_source_names(
            datasets
        )

    audit_suspicious_encoding(
        datasets
    )

    canonical_teams = set()

    for dataframe in datasets.values():

        normalized = normalize_team_columns(
            dataframe
        )

        canonical_teams.update(
            get_team_names(
                normalized
            )
        )

    total_matches = sum(
        len(
            dataframe
        )
        for dataframe in datasets.values()
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nDatasets checked: "
        f"{len(datasets):,}"
    )

    print(
        f"Total matches available: "
        f"{total_matches:,}"
    )

    print(
        f"Total canonical clubs across all seasons: "
        f"{len(canonical_teams):,}"
    )

    print(
        "\nNo files have been modified."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_project_team_audit()