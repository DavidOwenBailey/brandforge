"""The fixed eval set: 30 cases, 10 briefs per brand, validated without a model."""

from collections import Counter
from pathlib import Path

import pytest

import brandforge
from brandforge.brands import list_brand_ids, load_brand
from brandforge.evals import CaseLoadError, list_case_ids, load_case, load_cases
from brandforge.interfaces.cli import load_brief

BRIEFS_DIR = Path(brandforge.__file__).parent / "briefs"
CHANNELS = {"search", "social", "display", "email"}
OBJECTIVES = {"awareness", "consideration", "conversion"}


def _write(directory: Path, name: str, text: str) -> None:
    directory.joinpath(name).write_text(text, encoding="utf-8")


def test_thirty_cases_ten_per_brand() -> None:
    cases = load_cases()
    counts = Counter(case.brand_id for case in cases)
    assert list_case_ids() == sorted(case.id for case in cases)
    assert len(cases) == 30
    assert set(counts) == set(list_brand_ids())
    assert all(count == 10 for count in counts.values())


@pytest.mark.parametrize("case_id", list_case_ids())
def test_case_id_matches_the_file_and_the_brand_prefix(case_id: str) -> None:
    case = load_case(case_id)
    assert case.id == case_id
    assert case.id.startswith(f"{case.brand_id}_")
    assert case.brand_id in list_brand_ids()


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_brand_cases_cover_every_channel_objective_and_shape(brand_id: str) -> None:
    cases = [case for case in load_cases() if case.brand_id == brand_id]
    assert {channel for case in cases for channel in case.brief.channels} == CHANNELS
    assert {case.brief.objective for case in cases} == OBJECTIVES
    assert any(len(case.brief.channels) == 1 for case in cases)
    assert any(set(case.brief.channels) == CHANNELS for case in cases)
    products = [case.brief.product for case in cases]
    assert len(set(products)) == len(products)


def test_sample_briefs_match_the_cases_with_the_same_name() -> None:
    cases = {case.id: case for case in load_cases()}
    sample_paths = sorted(BRIEFS_DIR.glob("*.yaml"))
    assert len(sample_paths) == 10
    for path in sample_paths:
        assert cases[path.stem].brief == load_brief(path)
        assert cases[path.stem].brand_id == path.stem.split("_")[0]


def test_email_caps_follow_the_brief_not_one_global_number() -> None:
    """Brightleaf and Voltride cap email subjects at 50. Ledgerly says 60."""
    caps = {
        case.id: case.hard_constraints.headline_max_chars["email"]
        for case in load_cases()
        if "email" in case.hard_constraints.headline_max_chars
    }
    assert caps["brightleaf_01_spring_blossom"] == 50
    assert caps["voltride_03_fold_commuter"] == 50
    assert caps["ledgerly_01_vat_reminders"] == 60


def test_social_headlines_are_not_capped_in_this_set() -> None:
    capped = {
        channel for case in load_cases() for channel in case.hard_constraints.headline_max_chars
    }
    assert {"search", "display", "email"} <= capped
    assert "social" not in capped


def test_an_invitation_records_the_phrase_it_asks_for() -> None:
    for case in load_cases():
        invites = [
            line for line in case.brief.constraints if line.lower().startswith("invite readers")
        ]
        if not invites:
            continue
        assert case.hard_constraints.must_mention
        assert any(
            phrase.casefold() in invites[0].casefold()
            for phrase in case.hard_constraints.must_mention
        )


@pytest.mark.parametrize("brand_id", list_brand_ids())
def test_briefs_do_not_use_the_brand_banned_words(brand_id: str) -> None:
    brand = load_brand(brand_id)
    for case in load_cases():
        if case.brand_id != brand_id:
            continue
        text = "\n".join(
            [case.brief.product, case.brief.audience, *case.brief.constraints]
        ).casefold()
        for banned in brand.banned_words:
            assert banned.casefold() not in text, f"{case.id} uses banned word {banned!r}"


def test_unknown_case_raises() -> None:
    with pytest.raises(CaseLoadError, match="Unknown case"):
        load_case("nope")


def test_path_traversal_is_rejected() -> None:
    with pytest.raises(CaseLoadError):
        load_case("../brightleaf_01_spring_blossom")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "id: [unclosed")
    with pytest.raises(CaseLoadError, match="invalid YAML"):
        load_case("bad", cases_dir=tmp_path)


def test_a_non_mapping_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "acme_01.yaml", "- just a list\n")
    with pytest.raises(CaseLoadError):
        load_case("acme_01", cases_dir=tmp_path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
note: extra
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
""",
    )
    with pytest.raises(CaseLoadError):
        load_case("acme_01", cases_dir=tmp_path)


def test_unknown_hard_constraint_field_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
hard_constraints:
  banned_words: [miracle]
""",
    )
    with pytest.raises(CaseLoadError):
        load_case("acme_01", cases_dir=tmp_path)


def test_id_must_match_filename(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "other.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
""",
    )
    with pytest.raises(CaseLoadError, match="must match the filename"):
        load_case("other", cases_dir=tmp_path)


def test_filename_must_start_with_the_brand_id(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "brightleaf_01.yaml",
        """
id: brightleaf_01
brand_id: ledgerly
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
""",
    )
    with pytest.raises(CaseLoadError, match="must start with the brand id"):
        load_case("brightleaf_01", cases_dir=tmp_path)


def test_a_cap_for_an_unrequested_channel_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
  constraints:
    - email subject line max 50 characters
hard_constraints:
  headline_max_chars:
    email: 50
""",
    )
    with pytest.raises(CaseLoadError, match="does not request"):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_prose_cap_without_the_map_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [email]
  constraints:
    - email subject line max 50 characters
""",
    )
    with pytest.raises(CaseLoadError, match="headline_max_chars must equal"):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_map_cap_without_the_prose_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [email]
hard_constraints:
  headline_max_chars:
    email: 50
""",
    )
    with pytest.raises(CaseLoadError, match="headline_max_chars must equal"):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_mismatched_cap_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [email]
  constraints:
    - email subject line max 50 characters
hard_constraints:
  headline_max_chars:
    email: 60
""",
    )
    with pytest.raises(CaseLoadError, match="headline_max_chars must equal"):
        load_case("acme_01", cases_dir=tmp_path)


def test_disagreeing_prose_caps_are_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [email]
  constraints:
    - email subject line max 50 characters
    - email subject line max 60 characters
hard_constraints:
  headline_max_chars:
    email: 50
""",
    )
    with pytest.raises(CaseLoadError, match="disagree"):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_reversed_cap_sentence_does_not_count(tmp_path: Path) -> None:
    """The channel word has to come before max N characters, as the sample briefs write it."""
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [search]
  constraints:
    - max 30 characters in search headlines
hard_constraints:
  headline_max_chars:
    search: 30
""",
    )
    with pytest.raises(CaseLoadError, match="headline_max_chars must equal"):
        load_case("acme_01", cases_dir=tmp_path)


def test_two_caps_in_one_constraint_both_count(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [search, email]
  constraints:
    - Email headlines max 50 characters and search headlines max 30 characters
hard_constraints:
  headline_max_chars:
    search: 30
    email: 50
""",
    )
    case = load_case("acme_01", cases_dir=tmp_path)
    assert case.hard_constraints.headline_max_chars == {"search": 30, "email": 50}


def test_must_mention_must_be_stated_in_the_constraints(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [social]
hard_constraints:
  must_mention:
    - open evening
""",
    )
    with pytest.raises(CaseLoadError, match="must_mention"):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_zero_headline_cap_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [email]
  constraints:
    - email subject line max 0 characters
hard_constraints:
  headline_max_chars:
    email: 0
""",
    )
    with pytest.raises(CaseLoadError):
        load_case("acme_01", cases_dir=tmp_path)


def test_a_bad_channel_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "acme_01.yaml",
        """
id: acme_01
brand_id: acme
brief:
  product: Widget
  audience: People who need one
  objective: awareness
  channels: [billboard]
""",
    )
    with pytest.raises(CaseLoadError):
        load_case("acme_01", cases_dir=tmp_path)
