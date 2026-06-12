from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_table3_wikitext_all_position_footer_uses_big_split_numbers():
    text = (ROOT / "paper" / "main.tex").read_text()

    assert "all-position} KL (WikiText-test): $0.06969$ vs $0.07899$" in text
    assert "$-11.8\\%$; top-1 $+0.57$pp" in text
    assert "all-position} KL (WikiText-test): $0.03575$ vs $0.04094$" not in text


def test_27b_corpus_count_does_not_call_wikitext_window_third_disjoint_corpus():
    text = (ROOT / "paper" / "main.tex").read_text()

    assert "three disjoint held-out corpora" not in text
    assert "two disjoint held-out corpora" in text
    assert "second WikiText-test" in text
    assert "measurement window" in text


def test_fp8_knee_claim_discloses_resident_train_split_screen():
    text = (ROOT / "paper" / "main.tex").read_text()

    assert "$6\\times$ lower KL at\nmatched bytes in a resident RTN weight-only train-split screen" in text
    assert "not a served\nheld-out result" in text


def test_claude_paper_map_points_to_aura_and_archive():
    text = (ROOT / "CLAUDE.md").read_text()

    assert "paper/figures/fig_aura_rd_geometry.tex" in text
    assert "paper/archive/prismascout_paper_2026-06-05.tex" in text
    assert "paper/figures/fig_validated_27b.tex" not in text
    assert "paper's \"Methods Considered and Rejected\" section exists" not in text


def test_active_paper_figures_exclude_archived_prismascout_assets():
    active_figures = sorted(
        path.name for path in (ROOT / "paper" / "figures").glob("*.tex")
    )

    assert active_figures == ["fig_aura_rd_geometry.tex"]
    assert not (ROOT / "paper" / "prismaquant_animation.html").exists()
    assert not (ROOT / "paper" / "prismaquant_prismascout_paper.pdf").exists()
    assert (
        ROOT / "paper" / "archive" / "figures" / "fig_validated_27b.tex"
    ).is_file()
    assert (
        ROOT / "paper" / "archive" / "prismaquant_animation_2026-06-05.html"
    ).is_file()
    assert (
        ROOT
        / "paper"
        / "archive"
        / "prismaquant_prismascout_paper_legacy_2026-05-04.pdf"
    ).is_file()
