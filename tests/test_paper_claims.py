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
