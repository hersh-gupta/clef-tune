import clef_tune


def test_version_is_a_non_empty_string():
    assert isinstance(clef_tune.__version__, str)
    assert clef_tune.__version__
