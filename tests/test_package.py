import importlib.metadata

import aulax as alx


def test_version_matches_metadata():
    assert alx.__version__ == importlib.metadata.version("aulax")
