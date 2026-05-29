import importlib.util

import dace


def test_package_exposes_version():
    assert isinstance(dace.__version__, str)
    assert dace.__version__


def test_subpackages_are_discoverable():
    assert importlib.util.find_spec("dace.characteristics") is not None
    assert importlib.util.find_spec("dace.display") is not None
    assert importlib.util.find_spec("dace.utils") is not None
