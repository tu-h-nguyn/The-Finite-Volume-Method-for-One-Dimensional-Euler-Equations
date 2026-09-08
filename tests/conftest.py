import pytest

from euler1d import get_problem


@pytest.fixture
def sod():
    return get_problem("sod")


@pytest.fixture
def smooth():
    return get_problem("smooth")
