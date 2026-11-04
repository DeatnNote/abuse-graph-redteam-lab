import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def world():
    from agl.detector import Detector, split_groups
    from agl.sim import generate
    df = generate(5)
    tr = split_groups(df, seed=5)
    return df, tr, Detector().fit(df, tr)
