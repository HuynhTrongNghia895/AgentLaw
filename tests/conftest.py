import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    "src/utils",
    "src",
    "src/retrieval",
):
    sys.path.insert(0, str(PROJECT_ROOT / _p))

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Chạy test đánh dấu @pytest.mark.slow (tải model BGE thật).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tải model BGE thật (chạy với --runslow)"
    )


def pytest_collection_modifyitems(config, items):
    """Không có --runslow thì skip các test @pytest.mark.slow."""
    if not config.getoption("--runslow"):
        skip = pytest.mark.skip(reason="chạy --runslow để bật")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def relation_index(project_root):
    from DocxIngestionPipeline import DocxIngestionPipeline

    pipeline = DocxIngestionPipeline(project_root)
    return pipeline.load_relation_index()
