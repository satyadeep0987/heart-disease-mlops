from src.config import PROJECT_ROOT, RANDOM_SEED


def test_project_root_exists():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_seed_is_reproducible():
    assert isinstance(RANDOM_SEED, int)
