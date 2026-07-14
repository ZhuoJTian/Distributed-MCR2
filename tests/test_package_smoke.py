from distributed_mcr2 import __version__
from distributed_mcr2.paths import resolve_input_path


def test_version():
    assert __version__ == "0.3.0"


def test_packaged_adjacency_exists():
    assert resolve_input_path("10_adj_matrix.txt").is_file()
    assert resolve_input_path("4_adj_matrix.txt").is_file()
