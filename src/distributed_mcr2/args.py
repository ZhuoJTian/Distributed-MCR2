"""Shared command-line arguments for Distributed MCR² experiments."""
import argparse


def parse_args(
    *,
    default_num_clients: int = 10,
    default_result_path: str = "Exp_results/IID/CIFAR10/ColMCR",
    default_adjacency_file: str = "10_adj_matrix.txt",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed MCR² training")

    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dim-z", type=int, default=128)
    parser.add_argument("--loss-type", choices=("MCR", "CE"), default="MCR")

    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=3000)
    parser.add_argument("--data-dir", default="Datasets_Sim")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-local-epochs", type=int, default=1)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-clients", type=int, default=default_num_clients)
    parser.add_argument("--path-result", default=default_result_path)
    parser.add_argument("--adjacency-file", default=default_adjacency_file)
    parser.add_argument("--save-every", type=int, default=100)

    return parser.parse_args()
