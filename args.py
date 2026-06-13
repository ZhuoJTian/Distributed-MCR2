import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default=None, help="configs file",
    )

    # Model
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in the model",
    )

    parser.add_argument(
        "--dim-z", type=int, default=128, help="the dimension of the feature"
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="MCR", # CE, MCR
        help="type of the loss function",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10", # mnist, TMNIST
        help="Dataset for training and eval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=3000,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--data-dir", type=str, default='./Datasets_Sim/', help="path to datasets"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=1000, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5, help='weight decay'
    )

    # Additional
    parser.add_argument("--seed", type=int, default=1234, help="random seed")

    # the distributed parameters
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10
    )

    parser.add_argument(
        "--num_localeps",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--path-result", type=str, default="./Results4/ColMCR3/", help="path to result"
    )

    return parser.parse_args()
