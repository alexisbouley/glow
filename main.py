import argparse
import os
from train import train

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--dataroot", type=str, default="CelebA/", help="path to dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_false", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=2, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size used during training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train for"
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    args = parser.parse_args()

    os.makedirs(args.output_dir)


    kwargs = vars(args)

    train(**kwargs)