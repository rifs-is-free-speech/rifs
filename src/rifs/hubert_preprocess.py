#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation

Source: https://github.com/pytorch/audio/blob/main/examples/hubert/preprocess.py

Usage preprocess 1st iteration:

python preprocess.py --root-dir /home/datasets --feat-type mfcc --exp-dir
./exp --num-cluster 100

Usage preprocess 2nd iteration:

python preprocess.py --root-dir /home/datasets --feat-type hubert --exp-dir
./exp --layer-index 6 --checkpoint-path ./exp_iter1/ --num-rank 40
checkpoints_librispeech_hubert_pretrain_base/xxx.ckpt --num-cluster 500
--percent 0.1
"""
import logging
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import torch
from rifs.hubert_utils.common_utils import create_tsv
from rifs.hubert_utils.feature_utils import dump_features
from rifs.hubert_utils.kmeans import get_km_label, learn_kmeans


def _init_logger(debug=False):
    """Initialize logger."""
    message_fmt = (
        "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    )
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    """Parse command-line arguments"""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument(
        "--dataset",
        default="hubert",
        type=str,
        choices=["librispeech", "librilight", "hubert"],
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num-rank", default=5, type=int)
    parser.add_argument(
        "--feat-type", default="mfcc", choices=["mfcc", "hubert"], type=str
    )
    parser.add_argument(
        "--layer-index",
        default=6,
        type=int,
        help="The layer index in HuBERT model for feature extraction. (``1`` means the first layer output)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument("--use-gpu", default=False, type=bool)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num-cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="The percent of data for KMeans clustering. If negative, use all data. (Default: -1)",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Main function"""
    _init_logger(args.debug)

    if not args.exp_dir.exists():
        os.makedirs(args.exp_dir, exist_ok=True)
    if args.feat_type == "mfcc":
        data_dir = args.exp_dir / "data" / "mfcc"
    else:
        data_dir = args.exp_dir / "data" / f"{args.feat_type}_{args.layer_index}"
    data_dir.mkdir(parents=True, exist_ok=True)

    tsv_dir = data_dir / "tsv"
    feat_dir = data_dir / "feat"
    km_dir = data_dir / "km_model"
    label_dir = data_dir / "label"

    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create file lists for training and validation (optional)
    create_tsv(args.root_dir, tsv_dir)

    # Extract features for KMeans clustering
    if not feat_dir.exists():
        feat_dir.mkdir()

    """num_labels = {
        "train": sum(1 for _ in open(tsv_dir / "train.tsv")) -1,
        "valid": sum(1 for _ in open(tsv_dir / "valid.tsv")) - 1
    }"""

    num_labels = {
        "train": 5,
        "valid": 5
    }

    for split in ["train", "valid"]:
        for rank in range(1, num_labels[split] + 1):
            dump_features(
                tsv_file=os.path.join(tsv_dir, f"{split}.tsv"),
                out_dir=feat_dir,
                split=split,
                rank=rank,
                num_rank=num_labels[split],
                device=device,
                feature_type=args.feat_type,
                layer_index=args.layer_index,
                checkpoint_path=args.checkpoint_path,
                sample_rate=16_000,
            )

        # Fit KMeans clustering model
        learn_kmeans(
            feat_dir=feat_dir,
            split=split,
            num_rank=num_labels[split],
            km_dir=km_dir,
            n_clusters=args.num_cluster,
            percent=args.percent, # TODO: Set to 0.1 to not load entire dataset into memory
        )

        # Predict labels for MFCC or HuBERT features
        get_km_label(
            feat_dir=feat_dir,
            km_dir=km_dir,
            label_dir=label_dir,
            split=split,
            num_rank=num_labels[split],
            device=device,
        )

        with open(os.path.join(label_dir, f"{split}.pt"), 'w+') as f:
            for rank in range(num_labels[split]):
                f.write(f"{feat_dir}/{split}_{rank}_{num_labels[split]}.pt\n")

    # Create a dummy dict
    with open(os.path.join(label_dir, "dict.pt.txt"), "w+") as f:
        for x in range(args.num_cluster):
            f.write(f"{x} 1\n")


if __name__ == "__main__":
    main(_parse_args())
