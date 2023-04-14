"""
Preprocessing steps for the Hubert model as outlined in the fairseq Hubert example
"""

import subprocess
import os
import torch
import torchaudio
from pathlib import Path
from typing import Union


def hubert_preprocess_1st(ctx, fairseq_path: str, dataset: str) -> None:
    """
    python preprocess.py --root-dir /home/datasets --feat-type mfcc --exp-dir ./exp --num-cluster 100

    Parameters
    ----------
    ctx : dict
        The context dictionary.
    fairseq_path : str
        The path to the fairseq repository.
    dataset : str
        The name of the dataset to preprocess.

    Returns
    -------
    None
    """
    if ctx["verbose"]:
        print("HUBERT preprocessing started...")

    fairseq_hubert_preprocess(ctx, fairseq_path, dataset)


def hubert_preprocess_2nd(ctx, dataset: str) -> None:
    """
    To be implemented.
    """
    pass
    """
    python preprocess.py --root-dir /home/datasets --feat-type hubert --exp-dir
    ./exp --layer-index 6 --checkpoint-path ./exp_iter1/ --num-rank 40
    checkpoints_librispeech_hubert_pretrain_base/xxx.ckpt --num-cluster 500
    --percent 0.1
    """


def fairseq_hubert_preprocess(ctx, fairseq_path: str, dataset: str) -> None:
    """
    Run the preprocessing steps as outlined in the fairseq Hubert example:
    https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans

    Parameters
    ----------
    ctx : dict
        The context dictionary.
    fairseq_path : str
        The path to the fairseq repository.
    dataset : str
        The name of the dataset to preprocess.

    Returns
    -------
    None
    """

    tsv_dir = f"{ctx['data_path']}/fairseq/hubert/data/mfcc/tsv"
    feat_dir = f"{ctx['data_path']}/fairseq/hubert/data/mfcc/feat"
    km_path = f"{ctx['data_path']}/fairseq/hubert/data/mfcc/km_model"
    lab_dir = f"{ctx['data_path']}/fairseq/hubert/data/mfcc/label"

    os.makedirs(tsv_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(km_path, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    n_shard = 5
    n_cluster = 100

    create_tsv(
        f"{ctx['data_path']}/raw/{dataset}/alignments", tsv_dir, seed=ctx["seed"]
    )

    for split in ["train", "valid"]:
        for rank in range(n_shard):

            # Extract MFCC features
            result = subprocess.Popen(
                f"python {fairseq_path}/examples/hubert/simple_kmeans/dump_mfcc_feature.py {tsv_dir} {split} {n_shard} {rank} {feat_dir}",
                shell=True,
            ).wait()

            if result != 0:
                print(f"HUBERT preprocessing failed with exit code {result}!")
                exit(result)

            """# Extract Hubert features
            result = subprocess.Popen(
                f"python {fairseq_path}/examples/hubert/simple_kmeans/dump_hubert_feature.py {tsv_dir} {split} {ckpt_path} {layer} {n_shard} {rank} {feat_dir}",
                shell=True,
            ).wait()"""

        # Learn K-means
        km_model_path = f"{km_path}/{split}.pt"
        result = subprocess.Popen(
            f"python {fairseq_path}/examples/hubert/simple_kmeans/learn_kmeans.py {feat_dir} {split} {n_shard} {km_model_path} {n_cluster} --percent 0.1",
            shell=True,
        ).wait()

        if result != 0:
            print(f"HUBERT preprocessing failed with exit code {result}!")
            exit(result)

        # K-means applications
        for rank in range(n_shard):
            result = subprocess.Popen(
                f"python {fairseq_path}/examples/hubert/simple_kmeans/dump_km_label.py {feat_dir} {split} {km_model_path} {n_shard} {rank} {lab_dir}",
                shell=True,
            ).wait()

            if result != 0:
                print(f"HUBERT preprocessing failed with exit code {result}!")
                exit(result)

        with open(os.path.join(lab_dir, f"{split}.pt"), "w+") as f:
            for rank in range(n_shard):
                f.write(f"{feat_dir}/{split}_{rank}_{n_shard}.pt\n")

    # Create a dummy dict
    with open(os.path.join(lab_dir, "dict.pt.txt"), "w+") as f:
        for x in range(n_cluster):
            f.write(f"{x} 1\n")


def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    valid_percent: float = 0.01,
    seed: int = 0,
    extension: str = "wav",
) -> None:
    """
    Create file lists for training and validation.

    Parameters
    ----------
    root_dir : str or Path
        The directory of the dataset.
    out_dir : str or Path
        The directory to store the file lists.
    valid_percent : float, optional
        The percentage of data for validation. (Default: 0.01)
    seed : int
        The seed for randomly selecting the validation files.
    extension : str, optional
        The extension of audio files. (Default: ``flac``)

    Returns
    -------
    None
    """
    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(seed)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = open(out_dir / "valid.tsv", "w") if valid_percent > 0 else None
    # search_pattern = ".*train.*"
    with open(out_dir / "train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        for fname in root_dir.glob(f"**/*.{extension}"):
            # if re.match(search_pattern, str(fname)):
            frames = torchaudio.info(fname).num_frames
            dest = train_f if torch.rand(1) > valid_percent else valid_f
            print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
    if valid_f is not None:
        valid_f.close()
