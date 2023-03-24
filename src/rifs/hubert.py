"""

"""

import subprocess
import os


def hubert_preprocess_1st(ctx, dataset: str):
    """
    python preprocess.py --root-dir /home/datasets --feat-type mfcc --exp-dir ./exp --num-cluster 100
    """
    folder = os.path.dirname(__file__)

    if ctx["verbose"]:
        print("HUBERT preprocessing started...")
    result = subprocess.Popen(
        f"python {folder}/hubert_preprocess.py --root-dir {ctx['data_path']}/raw/{dataset}/audio_segmented --feat-type mfcc --exp-dir {ctx['data_path']}/fairseq/hubert --num-cluster 100",
        shell=True,
    ).wait()

    if result != 0:
        print(f"HUBERT preprocessing failed with exit code {result}!")
        exit(result)
    elif ctx["verbose"]:
        print("HUBERT preprocessing completed!")


def hubert_preprocess_2nd(ctx, dataset: str):
    pass
    """
    python preprocess.py --root-dir /home/datasets --feat-type hubert --exp-dir
    ./exp --layer-index 6 --checkpoint-path ./exp_iter1/ --num-rank 40
    checkpoints_librispeech_hubert_pretrain_base/xxx.ckpt --num-cluster 500
    --percent 0.1
    """
