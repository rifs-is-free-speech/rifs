""" Interface for fairseq pre-training.
Contains the commands found fairseq/examples/...
"""
import subprocess

import os
import yaml
import collections
from os.path import join


def run_fairseq_pretrain(
    fairseq_path: str, model_dict: dict, ctx: dict, manifest_source: str = None
):
    """Run fairseq pre-training.

    Parameters:
    -----------
    fairseq_path: str
        Path to fairseq directory.

    model_dict: dict
        Dictionary of arguments to pass to fairseq pre-training.

    ctx: dict
        Dictionary of rifs context arguments.

    manifest_source: str
        Path to source manifest folder.


    Returns:
    --------
    None
    """

    if ctx["verbose"]:
        print("Checking fairseq config...")

    if not check_fairseq_config(fairseq_path, model_dict):
        # return
        pass

    if ctx["verbose"]:
        print("Running fairseq pre-training...")

    command = fairseq_constructor(fairseq_path, model_dict, ctx, manifest_source)
    if ctx["verbose"]:
        print("Command: ", command)
    wd = os.getcwd()
    if ctx["verbose"]:
        print("Changing directory to fairseq path.")
    os.chdir(fairseq_path)
    # subprocess.Popen(f"which python", shell=True).wait()
    subprocess.Popen(f"python {command}", shell=True).wait()
    if ctx["verbose"]:
        print("Fairseq pre-training ended")
        print("Changing directory back to original working directory.")
    os.chdir(wd)


def fairseq_constructor(
    fairseq_path: str, model_dict: dict, ctx: dict, manifest_source=None
) -> str:
    """Creates the fairseq pre-training command.

    Parameters:
    -----------
    fairseq_path: str
        Path to fairseq directory.

    model_dict: dict
        Dictionary of arguments to pass to fairseq pre-training.

    ctx: dict
        Dictionary of rifs context arguments.

    manifest_source: str
        Path to source manifest folder.

    Returns:
    --------
    str
    """
    k = 1  # Number of GPUs

    # TODO: What about iteration 2?
    label_path = join(ctx["data_path"], "fairseq", "hubert", "data", "mfcc", "label")

    command_path = join(fairseq_path, model_dict["command"])
    command = f"{command_path} "
    if model_dict["pos_arg"]:
        if model_dict["pos_arg"] == "DIR":
            if manifest_source:
                command += f"{manifest_source} "
            else:
                raise Exception(
                    "No source manifest provided. This is required for fairseq manifest creation."
                )
        else:
            command += f"{model_dict['pos_arg']} "
    else:
        command += "-m "
    end_command = ""
    if model_dict["--config-dir"]:
        config_dir = join(fairseq_path, model_dict["--config-dir"])
        config_name = model_dict["--config-name"]
        end_command += f"--config-dir {config_dir} --config-name {config_name} "
    for required_args in model_dict["required-args"]:
        if required_args == "task.data":
            if model_dict["required-args"][required_args] == "WAV2VEC2":
                command += (
                    f"task.data={join(ctx['data_path'],'fairseq','wav2vec_manifest')} "
                )
            elif model_dict["required-args"][required_args] == "HUBERT":
                command += f"task.data={join(ctx['data_path'],'fairseq','hubert','data','mfcc','tsv')} "
            else:
                command += f"task.data={join(ctx['data_path'],'fairseq')} "
        elif required_args == "distributed_training.distributed_world_size":
            command += f"distributed_training.distributed_world_size={k} "
        elif required_args == "task.label_dir":
            command += f"task.label_dir={label_path} "
        elif required_args == "--dest":
            end_command += (
                f"--dest {join(ctx['data_path'],'fairseq','wav2vec_manifest')} "
            )

        elif required_args[:2] == "--":
            if model_dict["required_args"][required_args]:
                end_command += (
                    f"{required_args} {model_dict['required-args'][required_args]} "
                )
            else:
                end_command += f"{required_args} "
        else:
            if model_dict["required-args"][required_args]:
                command += (
                    f"{required_args}={model_dict['required-args'][required_args]} "
                )
            else:
                command += f"{required_args} "

    for extra_args in model_dict["extra_args"]:
        if extra_args == "optimization.update_freq='[x]'":
            x = model_dict["x/k"]
            command += f"optimization.update_freq='[{x//k}]' "
        elif extra_args == "common.user_dir":
            user_dir = join(fairseq_path, model_dict["extra_args"]["common.user_dir"])
            command += f"common.user_dir={user_dir} "
        elif extra_args[:2] == "--":
            if model_dict["extra_args"][extra_args]:
                end_command += f"{extra_args} {model_dict['extra_args'][extra_args]} "
            else:
                command += f"{extra_args} "
        else:
            if model_dict["extra_args"][extra_args]:
                command += f"{extra_args} {model_dict['extra_args'][extra_args]} "
            else:
                command += f"{extra_args} "
    command += end_command
    return command


def check_fairseq_config(fairseq_path: str, model_dict: dict):
    """
    Checks for the existence of the fairseq config directory.
    """

    if not model_dict["--config-dir"]:
        return True

    config_file = join(
        fairseq_path, model_dict["--config-dir"], f"{model_dict['--config-name']}.yaml"
    )

    success = True
    try:
        with open(config_file, "r") as stream:
            d = yaml.safe_load(stream)

            for k, v in flatten(d).items():
                if v == "???":
                    if (
                        k in model_dict["required-args"].keys()
                        or k in model_dict["extra_args"].keys()
                    ):
                        continue
                    print(f"Please set {k} in the config file at {config_file}.")
                    success = False

    except FileNotFoundError as e:
        print(e)
        return False
    except yaml.YAMLError as e:
        print(e)
        return False

    return success


def flatten(d, parent_key="", sep="."):
    """
    Flattens a dictionary.

    Parameters:
    -----------
    d: dict
        Dictionary to flatten.
    parent_key: str
        Parent key.
    sep: str
        Separator.

    Returns:
    --------
    dict

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


all_models = {
    "manifest_wav2vec2": {
        "help_text": "Prepare manifest files for wav2vec 2.0 pre-training.",
        "command": "examples/wav2vec/wav2vec_manifest.py",
        "--config-name": None,
        "--config-dir": None,
        "required-args": {
            "--dest": None,
        },
        "extra_args": {
            "--valid-percent": 0.01,
            "--ext": "wav",
        },
        "pos_arg": "DIR",
    },
    "wav2vec2_base": {
        "help_text": "wav2vec2.0 base model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "wav2vec2_base_librispeech",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": "WAV2VEC2",
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
        },
        "x/k": 64,
        "pos_arg": None,
    },
    "wav2vec2_large": {
        "help_text": "wav2vec2.0 large model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "wav2vec2_large_librivox",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": "WAV2VEC2",
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
        },
        "x/k": 128,
        "pos_arg": None,
    },
    "wav2vec2_conformer_base": {
        "help_text": "wav2vec2.0 conformer base model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "wav2vec2_conformer_base_librispeech",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": "WAV2VEC2",
        },
        "extra_args": {
            "--attn-type": "espnet",
            "--pos-enc-type": "rel_pos",
        },
        "pos_arg": None,
    },
    "wav2vec2_conformer_large": {
        "help_text": "wav2vec2.0 conformer large model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "wav2vec2_conformer_large_librivox",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": "WAV2VEC2",
        },
        "extra_args": {
            "--attn-type": "espnet",
            "--pos-enc-type": "rel_pos",
        },
        "pos_arg": None,
    },
    "hubert_base": {
        "help_text": "hubert base model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "hubert_base_librispeech",
        "--config-dir": "examples/hubert/config/pretrain",
        "required-args": {
            "task.data": "HUBERT",
            "task.label_dir": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            """task.labels='["km"]'""": None,
            "model.label_rate=100": None,
        },
        "pos_arg": None,
    },
    "data2vec_base": {
        "help_text": "data2vec base model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-name": "base_librispeech",
        "--config-dir": "examples/data2vec/config/audio/pretraining",
        "required-args": {
            "task.data": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
            "common.user_dir": "examples/data2vec",
        },
        "x/k": 64,
        "pos_arg": None,
    },
    "data2vec2_base": {
        "help_text": "data2vec 2.0 base model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-dir": "examples/data2vec/config/v2",
        "--config-name": "base_audio_only_task",
        "required-args": {
            "task.data": None,
        },
        "extra_args": {},
        "pos_arg": None,
    },
    "data2vec2_large": {
        "help_text": "data2vec 2.0 large model from fairseq",
        "command": "fairseq_cli/hydra_train.py",
        "--config-dir": "examples/data2vec/config/v2",
        "--config-name": "large_audio_only_task",
        "required-args": {
            "task.data": None,
        },
        "extra_args": {},
        "pos_arg": None,
    },
    "speechT5_base": {
        "help_text": "speechT5 base model from fairseq. Not yet implemented",
        "command": "fairseq_cli/train.py",
        "--config-name": None,
        "--config-dir": None,
        "required_args": {
            "--save-dir": None,
            "--tensorboard-logdir": None,
            "--train-subset": None,
            "--valid-subset": None,
            "--hubert-label-dir": None,
        },
        "extra_args": {
            "--distributed-world-size": 32,
            "--distributed-port": 0,
            "--ddp-backend": "legacy_ddp",
            "--user-dir": "SpeechT5/speecht5",
            "--log-format": "json",
            "--seed": 1337,
            "--fp16": None,
            "--task": "speecht5",
            "--t5-task": "pretrain",
            "--label-rates": 50,
            "--sample-rate": 16000,
            "--random-crop": None,
            "--num-workers": 0,
            "--max-tokens": 1400000,
            "--max-speech-sample-size": 250000,
            "--update-freq": 2,
            "--batch-ratio": "[1,0.0086]",
            "--criterion": "speecht5",
            "--optimizer": "adam",
            "--reset-optimizer": None,
            "--adam-betas": "(0.9, 0.98)",
            "--adam-eps": 1e-06,
            "--weight-decay": 0.01,
            "--power": 1,
            "--clip-norm": 5.0,
            "--lr": 0.0002,
            "--lr-scheduler": "polynomial_decay",
            "--max-update": 800000,
            "--warmup-updates": 64000,
            "--total-num-update": 800000,
            "--save-interval-updates": 3000,
            "--skip-invalid-size-inputs-valid-test": None,
            "--required-batch-size-multiple": 1,
            "--arch": "t5_transformer_base",
            "--share-input-output-embed": None,
            "--find-unused-parameters": None,
            "--bert-init": None,
            "--relative-position-embedding": None,
            "--use-codebook": None,
            "--codebook-prob": 0.1,
            "--loss-weights=[10,0.1]": None,
            "--max-text-positions": 600,
        },
        "pos_arg": None,
    },
}
