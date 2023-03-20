""" Interface for fairseq pre-training.
Contains the commands found fairseq/examples/...
"""


def run_fairseq_pretrain(model_dict: dict, ctx: dict):
    """Run fairseq pre-training.

    Parameters:
    -----------
    model_dict: dict
        Dictionary of arguments to pass to fairseq pre-training.

    ctx: dict
        Dictionary of rifs context arguments.

    Returns:
    --------
    None
    """
    if ctx["verbose"]:
        print("Running fairseq pre-training...")


all_models = {
    "wav2vec2_base": {
        "help_text": "wav2vec2.0 base model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "wav2vec2_base_librispeech",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
        },
        "x/k": 64,
    },
    "wav2vec2_large": {
        "help_text": "wav2vec2.0 large model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "wav2vec2_large_librivox",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
        },
        "x/k": 128,
    },
    "wav2vec2_conformer_base": {
        "help_text": "wav2vec2.0 conformer base model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "wav2vec2_conformer_base_librispeech",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": None,
        },
        "extra_args": {
            "--attn-type": "espnet",
            "--pos-enc-type": "rel_pos",
        },
    },
    "wav2vec2_conformer_large": {
        "help_text": "wav2vec2.0 conformer large model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "wav2vec2_conformer_large_librivox",
        "--config-dir": "examples/wav2vec/config/pretraining",
        "required-args": {
            "task.data": None,
        },
        "extra_args": {
            "--attn-type": "espnet",
            "--pos-enc-type": "rel_pos",
        },
    },
    "hubert_base": {
        "help_text": "hubert base model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "hubert_base_librispeech",
        "--config-dir": "examples/hubert/config/pretrain",
        "required-args": {
            "task.data": None,
            "task.label_dir": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            """task.labels='["km"]'""": None,
            "model.label_rate=100": None,
        },
    },
    "data2vec_base": {
        "help_text": "data2vec base model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-name": "base_librispeech",
        "--confic-dir": "examples/data2vec/config/audio/pretraining",
        "required-args": {
            "task.data": None,
            "common.user_dir": None,
            "distributed_training.distributed_world_size": None,
        },
        "extra_args": {
            "optimization.update_freq='[x]'": None,
        },
    },
    "data2vec2_base": {
        "help_text": "data2vec 2.0 base model from fairseq",
        "command": "fairseq-hydra-train",
        "--confic-dir": "examples/data2vec/config/v2",
        "--config-name": "base_audio_only_task",
        "required-args": {
            "task.data": None,
        },
        "extra_args": None,
    },
    "data2vec2_large": {
        "help_text": "data2vec 2.0 large model from fairseq",
        "command": "fairseq-hydra-train",
        "--config-dir": "examples/data2vec/config/v2",
        "--config-name": "large_audio_only_task",
        "required-args": {
            "task.data": None,
        },
        "extra_args": None,
    },
    "speechT5_base": {
        "help_text": "speechT5 base model from fairseq. Not yet implemented",
        "command": "fairseq-train",
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
    },
}
