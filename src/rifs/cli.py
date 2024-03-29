"""CLI for rifs package. Contains all the commands that the library supports.
The CLI is written with click.

.. code-block:: text

    Usage: rifs [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

      CLI for rifs package. Contains all the commands that the library supports.
      The CLI is written with click.

      Usage:  rifs [OPTIONS]

    Options:
      --version              Prints the version of the package.
      --verbose              Enable verbose output.
      --quiet                Disable all output.
      --seed INTEGER         Seed for the random number generator. Default: 0
      --data-path PATH       Path to the data directory. Default: data
      --model-path PATH      Path to the model directory. Default: models
      --output-path PATH     Path to the output directory. Default: output
      --noise-path PATH      Path to the noise directory. Default: noise
      --custom-dataset PATH  Name of a custom dataset. Default: None
      --help                 Show this message and exit.

    Commands:
      align                Usage: align [OPTIONS] DATASET
      augment              Usage: augment [OPTIONS] DATASET
      datasplit            Usage: datasplit [OPTIONS] DATASET
      download-dataset     Usage: download-dataset DATASET
      download-noise       Usage: download-noise NOISE_PACK
      evaluate             Usage: evaluate [OPTIONS] DATASET EXPERIMENT_NAME
      evaluate-alignments  Usage: evaluate-dataset DATASET
      export-table         Usage: export-table EXPERIMENT_NAME
      finetune             Usage: finetune [OPTIONS] DATASET MODEL_NAME
      hubert-preprocess    Usage: hubert-preprocess [OPTIONS] DATASET
      merge-datasets       Usage: merge-datasets [OPTIONS] DATASET, ...
      pretrain             Usage: pretrain MODEL

"""
from __future__ import annotations

import click
from art import text2art
from os.path import join, abspath, exists
from os import makedirs

from rifs import __version__


from rifs.hubert import hubert_preprocess_1st, hubert_preprocess_2nd
from rifs.fairseq import all_models, run_fairseq_pretrain
from rifsdatasets import all_datasets
from rifsalignment import alignment_methods

extra_dataset_choice = "Custom"
dataset_choices = list(all_datasets.keys()) + [extra_dataset_choice]


@click.group(chain=True, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Prints the version of the package.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", is_flag=True, help="Disable all output.")
@click.option(
    "--seed", default=0, help="Seed for the random number generator. Default: 0"
)
@click.option(
    "--data-path",
    default="data",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the data directory. Default: data",
)
@click.option(
    "--model-path",
    default="models",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the model directory. Default: models",
)
@click.option(
    "--output-path",
    default="output",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the output directory. Default: output",
)
@click.option(
    "--noise-path",
    default="noise",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the noise directory. Default: noise",
)
@click.option(
    "--custom-dataset",
    default=None,
    type=click.Path(exists=True, resolve_path=True),
    help="Name of a custom dataset. Default: None",
)
@click.pass_context
def cli(
    ctx,
    version,
    verbose,
    quiet,
    seed,
    data_path,
    model_path,
    output_path,
    noise_path,
    custom_dataset,
):
    """CLI for rifs package. Contains all the commands that the library supports.
    The CLI is written with click.

    Usage:  rifs [OPTIONS]
    """
    if version:
        click.echo(__version__)
        exit(0)
    if not ctx.invoked_subcommand:
        click.echo(text2art("rifs") + "Welcome to the cli of rifs is free speech.")
        click.echo(cli.get_help(ctx))
        exit(0)

    ctx.ensure_object(dict)
    params = [
        (verbose, "verbose"),
        (quiet, "quiet"),
        (seed, "seed"),
        (data_path, "data_path"),
        (model_path, "model_path"),
        (output_path, "output_path"),
        (noise_path, "noise_path"),
        (custom_dataset, "custom_dataset"),
    ]
    for param, param_name in params:
        ctx.obj[param_name] = param

    if not quiet:
        if verbose:
            click.echo(text2art("rifs") + "Welcome to the cli of rifs is free speech.")
            click.echo("Verbose output is enabled")
            click.echo("Global parameters:")
            for param, param_name in params:
                click.echo(f"\t{param_name}: {param}")


@cli.command()
@click.argument("noise-pack", nargs=1)
@click.pass_context
def download_noise(ctx, noise_pack):
    """Usage:  download-noise NOISE_PACK"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Download parameters:")
            click.echo("\tnoise_pack: " + noise_pack)
        click.echo(f"Trying to download {noise_pack} noise pack from FreeSound.org")
    raise NotImplementedError


@cli.command()
@click.argument(
    "dataset", nargs=1, type=click.Choice(all_datasets.keys(), case_sensitive=False)
)
@click.pass_context
def download_dataset(ctx, dataset):
    """Usage:  download-dataset DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Download parameters:")
            click.echo("\tdataset: " + dataset)
        click.echo(f"Downloading {dataset}")

    all_datasets[dataset].download(
        target_folder=join(abspath(ctx.obj["data_path"]), "raw"),
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )


@cli.command()
@click.option(
    "--specify-dir",
    "-s",
    type=str,
    default=None,
    multiple=True,
    help=(
        "Specify the directory to use for the dataset. If not set will copy everything. "
        "Useful to copy only, for example, the alignments. Default: None"
    ),
)
@click.argument("dataset", type=click.Path(exists=True), nargs=-1)
@click.argument("new_dataset", nargs=1)
@click.pass_context
def merge_datasets(ctx, specify_dir, dataset, new_dataset):
    """Usage:  merge-datasets [OPTIONS] DATASET, ... , DATASET NEW_DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Merge parameters:")
            click.echo("\tspecify_dir: " + str(specify_dir))
            click.echo(f"\tdataset: {', '.join(dataset)}")
            click.echo("\tnew_dataset: " + new_dataset)
        click.echo(f"Merging {', '.join(dataset)} into {new_dataset}")

    assert len(dataset) > 1, "You need to provide at least two datasets to merge."

    src_dataset = [d for d in dataset]
    trg_dataset = join(ctx.obj["data_path"], "custom", new_dataset)

    from rifsdatasets import merge_rifsdatasets

    merge_rifsdatasets(
        src_dataset=src_dataset,
        trg_dataset=trg_dataset,
        specify_dirs=specify_dir,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )

    print(f"Done creating {new_dataset}.")


@cli.command()
@click.option(
    "--alignment-method",
    default=list(alignment_methods.keys())[1],
    help="Alignment method.",
    type=click.Choice(alignment_methods.keys(), case_sensitive=False),
)
@click.option(
    "--model",
    default="Alvenir/wav2vec2-base-da-ft-nst",
    help="The path to the model to use for alignment. Can be a huggingface model or a local path.",
    type=str,
)
@click.option(
    "--max-duration",
    default=15,
    help="Maximum duration of the audio files in seconds. Default: 10",
    type=float,
)
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.pass_context
def align(ctx, alignment_method, model, max_duration, dataset):
    """Usage:  align [OPTIONS] DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Align parameters:")
            click.echo("\talignment_method: " + alignment_method)
            click.echo("\tmodel: " + model)
            click.echo("\tmax_duration: " + str(max_duration))
            click.echo("\tdataset: " + dataset)
        click.echo(f"Aligning {dataset}")

    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"
    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    from rifsalignment import align_csv

    align_csv(
        data_path=join(abspath(ctx.obj["data_path"]), folder, dataset),
        align_method=alignment_methods[alignment_method],
        model=model,
        max_duration=max_duration,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )

    print(f"Finished aligning the {dataset} dataset!")


@cli.command()
@click.option(
    "--with-noise-pack",
    type=str,
    help="Preprocess with noise pack. If not specified, no noise will be added.",
)
@click.option(
    "--with-room-simulation",
    is_flag=True,
    default=False,
    help="Preprocess with simulated room acoustics.",
)
@click.option(
    "--with-speed-modification",
    type=float,
    help="Preprocess with speed modification.",
    default=1.0,
)
@click.option(
    "--include-audio-folder",
    is_flag=True,
    help="Include the audio folder in data augmentation.",
)
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.pass_context
def augment(
    ctx,
    with_noise_pack,
    with_room_simulation,
    with_speed_modification,
    include_audio_folder,
    dataset,
):
    """Usage:  augment [OPTIONS] DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Preprocess parameters:")
            click.echo(f"\twith_noise_pack: {with_noise_pack}")
            click.echo(f"\twith_room_simulation: {with_room_simulation}")
            click.echo(f"\twith_speed_modification: {with_speed_modification}")

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]
    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"

    augments = []
    if with_noise_pack:
        augments.append(with_noise_pack)
    if with_room_simulation:
        augments.append("room")
    if with_speed_modification != 1.0:
        augments.append(f"speed{with_speed_modification}")
    augments = "_".join(augments)
    assert augments, "You need to specify at least one augmentation."

    from rifsaugmentation import augment_all

    if with_noise_pack:
        if not exists(with_noise_pack):
            if not ctx.obj["quiet"]:
                click.echo(
                    f"Could not find noise pack '{with_noise_pack}'. Will try to find it in 'data/noise'"
                )
            with_noise_pack = join(ctx.obj["data_path"], "noise", with_noise_pack)
        assert exists(
            with_noise_pack
        ), f"Noise pack '{with_noise_pack}' does not exist."
        if not ctx.obj["quiet"]:
            click.echo(f"Found '{with_noise_pack}'")

    augment_all(
        source_path=join(ctx.obj["data_path"], folder, dataset),
        target_path=join(
            ctx.obj["data_path"],
            "custom",
            f"{dataset}_{augments}",
        ),
        with_room_simulation=with_room_simulation,
        speed=with_speed_modification,
        noise_path=join(abspath(ctx.obj["noise_path"]), with_noise_pack)
        if with_noise_pack
        else None,
        recursive=True,
        move_other_files=True,
        skip_audio_folder=not include_audio_folder,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )

    print(f"Finished augmenting the {dataset} dataset!")


@cli.command()
@click.option(
    "--iteration",
    default="1st",
    type=click.Choice(["1st", "2nd"]),
    help="Features for either first or second iteration  of HUBERT training.",
)
@click.option(
    "--fairseq-path",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the fairseq directory.",
)
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.pass_context
def hubert_preprocess(ctx, iteration, fairseq_path, dataset):
    """Usage:  hubert-preprocess [OPTIONS] DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Preprocess parameters:")
            click.echo(f"\titeration: {iteration}")
            click.echo(f"\tfairseq_path: {fairseq_path}")
            click.echo(f"\tdataset: {dataset}")

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    if iteration == "1st":
        hubert_preprocess_1st(ctx=ctx.obj, fairseq_path=fairseq_path, dataset=dataset)

    elif iteration == "2nd":
        hubert_preprocess_2nd(ctx=ctx.obj, dataset=dataset)


@cli.command()
@click.option(
    "--fairseq-path",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to the fairseq directory.",
)
@click.option(
    "--manifest-source",
    type=click.Path(exists=True, resolve_path=True),
    help="Optional: Path to wav files for manifest creation",
)
@click.argument(
    "model", nargs=1, type=click.Choice(all_models.keys(), case_sensitive=False)
)
@click.pass_context
def pretrain(ctx, fairseq_path, model, manifest_source):
    """Usage:  pretrain MODEL"""
    if fairseq_path is None:
        click.echo("Please specify the path to the fairseq directory to pretrain.")
        click.echo("You can do this with the --fairseq-path option.")
        exit(1)

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Pretrain parameters:")
            click.echo(f"\tmodel: {model}")
            click.echo(f"\tfairseq_path: {fairseq_path}")
            for param, param_value in all_models[model].items():
                click.echo(f"\t{param}: {param_value}")
    run_fairseq_pretrain(
        fairseq_path=fairseq_path,
        model_dict=all_models[model],
        ctx=ctx.obj,
        manifest_source=manifest_source,
    )


@cli.command()
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.option(
    "--split-method",
    type=click.Choice(["random"], case_sensitive=False),
    default="random",
    help="How to split the dataset into train, dev and test.",
)
@click.option(
    "--split-ratio",
    type=click.FLOAT,
    default=0.8,
    help="Ratio of train to test/dev data.",
)
@click.option(
    "--split-test-ratio",
    type=click.FLOAT,
    default=0.5,
    help="Ratio of test to dev data.",
)
@click.option(
    "--check-alignments",
    is_flag=True,
    help="Check if alignments are correct. Requires model_output",
)
@click.pass_context
def datasplit(
    ctx, dataset, split_method, split_ratio, split_test_ratio, check_alignments
):
    """Usage:  datasplit [OPTIONS] DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Data split parameters:")
            click.echo(f"\tdataset: {dataset}")
            click.echo(f"\tsplit_method: {split_method}")
            click.echo(f"\tsplit_ratio: {split_ratio}")
            click.echo(f"\tsplit_test_ratio: {split_test_ratio}")
            click.echo(f"\tcheck_alignments: {check_alignments}")

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    from rifsdatasets import split_dataset

    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"

    split_dataset(
        dataset_path=join(ctx.obj["data_path"], folder, dataset),
        split_method=split_method,
        split_ratio=split_ratio,
        split_test_ratio=split_test_ratio,
        check_for_bad_alignments=check_alignments,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
        seed=ctx.obj["seed"],
    )

    print(f"Finished splitting the '{dataset}' dataset!")


@cli.command()
@click.option(
    "--pretrained-model",
    default="Alvenir/wav2vec2-base-da",
    help="Path to the pretrained model on disk or name of Huggingface model to base model on",
)
@click.option("--hours", default=0, help="Number of hours to train for. Default: 0")
@click.option("--minutes", default=1, help="Number of minutes to train for. Default: 1")
@click.option(
    "--reduced-training-arguments",
    is_flag=True,
    help="Reduce training arguments for testing",
)
@click.option("--warmup-steps", default=0, help="Number of warmup steps. Default: 0")
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.argument(
    "model_name",
    nargs=1,
)
@click.pass_context
def finetune(
    ctx,
    pretrained_model,
    hours,
    minutes,
    reduced_training_arguments,
    warmup_steps,
    dataset,
    model_name,
):
    """Usage:  finetune [OPTIONS] DATASET MODEL_NAME"""

    from rifstrain import finetune as finetune_rifs

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Finetune parameters:")
            click.echo(f"\tpretrained_model: {pretrained_model}")
            click.echo(f"\thours: {hours}")
            click.echo(f"\tminutes: {minutes}")
            click.echo(f"\tdataset: {dataset}")
            click.echo(f"\treduced_training_arguments: {reduced_training_arguments}")
            click.echo(f"\twarmup_steps: {warmup_steps}")
            click.echo(f"\tmodelname: {model_name}")

    assert warmup_steps >= 0, "Warmup steps must be greater than or equal to 0."

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"

    if not ctx.obj["quiet"]:
        print_string = f"Finetuning the '{dataset}' dataset for "
        if hours > 0:
            print_string += f"{hours} hours{'s' if hours > 1 else ''}{' and ' if minutes > 0 else ''}"
        if minutes > 0:
            print_string += f"{minutes} minute{'s' if minutes > 1 else ''}"
        click.echo(print_string)

    finetune_rifs(
        csv_train_file=join(
            abspath(ctx.obj["data_path"]), folder, dataset, "train.csv"
        ),
        csv_test_file=join(abspath(ctx.obj["data_path"]), folder, dataset, "valid.csv"),
        pretrained_path=pretrained_model,
        hours=hours,
        minutes=minutes,
        reduced_training_arguments=reduced_training_arguments,
        model_save_location=join(ctx.obj["model_path"], model_name),
        warmup_steps=warmup_steps,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
        seed=ctx.obj["seed"],
    )


@cli.command()
@click.option(
    "--model",
    default="Alvenir/wav2vec2-base-da",
    help="Path to the model on disk or name of Huggingface model to evaluate with",
)
@click.option(
    "--split",
    default="test",
    help="Which split to evaluate on. Default: test",
)
@click.option(
    "--spellcheck-model",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to spellcheck model. Default: None",
)
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.argument(
    "experiment_name",
    nargs=1,
)
@click.pass_context
def evaluate(
    ctx,
    model,
    split,
    spellcheck_model,
    dataset,
    experiment_name,
):
    """Usage:  evaluate [OPTIONS] DATASET EXPERIMENT_NAME"""

    from rifstrain import evaluate as evaluate_rifs

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Finetune parameters:")
            click.echo(f"\tmodel: {model}")
            click.echo(f"\tsplit: {split}")
            click.echo(f"\tspellcheck_model: {spellcheck_model}")
            click.echo(f"\tdataset: {dataset}")
            click.echo(f"\texperimentname: {experiment_name}")

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"

    output_path = join(ctx.obj["output_path"], experiment_name)
    if ctx.obj["quiet"] and not ctx.obj["verbose"]:
        print(f"Creating experiment folder at '{output_path}'")
    makedirs(output_path, exist_ok=True)

    if not ctx.obj["quiet"]:
        click.echo(
            f"Evaluating the '{dataset}' dataset as part of the {experiment_name} experiment."
        )

    evaluate_rifs(
        csv_test_file=join(
            abspath(ctx.obj["data_path"]), folder, dataset, f"{split}.csv"
        ),
        pretrained_path=model,
        output_path=output_path,
        experiment_name=experiment_name,
        spellcheck_model=spellcheck_model,
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )


@cli.command()
@click.argument(
    "dataset", nargs=1, type=click.Choice(dataset_choices, case_sensitive=False)
)
@click.pass_context
def evaluate_alignments(
    ctx,
    dataset,
):
    """Usage:  evaluate-dataset DATASET"""
    from rifsalignment import align_experiment_folder

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Evaluate-dataset parameters:")
            click.echo(f"\tdataset: {dataset}")

    if dataset.lower() == extra_dataset_choice.lower():
        assert ctx.obj["custom_dataset"], "You need to specify a custom dataset."
        assert exists(
            join(ctx.obj["data_path"], "custom", ctx.obj["custom_dataset"])
        ), f"Dataset '{ctx.obj['custom_dataset']}' does not exist."
        dataset = ctx.obj["custom_dataset"]

    if ctx.obj["custom_dataset"]:
        folder = "custom"
    else:
        folder = "raw"

    if not ctx.obj["quiet"]:
        click.echo(f"Evaluating the '{dataset}' dataset for quality of alignments")

    align_experiment_folder(
        folder_path=join(abspath(ctx.obj["data_path"]), folder, dataset),
        verbose=ctx.obj["verbose"],
        quiet=ctx.obj["quiet"],
    )


@cli.command()
@click.argument(
    "experiment_name",
    nargs=1,
)
@click.pass_context
def export_table(ctx, experiment_name):
    """Usage:  export-table EXPERIMENT_NAME"""

    import numpy as np
    import pandas as pd
    import pyperclip as pc

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Finetune parameters:")
            click.echo(f"\texperiment_name: {experiment_name}")

    df = pd.read_csv(
        join(ctx.obj["output_path"], experiment_name, "results.csv")
    ).replace(r"_", "-", regex=True)

    models = df["model"].unique().tolist()
    datasets = df["dataset"].unique().tolist()
    metrics = df["metric"].unique().tolist()

    index = pd.MultiIndex.from_product(
        iterables=[df["dataset"].unique(), df["metric"].unique().tolist()],
        names=["Dataset", "Metrics"],
    )

    data = []
    for model in models:
        for dataset in datasets:
            for metric in metrics:
                try:
                    val = df.loc[
                        (df["model"] == model)
                        & (df["dataset"] == dataset)
                        & (df["metric"] == metric)
                    ]["value"].values[0]
                    data.append(val)
                except IndexError:
                    data.append(0)
    results = np.array(data).reshape(len(models), len(datasets) * len(metrics))

    new_df = pd.DataFrame(results, index=models, columns=index)

    caption = (
        f"Results for {experiment_name}. The best result for each metric is highlighted. "
        "WER is the word error rate, CER is the character error rate, "
        "and LSR is the Levenshtein similarity ratio."
    )

    r = new_df.style.format("{:.2f}").to_latex(
        convert_css=True,
        hrules=True,
        position="h!",
        position_float="centering",
        multicol_align="c",
        caption=caption,
    )

    try:
        pc.copy(r)
    except pc.PyperclipException:
        click.echo("Table could not be copied to clipboard")

    if ctx.obj["verbose"]:
        click.echo(r + "\n")

    click.echo("Table copied to clipboard.")
