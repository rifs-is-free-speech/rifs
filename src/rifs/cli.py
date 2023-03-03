"""CLI for rifs package. Contains all the commands that the library supports.
The CLI is written with click."""

import click
from art import text2art
from os.path import join, abspath

from rifs.utils import is_package_installed
from rifs import __version__

from rifsdatasets import all_datasets

from rifsalignment import align_csv, alignment_methods


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
    type=click.Path(),
    help="Path to the data directory. Default: data",
)
@click.option(
    "--model-path",
    default="models",
    type=click.Path(),
    help="Path to the model directory. Default: models",
)
@click.option(
    "--output-path",
    default="output",
    type=click.Path(),
    help="Path to the output directory. Default: output",
)
@click.option(
    "--noise-path",
    default="noise",
    type=click.Path(),
    help="Path to the noise directory. Default: noise",
)
@click.pass_context
def cli(
    ctx, version, verbose, quiet, seed, data_path, model_path, output_path, noise_path
):
    """CLI for rifs package. Contains all the commands that the library supports.
    The CLI is written with click."""
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
    """Download NOISE-PACK from FreeSound.org"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Download parameters:")
            click.echo("\tnoise_pack: " + noise_pack)
        click.echo(f"Trying to download {noise_pack} noise pack from FreeSound.org")


@cli.command()
@click.argument(
    "dataset", nargs=1, type=click.Choice(all_datasets.keys(), case_sensitive=False)
)
@click.pass_context
def download_dataset(ctx, dataset):
    """Download rifs DATASET"""
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
    "--alignment-method",
    default="ctc",
    help="Alignment method.",
    type=click.Choice(alignment_methods.keys(), case_sensitive=False),
)
@click.option(
    "--model",
    help="The path to the model to use for alignment. Can be a huggingface model or a local path.",
    type=str,
)
@click.argument(
    "dataset", nargs=1, type=click.Choice(all_datasets.keys()), case_sensitive=False
)
@click.pass_context
def align(ctx, alignment_method, dataset, model):
    """Align DATASET"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Align parameters:")
            click.echo("\tdataset: " + dataset)
        click.echo(f"Aligning {dataset}")

    align_csv(
        data_path=join(abspath(ctx.obj["data_path"]), "raw", dataset, "all.csv"),
        align_method=alignment_methods[alignment_method],
        model=model,
    )


@cli.command()
@click.option(
    "--with-noise-pack",
    help="Preprocess with noise pack. If not specified, no noise will be added.",
)
@click.option(
    "--with-room-simulation",
    is_flag=True,
    help="Preprocess with simulated room acoustics.",
)
@click.option(
    "--with-voice-conversion", is_flag=True, help="Preprocess with voice conversion."
)
@click.pass_context
def augment(ctx, with_noise_pack, with_room_simulation, with_voice_conversion):
    """Preprocess the data and save copy to disk ready for training."""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Preprocess parameters:")
            click.echo(f"\twith_noise_pack: {with_noise_pack}")
            click.echo(f"\twith_room_simulation: {with_room_simulation}")
            click.echo(f"\twith_voice_conversion: {with_voice_conversion}")


@cli.command()
@click.pass_context
def pretrain(ctx):
    """Pretrain model unsupervised"""


@cli.command()
@click.option(
    "--pretrained-model",
    help="Path to the pretrained model on disk or name of Huggingface model to base model on",
)
@click.option("--hours", default=0, help="Number of hours to train for. Default: 0")
@click.option("--minutes", default=1, help="Number of minutes to train for. Default: 1")
@click.pass_context
def finetune(ctx, pretrained_model, hours, minutes):
    """Finetune model"""
    requirements = ["transformers", "torch", "soundfile", "librosa"]
    for package in requirements:
        if not is_package_installed(package):
            click.echo(f"Please install the '{package}' package to use this command.")
            exit(1)

    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Finetune parameters:")
            click.echo(f"\tpretrained_model: {pretrained_model}")
            click.echo(f"\thours: {hours}")
            click.echo(f"\tminutes: {minutes}")
