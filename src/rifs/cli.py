"""CLI for rifs package. Contains all the commands that the library supports.
The CLI is written with click."""

import click
from art import text2art

from rifs.utils import is_package_installed
from rifs import __version__


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
    if version:
        click.echo(__version__)
        exit(0)
    if not quiet:
        click.echo(text2art("RiFS") + "Welcome to the CLI of RiFS is Free Speech.")
        if verbose:
            click.echo("Verbose output is enabled")
            click.echo("Global parameters:")
            for param, param_name in params:
                click.echo(f"\t{param_name}: {param}")


@cli.command()
@click.option("--noise-pack", required=True, help="Name of the noise pack to download.")
@click.pass_context
def download(ctx, noise_pack):
    """Download noise packs from FreeSound.org"""
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Download paramers:")
            click.echo("\tnoise_pack: " + noise_pack)
        click.echo(f"Trying to download {noise_pack} noise pack from FreeSound.org")



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
    if not is_package_installed("transformers"):
        click.echo("Please install transformers package to use this command.")
        exit(1)
    if not ctx.obj["quiet"]:
        if ctx.obj["verbose"]:
            click.echo("Finetune parameters:")
            click.echo(f"\tpretrained_model: {pretrained_model}")
            click.echo(f"\thours: {hours}")
            click.echo(f"\tminutes: {minutes}")