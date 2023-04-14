from rifs import __version__ as VERSION
from rifs.cli import cli


def test_cli(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert not result.exception, result.output
        assert result.output == f"{VERSION}\n"
