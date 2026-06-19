from typer.testing import CliRunner

from yuxi_cli import __version__
from yuxi_cli.main import app


def test_version_option_without_command():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output
