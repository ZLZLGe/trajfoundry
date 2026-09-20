from pathlib import Path

import orjson
import pytest
from typer.testing import CliRunner

from trajfoundry.cli import app


@pytest.mark.parametrize(
    "lineage_path", ["lineage.jsonl", "generations/run-1/lineage.jsonl"]
)
def test_inspect_finds_top_level_and_generation_lineage(
    tmp_path: Path, lineage_path: str
) -> None:
    lineage = tmp_path / lineage_path
    lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_bytes(orjson.dumps({"trajectory_id": "trajectory-1"}) + b"\n")
    (tmp_path / "manifest.json").write_bytes(
        orjson.dumps({"files": [{"path": lineage_path}]})
    )

    result = CliRunner().invoke(
        app,
        [
            "inspect",
            "--output",
            str(tmp_path),
            "--trajectory-id",
            "trajectory-1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"trajectory_id": "trajectory-1"' in result.output
