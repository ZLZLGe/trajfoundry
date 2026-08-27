"""Command-line interface for TrajFoundry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import orjson
import typer

from .pipeline import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    PipelineConfig,
    stats_json,
)
from .pipeline import (
    normalize as run_normalize,
)
from .validation import validate_output

app = typer.Typer(
    name="trajfoundry",
    help="Normalize freerouter captures into auditable trajectory JSONL.",
    no_args_is_help=True,
)


@app.command("normalize")
def normalize_command(
    input_root: Annotated[
        Path,
        typer.Option("--input", help="Root containing freerouter JSON captures."),
    ] = DEFAULT_INPUT,
    output_root: Annotated[
        Path,
        typer.Option("--output", help="Destination for normalized shards."),
    ] = DEFAULT_OUTPUT,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Reuse the durable ingest state."),
    ] = False,
    state_path: Annotated[
        Path | None,
        typer.Option("--state", help="Override the SQLite state path."),
    ] = None,
    max_shard_mib: Annotated[
        int,
        typer.Option(min=1, help="Maximum uncompressed size of each JSONL shard."),
    ] = 512,
) -> None:
    """Normalize captures, aggregate trajectories, and publish output shards."""

    try:
        stats = run_normalize(
            PipelineConfig(
                input_root=input_root,
                output_root=output_root,
                state_path=state_path,
                resume=resume,
                max_shard_bytes=max_shard_mib * 1024 * 1024,
            )
        )
    except (OSError, ValueError) as error:
        typer.echo(f"normalize failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(stats_json(stats))


@app.command("validate")
def validate_command(
    output_root: Annotated[
        Path,
        typer.Option("--output", help="TrajFoundry output directory."),
    ] = DEFAULT_OUTPUT,
) -> None:
    """Verify checksums, schemas, lineage, and strict admission labels."""

    report = validate_output(output_root)
    typer.echo(json.dumps(report.model_dump(), ensure_ascii=False, sort_keys=True))
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_command(
    output_root: Annotated[
        Path,
        typer.Option("--output", help="TrajFoundry output directory."),
    ] = DEFAULT_OUTPUT,
    trajectory: Annotated[
        str | None,
        typer.Option("--trajectory-id", help="Show lineage for one trajectory ID."),
    ] = None,
) -> None:
    """Print the run summary or provenance for one trajectory."""

    manifest_path = output_root / "manifest.json"
    try:
        manifest = orjson.loads(manifest_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as error:
        typer.echo(
            f"inspect failed: invalid or missing manifest ({type(error).__name__})",
            err=True,
        )
        raise typer.Exit(code=2) from error

    if trajectory is None:
        summary = {
            "schema_version": manifest.get("schema_version"),
            "created_at": manifest.get("created_at"),
            "input_root": manifest.get("input_root"),
            "counts": manifest.get("counts", {}),
        }
        typer.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return

    lineage_entries = [
        entry.get("path")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry["path"].endswith("/lineage.jsonl")
    ]
    if len(lineage_entries) != 1:
        typer.echo("inspect failed: manifest has no unique lineage file", err=True)
        raise typer.Exit(code=2)
    lineage_path = output_root / lineage_entries[0]
    try:
        with lineage_path.open("rb") as handle:
            for raw_line in handle:
                try:
                    value = orjson.loads(raw_line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("trajectory_id") == trajectory:
                    typer.echo(
                        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
                    )
                    return
    except OSError as error:
        typer.echo(
            f"inspect failed: lineage unavailable ({type(error).__name__})", err=True
        )
        raise typer.Exit(code=2) from error
    typer.echo(f"trajectory not found: {trajectory}", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
