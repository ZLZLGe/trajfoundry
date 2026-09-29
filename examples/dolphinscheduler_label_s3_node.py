"""DolphinScheduler Python node for independent S3 trajectory classification."""


def main() -> None:
    import logging
    from pathlib import Path

    from trajfoundry.classification import (
        CLASSIFIER_REVISION,
        DEFAULT_MAX_CONTEXT_CHARS,
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    from trajfoundry.label_jobs import run_s3_label_job

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_s3_label_job(
        input_uri="${input_uri}",
        output_uri="${output_uri}",
        endpoint_url="${endpoint_url}",
        region_name="us-east-1",
        workspace_parent=Path("${workspace_parent}"),
        max_workers=4,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )

    print("classifier_revision =", CLASSIFIER_REVISION, flush=True)
    print("input_trajectories =", result.input_trajectories, flush=True)
    print("classified =", result.classified, flush=True)
    print("failed =", result.failed, flush=True)
    print("cache_hits =", result.cache_hits, flush=True)
    print("validation =", result.validation_valid, flush=True)


if __name__ == "__main__":
    main()
