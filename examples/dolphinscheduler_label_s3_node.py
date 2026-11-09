"""DolphinScheduler Python node for the compact v2 classification contract.

Parameters: input_uri, output_uri, endpoint_url (S3), classifier_api_url.
CLASSIFIER_API_KEY must already be available in the worker environment.
"""


def main() -> None:
    import logging
    from pathlib import Path

    import trajfoundry
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
    print("classification_source =", trajfoundry.__file__, flush=True)
    print("classifier_revision =", CLASSIFIER_REVISION, flush=True)
    result = run_s3_label_job(
        input_uri="${input_uri}",
        output_uri="${output_uri}",
        endpoint_url="${endpoint_url}",
        classifier_api_url="${classifier_api_url}",
        classifier_model="deepseek-v4.1-flash",
        region_name="us-east-1",
        workspace_parent=Path("/tmp"),
        max_workers=20,
        max_retries=20,
        request_timeout_seconds=180.0,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    print("input_trajectories =", result.input_trajectories, flush=True)
    print("classified =", result.classified, flush=True)
    print("skipped_no_toolcall =", result.skipped_no_toolcall, flush=True)
    print("failed =", result.failed, flush=True)
    print("cache_hits =", result.cache_hits, flush=True)
    print("validation =", result.validation_valid, flush=True)


if __name__ == "__main__":
    main()
