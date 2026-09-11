"""Minimal DolphinScheduler Python node for a direct S3 TrajFoundry run."""


def main():
    import logging
    from pathlib import Path

    from trajfoundry.jobs import run_s3_job

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_s3_job(
        input_uri="${input_uri}",
        output_uri="${output_uri}",
        input_format="${input_format}",
        endpoint_url="${endpoint_url}",
        region_name="us-east-1",
        workspace_parent=Path("${workspace_parent}"),
    )


if __name__ == "__main__":
    main()
