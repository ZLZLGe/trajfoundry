"""Minimal DolphinScheduler Python node for a TrajFoundry run."""


def main():
    import logging

    from trajfoundry.jobs import run_job

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_job(
        input_root="${input_root}",
        output_root="${output_root}",
        input_format="${input_format}",
        resume=True,
    )


if __name__ == "__main__":
    main()
