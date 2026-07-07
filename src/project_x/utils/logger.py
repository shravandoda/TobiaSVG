import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=True,
            )
        ],
        force=True,
    )

    for logger_name in ["httpx", "httpcore", "huggingface_hub", "datasets"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
