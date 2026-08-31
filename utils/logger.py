import logging

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    if HAS_COLORLOG:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(handler)
