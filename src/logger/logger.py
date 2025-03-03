"""Logger module for logging messages to console and file."""
import logging
import os
import uuid

from utils.singleton import Singleton


class Logger(metaclass=Singleton):
    """
    A class that provides a logger instance.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
        self._logger.setLevel(log_level)

        self._run_id = str(uuid.uuid4())

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # Create file handler
        log_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'logs')
        log_fname = os.path.join(log_dir, f'app-{self._run_id}.log')
        fh = logging.FileHandler(log_fname, encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def info(self, message: str) -> None:
        """Logs a message at the INFO level.

        Args:
            message (str): message to be logged
        """
        self._logger.info(message)

    def warn(self, message: str) -> None:
        """Logs a message at the WARN level.

        Args:
            message (str): message to be logged
        """
        self._logger.warning(message)

    def debug(self, message: str) -> None:
        """Logs a message at the VERBOSE level.

        Args:
            message (str): message to be logged
        """
        self._logger.debug(message)

    def error(self, message: str) -> None:
        """Logs a message at the ERROR level.

        Args:
            message (str): message to be logged
        """
        self._logger.error(message)

    def get_run_id(self) -> str:
        """Returns the run id.

        Returns:
            str: run id
        """
        return self._run_id
