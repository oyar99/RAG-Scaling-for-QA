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
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self._logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
        self._run_id = str(uuid.uuid4())
    
    def info(self, message: str) -> None:
        """Logs a message at the INFO level.

        Args:
            message (str): message to be logged
        """
        self._logger.info(message)
        
    def get_run_id(self) -> str:
        """Returns the run id.

        Returns:
            str: run id
        """
        return self._run_id