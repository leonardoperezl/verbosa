from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Sequence
import logging.config


from verbosa.utils.global_typings import Pathlike
from verbosa.utils.validation_helpers import is_file_path
from verbosa.data.readers.local import FileDataReader


LogLevels = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ConfigPathDescriptor:
    def __set_name__(self, owner: type[LogsMachine], name: str) -> None:
        self.name = name
    
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value: Pathlike) -> None:
        config_dict_attr = "config_dict"
        instance.__dict__[config_dict_attr] = FileDataReader.read_file(value)
        instance.__dict__[self.name] = value


class LogsMachine:
    config_path: Pathlike = ConfigPathDescriptor()
    
    def __init__(self, config_path: Pathlike) -> None:
        self.config_dict: dict = dict()
        self.config_path: Pathlike = config_path
    
    def on(self) -> None:
        logging.config.dictConfig(self.config_dict)
        logger = logging.getLogger(__name__)
        logger.info("LoggerMachine activated.")


if __name__ == "__main__":
    ...