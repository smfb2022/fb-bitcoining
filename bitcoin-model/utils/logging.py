"""
A wrapper of built-in logging with custom level support and utilities.
"""

import logging as _logging
import os
import types
from contextlib import contextmanager
from logging import *  

VERBOSE = 15

_logging.addLevelName(VERBOSE, "VERBOSE")


def _verbose(self, message, *args, **kwargs):
    """_summary_

    Args:
        message (_type_): _description_
    """
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)  


def getLogger(name=None):  
    """_summary_

    Args:
        name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger = _logging.getLogger(name)
    # Inject verbose method to logger object instead logging module
    logger.verbose = types.MethodType(_verbose, logger)
    return logger


_BASIC_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_VERBOSE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"


def basicConfig(**kwargs):  # pylint: disable=invalid-name, function-redefined
    """Do basic configuration for the logging system. tf verbosity is updated accordingly."""
    # Choose pre-defined format if format argument is not specified
    if "format" not in kwargs:
        level = kwargs.get("level", _logging.root.level)
        kwargs["format"] = _BASIC_LOG_FORMAT if level >= INFO else _VERBOSE_LOG_FORMAT
    # config will make effect only when root.handlers is empty, so add the following statement to make sure it
    _logging.root.handlers = []
    _logging.basicConfig(**kwargs)
    set_tf_verbosity(_logging.getLogger().getEffectiveLevel())


_LOG_LEVELS = [FATAL, ERROR, WARNING, INFO, VERBOSE, DEBUG]


def get_verbosity_level(verbosity, base_level=INFO):
    """If verbosity is specified, return corresponding level, otherwise, return default_level.

    Args:
        verbosity (_type_): _description_
        base_level (_type_, optional): _description_. Defaults to INFO.

    Returns:
        _type_: _description_
    """
    if verbosity is None:
        return base_level
    verbosity = min(max(0, verbosity) + _LOG_LEVELS.index(base_level), len(_LOG_LEVELS) - 1)
    return _LOG_LEVELS[verbosity]


def set_level(level, name=""):
    """Set logging level. tf verbosity is updated accordingly.

    Args:
        level (_type_): _description_
        name (str, optional): _description_. Defaults to "".
    """
    _logging.getLogger(name).setLevel(level)


@contextmanager
def set_scope_level(level, logger=None):
    """Set logging level to logger within context, reset level to previous value when exit context.
    TF verbosity is NOT affected.

    Args:
        level (_type_): _description_
        logger (_type_, optional): _description_. Defaults to None.

    Yields:
        _type_: _description_
    """
    if logger is None:
        logger = getLogger()

    current_level = logger.level
    logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(current_level)