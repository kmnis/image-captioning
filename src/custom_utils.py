import os
import argparse
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
import logging
import sys
import json

# ---------------------------------- Logger ---------------------------------- #

def get_logger(name: str) -> logging.Logger:
    """
    Parameters
    ----------
    name : str
        A string that specifies the name of the logger.

    Returns
    -------
    logging.Logger
        A logger with the specified name.
    """
    logger = logging.getLogger(name)  # Return a logger with the specified name
    
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    
    return logger

# --------------------- Parse argument from command line --------------------- #

def parser() -> argparse.ArgumentParser:
    """
    Function that parses arguments from the command line.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object that contains the arguments passed from command line.
    """
    parser = argparse.ArgumentParser()

    # Flag (true/false) testing with small samples
    parser.add_argument('--test_mode', action='store_true')

    return parser

# ------ Function decorator for adding additional command line arguments ----- #

def add_additional_args(parser_func: Callable, additional_args: Dict[str, type]) -> Callable:
    """
    Function decorator that adds additional command line arguments to the parser.
    This allows for adding additional arguments without having to change the base
    parser.

    Parameters
    ----------
    parser_func : Callable
        The parser function to add arguments to.
    additional_args : Dict[str, type]
        A dictionary where the keys are the names of the arguments and the values
        are the types of the arguments, e.g. {'arg1': str, 'arg2': int}.

    Returns
    -------
    Callable
        A parser function that returns the ArgumentParser object with the additional arguments added to it.
    """
    def wrapper():
        # Call the original parser function to get the parser object
        parser = parser_func()

        for arg_name, arg_type in additional_args.items():
            parser.add_argument(f'--{arg_name}', type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper