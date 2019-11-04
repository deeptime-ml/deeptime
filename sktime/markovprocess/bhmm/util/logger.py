
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import sys
from bhmm.util import config


def logger(name='BHMM', pattern='%(asctime)s %(levelname)s %(name)s: %(message)s',
           date_format='%H:%M:%S', handler=logging.StreamHandler(sys.stdout)):
    """
    Retrieves the logger instance associated to the given name.

    :param name: The name of the logger instance.
    :type name: str
    :param pattern: The associated pattern.
    :type pattern: str
    :param date_format: The date format to be used in the pattern.
    :type date_format: str
    :param handler: The logging handler, by default console output.
    :type handler: FileHandler or StreamHandler or NullHandler

    :return: The logger.
    :rtype: Logger
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(config.log_level())
    if not _logger.handlers:
        formatter = logging.Formatter(pattern, date_format)
        handler.setFormatter(formatter)
        handler.setLevel(config.log_level())
        _logger.addHandler(handler)
        _logger.propagate = False
    return _logger
