# basic_generators.py
# Author: Fan Wang (wang.fan@baidu.com)
#
# Generators base class

import sys
import random
from environment.mansion.utils import PersonType
from environment.mansion.mansion_config import MansionConfig


class PersonGeneratorBase(object):
    '''
    Basic Generator Class
    Generates Random Person from Random Floor, going to random other floor
    '''

    def __init__(self):
        pass

    def configure(self, configuration):
        pass

    def link_mansion(self, mansion_config):
        self._config = mansion_config
        self._floor_number = self._config.number_of_floors
        self._last_generate_time = self._config.raw_time

    def generate_person(self, time):
        '''
        Generate Random Persons from Poisson Distribution
        Args:
          None
        Returns:
          List of Random Persons
        '''
        raise NotImplementedError()
