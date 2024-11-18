
import os
import time
from typing import List, Dict, Set
from heuristics import jw_os, jw_ts, mom
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS,to_DIMACS_Sixteen,DIMACS_reader_Sixteen


class Metrics:
    def __init__(self):
        self._start_time = 0
        self._end_time = 0
        self._backtrack_counter = 0
        self._conflict_counter = 0
        self._unit_clause_counter = 0
        self._pure_literal_counter = 0


    def get_start_time(self):
        self._start_time = time.time()

    def get_end_time(self):
        self._end_time = time.time()

    def get_time_interval(self):
        return self._end_time - self._start_time
    
    def increase_backtrack_counter(self):
        self._backtrack_counter += 1

    def get_backtrack_counter(self):
        return self._backtrack_counter
    
    def increase_conflict_counter(self):
        self._conflict_counter += 1

    def get_conflict_counter(self):
        return self._conflict_counter
    
    def increase_pure_literal_counter(self):
        self._pure_literal_counter += 1

    def get_pure_literal_counter(self):
        return self._pure_literal_counter
    
    def increase_unit_clause_counter(self):
        self._unit_clause_counter += 1

    def get_conflict_counter(self):
        return self._unit_clause_counter