import time


class Metrics:
    """Class for tracking various metrics within the DPLL algorithm."""

    def __init__(self):
        self._start_time = 0
        self._end_time = 0
        self._backtrack_counter = 0
        self._conflict_counter = 0
        self._unit_clause_counter = 0
        self._pure_literal_counter = 0

    def start_timing(self):
        """Start the timer for the algorithm execution."""
        self._start_time = time.perf_counter()

    def end_timing(self):
        """End the timer for the algorithm execution."""
        self._end_time = time.perf_counter()

    def get_time_interval(self):
        """Calculate the elapsed time from start to end."""
        return self._end_time - self._start_time

    def increment_backtrack_counter(self):
        """Increment the backtrack counter by one."""
        self._backtrack_counter += 1

    def get_backtrack_counter(self):
        """Get the current value of the backtrack counter."""
        return self._backtrack_counter

    def increment_conflict_counter(self):
        """Increment the conflict counter by one."""
        self._conflict_counter += 1

    def get_conflict_counter(self):
        """Get the current value of the conflict counter."""
        return self._conflict_counter

    def increment_unit_clause_counter(self):
        """Increment the unit clause counter by one."""
        self._unit_clause_counter += 1

    def get_unit_clause_counter(self):
        """Get the current value of the unit clause counter."""
        return self._unit_clause_counter

    def increment_pure_literal_counter(self):
        """Increment the pure literal counter by one."""
        self._pure_literal_counter += 1

    def get_pure_literal_counter(self):
        """Get the current value of the pure literal counter."""
        return self._pure_literal_counter
