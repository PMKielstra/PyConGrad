from time import time

from tqdm import tqdm

class Monitor:

    def setup(self, stop_at, maxiter):
        """Called once, just before the solve begins.

        :param stop_at: The maximum absolute error (possibly with relative tolerance for this particular problem baked in) allowed in the solution.
        :type stop_at: vector
        :param maxiter: The maximum number of iterations (can be None).
        :type maxiter: int"""
        pass

    def step(self, iteration, res_norm, x, is_end):
        """Called after every iterative step.

        :param iteration: The number of (batched) matvecs so far.
        :type iteration: int
        :param res_norm: The residual norm, in the format used by the backend.
        :type res_norm: vector
        :param x: The candidate solution, in the format used by the backend.
        :type x: vector
        :param is_end: True if this value of x is about to be returned.
        :type is_end: bool
        """
        pass

class DefaultMonitor(Monitor):
    """A monitor that prints some basic information every few iterations.  The exact number of iterations can be passed to the constructor."""
    def __init__(self, n=20):
        self.n = n

    def setup(self, stop_at, maxiter):
        self.start_time = time()

    def step(self, iteration, res_norm, x, is_end):
        if is_end:
            t = time() - self.start_time
            print(f"Finished in {t:.5e} seconds after {iteration} iterations ({(iteration / t):.5e} iterations/second) with a maximum residual of {res_norm:.5e}.")
        elif iteration % self.n == 0:
            print(f"{iteration:03d}: {res_norm:.5e} ({time() - self.start_time:.5e} seconds)")

class ProgressBarMonitor(Monitor):
    """A monitor that displays a simple progress bar.  Powered by tqdm <https://tqdm.github.io/>."""
    def setup(self, stop_at, maxiter):
        self.pbar = tqdm(total=maxiter)

    def step(self, iteration, res_norm, x, is_end):
        self.pbar.update(1)
        if is_end:
            self.pbar.close()
        