import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed_time = self.end - self.start
        print(f"Elapsed time: {self.elapsed_time:.2f} seconds")
