"""Several processes building one cache at once must not collide on a temporary file."""
import multiprocessing
import os
import tempfile
import time
import unittest

import numpy as np

from possession import config, data


def _build():
    time.sleep(0.5)  # keep every worker inside build() together
    return {"a": np.arange(1000)}


def _worker(cache_dir, src, start, out):
    os.environ["MTAH_CACHE_DIR"] = cache_dir
    start.wait()
    try:
        out.put(int(data._cached("conc", [src], _build)["a"].sum()))
    except Exception as exc:  # report to parent
        out.put(repr(exc))


class CacheConcurrency(unittest.TestCase):
    def test_parallel_builders(self):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            with open(src, "w") as f:
                f.write("x")
            start, out = ctx.Event(), ctx.Queue()
            procs = [ctx.Process(target=_worker, args=(d, src, start, out)) for _ in range(4)]
            for p in procs:
                p.start()
            start.set()
            results = [out.get(timeout=60) for _ in procs]
            for p in procs:
                p.join()
            self.assertEqual(results, [499500] * 4)
            leftovers = [n for n in os.listdir(d) if ".tmp" in n]
            self.assertEqual(leftovers, [])

    def test_tmp_path_unique(self):
        a, b = config.tmp_path("/x/c.npz"), config.tmp_path("/x/c.npz")
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith(".npz"))


if __name__ == "__main__":
    unittest.main()
