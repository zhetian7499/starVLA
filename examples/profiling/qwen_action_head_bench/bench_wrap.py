#!/usr/bin/env python
"""Per-rank wrapper: invokes model_prof's prof.sh only on rank 0, then bench.py.

Why: nsys profile wrapping `accelerate launch` tries to attach to all 8 child
processes and write a single .nsys-rep, causing contention. By only wrapping
rank 0, we get clean profile data; other ranks run bench.py directly and
participate in collectives normally (so NCCL still happens and is visible
on rank 0's stream).

Required env vars (set by run.sh):
  PROF_DIR        - root of model_prof checkout
  REPORT_PREFIX   - base path for the rank-0 .nsys-rep / .asysrep
Forwarded by accelerate / torchrun:
  LOCAL_RANK
"""
import os
import sys


def main():
    rank = os.environ.get("LOCAL_RANK", "0")
    bench_py = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bench.py"
    )

    if rank == "0":
        prof_dir = os.environ.get("PROF_DIR")
        report_prefix = os.environ.get("REPORT_PREFIX")
        if not prof_dir or not report_prefix:
            sys.stderr.write(
                "ERROR: rank-0 requires PROF_DIR and REPORT_PREFIX env vars\n"
            )
            sys.exit(1)
        prof_sh = os.path.join(prof_dir, "model_prof", "tool", "prof.sh")
        if not os.access(prof_sh, os.X_OK):
            sys.stderr.write(f"ERROR: not executable: {prof_sh}\n")
            sys.exit(1)
        cmd = [
            prof_sh,
            report_prefix,
            sys.executable, bench_py, *sys.argv[1:],
        ]
    else:
        # Non-leader ranks: just run bench.py, no profiler wrap.
        cmd = [sys.executable, bench_py, *sys.argv[1:]]

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
