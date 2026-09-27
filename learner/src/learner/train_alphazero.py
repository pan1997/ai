"""[DEPRECATED] Legacy standalone training entrypoint.

This module is deprecated in favor of `learner.orchestrator`.
It provides backward-compatible forwarding to `ExperimentOrchestrator` with self-play disabled.
"""

import sys
import warnings

from learner.orchestrator import main as orchestrator_main


def main():
    warnings.warn(
        "learner.train_alphazero is deprecated. Please use 'python -m learner.orchestrator' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("[DEPRECATED] 'learner.train_alphazero' is deprecated. Forwarding to 'learner.orchestrator'...")
    if "--no-selfplay" not in sys.argv:
        sys.argv.append("--no-selfplay")
    orchestrator_main()


if __name__ == "__main__":
    main()
