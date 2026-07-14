#!/usr/bin/env bash
set -euo pipefail
python -m distributed_mcr2.train_iid "$@"
