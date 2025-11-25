#!/bin/bash

mkdir -p logs

for script in scripts/*; do
    [ -f "$script" ] || continue
    name=$(basename "$script")
    echo "Running $name..."
    PYTHONUNBUFFERED=1 "$script" 2>&1 | tee /dev/tty | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tr '\r' '\n' > "logs/${name}.log"
done
