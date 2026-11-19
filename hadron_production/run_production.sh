#!/bin/bash
# Usage: ./run_production.sh <pp|pPb|PbPb> [n_threads] [events_per_thread]

set -e

SYSTEM=$1
NTHREADS=${2:-4}
NEVENTS=${3:-50000}

if [ -z "$SYSTEM" ]; then
    echo "Usage: $0 <pp|pPb|PbPb> [n_threads] [events_per_thread]"
    exit 1
fi

CMND="input/${SYSTEM}5020.cmnd"
if [ ! -f "$CMND" ]; then
    echo "Error: $CMND not found"
    exit 1
fi

echo "Production: $SYSTEM"
echo "  config:      $CMND"
echo "  threads:     $NTHREADS"
echo "  events/thread: $NEVENTS"
echo "  total events:  $((NTHREADS * NEVENTS))"
echo ""

mkdir -p logs output

for i in $(seq 1 $NTHREADS); do
    SEED=$((1000 + i * 100))
    echo "  seed $SEED -> output/${SYSTEM}5020_seed${SEED}.root"
    python3 scripts/run_pythia.py -i "$CMND" -n "$NEVENTS" -s "$SEED" \
        > "logs/${SYSTEM}_seed${SEED}.log" 2>&1 &
done

wait

echo ""
echo "Done. Files:"
ls -la output/${SYSTEM}5020_seed*.root
