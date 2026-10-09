# Central pipeline on the home machine (10.150.41.5)
export REDPANDA_BROKERS="10.150.41.5:19093"   # external listener (19092 advertises localhost)
export REDIS_HOST="10.150.41.5"
export DANISTAY_DATA="$DIR/data"
# publisher checkpoint (default points at the central box's home dir)
export LEGAL_STATE_DIR="$DIR/.state"
# labelling pipeline (labeling_pipeline.py): Redpanda for the labelled output.
# Leave empty until the broker/port is decided; labelling runs anyway and the
# backlog is sent once this is set.
export LABELED_BROKERS=""
export LABELED_TOPIC="legal.danistay.labeled"

# Use the kit's own virtualenv if present (created by start.sh or manually).
# Both names are in use (.venv from start.sh, venv/ on older nodes); PY is the
# interpreter the scripts run.
for v in .venv venv; do
  if [ -x "$DIR/$v/bin/python3" ]; then
    export PATH="$DIR/$v/bin:$PATH"
    PY="$DIR/$v/bin/python3"
    break
  fi
done
export PY="${PY:-python3}"
