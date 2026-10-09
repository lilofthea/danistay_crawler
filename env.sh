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
