# Central pipeline on the home machine (10.150.41.5)
export REDPANDA_BROKERS="10.150.41.5:19092"
export REDIS_HOST="10.150.41.5"
export DANISTAY_DATA="$DIR/data"

# Use the kit's own virtualenv if present (created by start.sh or manually)
if [ -x "$DIR/.venv/bin/python3" ]; then
  export PATH="$DIR/.venv/bin:$PATH"
fi
