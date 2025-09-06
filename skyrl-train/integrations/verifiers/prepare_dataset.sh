# Prepares the dataset for the Verifiers environment and converts it to SkyRL format.
#
# Example:
#   bash integrations/verifiers/prepare_dataset.sh will/wordle@0.1.4
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify environment from Environments Hub in form "org/name@version" (e.g., will/wordle@0.1.4)
if [ -z "$1" ]; then
  echo "Usage: $(basename "$0") <ENV_ID>"
  exit 1
fi
ENV_ID="$1"
DATA_DIR="$HOME/data/$ENV_ID"

uv run --isolated --with verifiers -- python "$SCRIPT_DIR/verifiers_dataset.py" --env_id $ENV_ID --output_dir $DATA_DIR