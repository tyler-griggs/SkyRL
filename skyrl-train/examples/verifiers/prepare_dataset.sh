set -x
# Helper that converts Environment Hub ID to required uv flags
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env_resolver.sh"

# Specify environment from Prime hub in form "org/name@version" (e.g., will/wordle@0.1.4)
ENV_ID="primeintellect/reverse-text"

ENV_UV_INSTALL_FLAGS="$(verifiers_env_to_uv_flags "$ENV_ID")"
uv run --isolated --extra verifiers $ENV_UV_INSTALL_FLAGS -- python verifiers_dataset.py --env_id $ENV_ID