set -x

# Specify environment from Environments Hub in form "org/name@version" (e.g., will/wordle@0.1.4)
if [ -z "$1" ]; then
  echo "Usage: $(basename "$0") <ENV_ID>"
  exit 1
fi
ENV_ID="$1"

verifiers_env_to_uv_flags() {
  local spec="$1"
  local org="${spec%%/*}"
  local rest="${spec#*/}"
  local name="${rest%%@*}"
  local version=""
  if [[ "$rest" == *"@"* ]]; then
    version="${rest##*@}"
  fi
  local index_url="https://hub.primeintellect.ai/${org}/simple/"
  local with_req="$name"
  if [[ -n "$version" ]]; then
    with_req="${name}==${version}"
  fi
  echo "${with_req} --index ${index_url}"
} 

ENV_UV_INSTALL_FLAGS="$(verifiers_env_to_uv_flags "$ENV_ID")"

uv add --active $ENV_UV_INSTALL_FLAGS