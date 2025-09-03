# Add the current user to the docker group
sudo usermod -aG docker $USER

# Activate the docker group in the current shell session
exec newgrp docker

# Verify Docker works without sudo
docker ps

# Verify Docker Compose works
docker compose version

docker network prune -f
docker network ls --format '{{.Name}}' | grep '^hello-world___' | xargs -r -n1 docker network rm

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "default-address-pools": [
    {"base": "10.200.0.0/16", "size": 24}
  ]
}
EOF
sudo systemctl restart docker


# Per session launch
# 1) Enter the project
cd /home/ubuntu/tgriggs/SkyRL/skyrl-train

# 2) Make sure the docker group is active in THIS shell
exec newgrp docker

# 3) Clean old Ray (safe if none running)
uv run --extra vllm --extra litellm -- ray stop --force || true

# 4) Launch tbench (adjust WANDB/API as needed)
bash examples/tbench/run_tbench.sh

# Optional sanity check
cd /home/ubuntu/tgriggs/SkyRL/skyrl-train
uv run --extra vllm --extra litellm -- python - <<'PY'
import ray, subprocess, json
ray.init()
@ray.remote
def f():
    return {
        'id': subprocess.getoutput('id'),
        'sock': subprocess.getoutput('ls -l /var/run/docker.sock'),
        'docker_ps': subprocess.getoutput('docker ps 2>&1 | head -n 3')
    }
print(json.dumps(ray.get(f.remote()), indent=2))
ray.shutdown()
PY