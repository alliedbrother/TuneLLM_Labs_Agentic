"""SSH-based remote agent deployment with automatic reverse tunnel.

Deploys the FULL TuneLLM agent (with DirectRunner for job execution)
and training script to remote GPU instances.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SSH_KEY_DIR = Path.home() / ".tunellm" / "ssh"
_active_tunnels: dict[str, asyncio.subprocess.Process] = {}


def get_or_create_ssh_key() -> tuple[str, str]:
    """Get or create an SSH key pair for TuneLLM."""
    SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
    private_key = SSH_KEY_DIR / "tunellm_rsa"
    public_key = SSH_KEY_DIR / "tunellm_rsa.pub"
    if not private_key.exists():
        os.system(f'ssh-keygen -t rsa -b 4096 -f {private_key} -N "" -C "tunellm-agent" -q')
    return str(private_key), public_key.read_text().strip()


async def upload_ssh_key_to_vastai(api_key: str, public_key: str) -> bool:
    """Upload our SSH public key to Vast.ai account."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get("https://console.vast.ai/api/v0/ssh/", headers={"Authorization": f"Bearer {api_key}"})
            existing = resp.json()
            if isinstance(existing, dict):
                existing = existing.get("ssh_keys", [])
            for k in existing:
                if "tunellm-agent" in (k.get("public_key", "") if isinstance(k, dict) else str(k)):
                    return True
            resp = await client.put("https://console.vast.ai/api/v0/ssh/", headers={"Authorization": f"Bearer {api_key}"}, json={"ssh_key": public_key})
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"SSH key upload failed: {e}")
        return False


def _get_ssh_keys() -> list[str]:
    """Get list of SSH key paths to try."""
    keys = []
    tunellm_key = SSH_KEY_DIR / "tunellm_rsa"
    if tunellm_key.exists():
        keys.append(str(tunellm_key))
    for name in ["id_rsa", "id_ed25519", "id_ecdsa"]:
        p = Path.home() / ".ssh" / name
        if p.exists() and str(p) not in keys:
            keys.append(str(p))
    return keys


async def start_reverse_tunnel(
    ssh_host: str, ssh_port: int,
    local_port: int = 8000, remote_port: int = 8000,
    instance_id: str = "",
) -> bool:
    """Start a persistent SSH reverse tunnel in the background."""
    await stop_tunnel(instance_id)

    keys = _get_ssh_keys()
    if not keys:
        logger.error("No SSH keys available for tunnel")
        return False

    for key_path in keys:
        try:
            # First, kill any existing tunnel on the remote port
            kill_cmd = (
                f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
                f"-o BatchMode=yes -i {key_path} -p {ssh_port} root@{ssh_host} "
                f"'fuser -k {remote_port}/tcp 2>/dev/null; echo ok'"
            )
            kill_proc = await asyncio.create_subprocess_shell(
                kill_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(kill_proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                kill_proc.kill()

            await asyncio.sleep(1)

            # Now start the tunnel
            cmd = (
                f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
                f"-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "
                f"-i {key_path} -N "
                f"-R {remote_port}:localhost:{local_port} "
                f"-p {ssh_port} root@{ssh_host}"
            )
            logger.info(f"Starting tunnel: {ssh_host}:{ssh_port} (key={key_path})")
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.sleep(4)

            if proc.returncode is not None:
                stderr = (await proc.stderr.read()).decode().strip()
                logger.warning(f"Tunnel exited immediately (key={key_path}): {stderr}")
                continue

            _active_tunnels[instance_id] = proc
            logger.info(f"Tunnel established: {ssh_host}:{ssh_port} using {key_path}")
            asyncio.create_task(_tunnel_watchdog(instance_id, ssh_host, ssh_port, local_port, remote_port, key_path))
            return True
        except Exception as e:
            logger.warning(f"Tunnel attempt failed (key={key_path}): {e}")
            continue

    logger.error(f"All tunnel attempts failed for {ssh_host}:{ssh_port}")
    return False


async def _tunnel_watchdog(instance_id, ssh_host, ssh_port, local_port, remote_port, key_path):
    """Auto-restart tunnel forever with exponential backoff. Never gives up."""
    import time
    restarts = 0
    backoff = 5
    last_stable = time.time()

    while True:
        proc = _active_tunnels.get(instance_id)
        if proc is None:
            return  # Tunnel was intentionally stopped

        await proc.wait()
        restarts += 1

        # Reset backoff if tunnel was stable for 5+ minutes
        if time.time() - last_stable > 300:
            backoff = 5

        logger.warning(f"Tunnel to {instance_id} dropped (restart #{restarts}), retrying in {backoff}s...")
        await asyncio.sleep(backoff)

        # Kill stale port binding on remote before reconnecting
        kill_cmd = (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-o BatchMode=yes -i {key_path} -p {ssh_port} root@{ssh_host} "
            f"'fuser -k {remote_port}/tcp 2>/dev/null; echo ok'"
        )
        kill_proc = await asyncio.create_subprocess_shell(kill_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(kill_proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            kill_proc.kill()
        await asyncio.sleep(1)

        cmd = (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
            f"-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "
            f"-i {key_path} -N "
            f"-R {remote_port}:localhost:{local_port} "
            f"-p {ssh_port} root@{ssh_host}"
        )
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await asyncio.sleep(4)

        if proc.returncode is None:
            _active_tunnels[instance_id] = proc
            last_stable = time.time()
            backoff = 5  # Reset on success
            logger.info(f"Tunnel to {instance_id} restarted (#{restarts})")
        else:
            # Increase backoff, cap at 60s
            backoff = min(backoff * 2, 60)


async def stop_tunnel(instance_id: str):
    """Stop an active tunnel."""
    proc = _active_tunnels.pop(instance_id, None)
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()


async def deploy_training_script(ssh_host: str, ssh_port: int, timeout: int = 15) -> bool:
    """SCP the training script to the remote instance."""
    script_candidates = [
        Path("/app/training/scripts/train_unsloth.py"),
        Path(__file__).parent.parent.parent.parent / "training" / "scripts" / "train_unsloth.py",
    ]
    script_path = None
    for c in script_candidates:
        if c.exists():
            script_path = c
            break
    if not script_path:
        return False

    for key_path in _get_ssh_keys():
        try:
            cmd = f"scp -o StrictHostKeyChecking=no -o ConnectTimeout={timeout} -o BatchMode=yes -i {key_path} -P {ssh_port} {script_path} root@{ssh_host}:/workspace/tunellm-agent/train_unsloth.py"
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False


async def deploy_agent_via_ssh(
    ssh_host: str,
    ssh_port: int,
    agent_api_key: str,
    timeout: int = 15,
) -> dict:
    """Deploy the FULL TuneLLM agent package to a remote GPU instance.

    Copies all agent Python files + training script, installs deps, starts the agent.
    The agent handles job polling, dataset download, training execution, and progress reporting.
    """
    keys = _get_ssh_keys()
    if not keys:
        return {"status": "failed", "message": "No SSH keys available"}

    # Find agent source files — try multiple locations
    agent_dir = None
    training_script = None
    for candidate in [
        Path(__file__).parent.parent.parent.parent / "agent" / "agent",
        Path("/app/agent/agent"),
        Path("/host/Documents/Thesis/TuneLLM/agent/agent"),
    ]:
        if candidate.exists() and any(candidate.glob("*.py")):
            agent_dir = candidate
            break
    for candidate in [
        Path(__file__).parent.parent.parent.parent / "training" / "scripts" / "train_unsloth.py",
        Path("/app/training/scripts/train_unsloth.py"),
        Path("/host/Documents/Thesis/TuneLLM/training/scripts/train_unsloth.py"),
    ]:
        if candidate.exists():
            training_script = candidate
            break
    logger.info(f"Agent dir: {agent_dir}, Training script: {training_script}")

    last_error = None
    ssh_opts = f"-o StrictHostKeyChecking=no -o ConnectTimeout={timeout} -o BatchMode=yes"

    for key_path in keys:
        opts = f"{ssh_opts} -i {key_path}"
        try:
            # 1. Create remote dirs
            cmd = f"ssh {opts} -p {ssh_port} root@{ssh_host} 'mkdir -p /workspace/tunellm-agent/agent /workspace/data/datasets /workspace/models'"
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode != 0:
                last_error = stderr.decode().strip()
                continue

            # 2. SCP agent package
            if agent_dir.exists():
                for f in agent_dir.glob("*.py"):
                    cmd = f"scp {opts} -P {ssh_port} {f} root@{ssh_host}:/workspace/tunellm-agent/agent/"
                    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await asyncio.wait_for(proc.communicate(), timeout=timeout)

            # 3. SCP training script
            if training_script.exists():
                cmd = f"scp {opts} -P {ssh_port} {training_script} root@{ssh_host}:/workspace/tunellm-agent/train_unsloth.py"
                proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await asyncio.wait_for(proc.communicate(), timeout=timeout)

            # 4. Create and upload startup script
            startup = f"""#!/bin/bash
pip install -q httpx psutil click rich pydantic-settings python-dotenv 2>/dev/null

export TUNELLM_AGENT_SERVER_URL='http://localhost:8000'
export TUNELLM_AGENT_API_KEY='{agent_api_key}'
export TUNELLM_AGENT_EXECUTION_MODE='direct'
export TUNELLM_AGENT_TRAINING_SCRIPT='/workspace/tunellm-agent/train_unsloth.py'
export TUNELLM_AGENT_DATA_PATH='/workspace/data'
export TUNELLM_AGENT_MODEL_PATH='/workspace/models'

pkill -f 'agent.main' 2>/dev/null; pkill -f 'run_agent' 2>/dev/null; pkill -f 'agent_runner' 2>/dev/null; sleep 1

# Create a restart-loop runner script (using single quotes to avoid f-string issues)
cat > /workspace/tunellm-agent/agent_runner.sh << 'RUNNER'
#!/bin/bash
RESTART_DELAY=5
cd /workspace/tunellm-agent
while true; do
    echo "Starting TuneLLM agent..."
    python3 -m agent.main --server-url http://localhost:8000 --api-key "$TUNELLM_AGENT_API_KEY" --node-name tunellm-gpu 2>&1 | tee -a /workspace/agent.log
    echo "Agent exited. Restarting in $RESTART_DELAY seconds..."
    sleep $RESTART_DELAY
    RESTART_DELAY=$((RESTART_DELAY < 30 ? RESTART_DELAY + 5 : 30))
done
RUNNER
chmod +x /workspace/tunellm-agent/agent_runner.sh

export TUNELLM_AGENT_API_KEY='{agent_api_key}'
nohup bash /workspace/tunellm-agent/agent_runner.sh > /workspace/agent_runner.log 2>&1 &
echo "Agent started with auto-restart (PID=$!)"
"""
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(startup)
                startup_path = f.name

            cmd = f"scp {opts} -P {ssh_port} {startup_path} root@{ssh_host}:/workspace/tunellm-agent/start_agent.sh"
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await asyncio.wait_for(proc.communicate(), timeout=timeout)
            os.unlink(startup_path)

            # 5. Run startup
            cmd = f"ssh {opts} -p {ssh_port} root@{ssh_host} 'bash /workspace/tunellm-agent/start_agent.sh'"
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 15)
            output = stdout.decode().strip()

            if proc.returncode == 0 or "Agent started" in output:
                logger.info("Full agent deployed successfully")
                return {"status": "success", "message": output[-200:]}
            else:
                last_error = f"Startup failed: {output}"
                continue

        except asyncio.TimeoutError:
            last_error = f"Timed out with key {key_path}"
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return {"status": "failed", "message": f"Deploy failed: {last_error}"}
