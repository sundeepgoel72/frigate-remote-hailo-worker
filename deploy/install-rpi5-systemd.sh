#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/hailo-detectord/app
VENV_DIR=/opt/hailo-detectord/venv
MODEL_DIR=/opt/hailo-detectord/models

sudo useradd --system --create-home --home-dir /opt/hailo-detectord hailo-detectord 2>/dev/null || true
sudo mkdir -p "$APP_DIR" "$MODEL_DIR"
sudo cp -R pyproject.toml README.md src "$APP_DIR"/
sudo cp deploy/hailo-detectord.service /etc/systemd/system/hailo-detectord.service

if [ ! -f /etc/hailo-detectord.env ]; then
  sudo cp deploy/hailo-detectord.env.example /etc/hailo-detectord.env
fi

python3 -m venv --system-site-packages "$VENV_DIR"
sudo "$VENV_DIR/bin/pip" install --upgrade pip
sudo "$VENV_DIR/bin/pip" install "$APP_DIR"

sudo chown -R hailo-detectord:hailo-detectord /opt/hailo-detectord
sudo systemctl daemon-reload
sudo systemctl enable hailo-detectord

cat <<'EOF'
Installed hailo-detectord.

Before starting:
  1. Copy the Frigate-trained .hef into /opt/hailo-detectord/models/
  2. Copy the matching labelmap.txt into /opt/hailo-detectord/models/ if needed
  3. Edit /etc/hailo-detectord.env
  4. Run: sudo systemctl start hailo-detectord
  5. Check: curl http://127.0.0.1:32168/health
EOF
