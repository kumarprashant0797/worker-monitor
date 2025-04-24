#!/bin/bash

echo "Installing Worker Activity Monitoring System..."

# Check if virtualenv is installed, if not install it
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    pip3 install virtualenv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install numpy opencv-python

# Create necessary directories
mkdir -p logs models

# Make scripts executable
chmod +x worker_monitor.py
chmod +x select_roi.py

# Generate service file but don't install it
cat > worker_monitor.service << EOF
[Unit]
Description=Worker Activity Monitoring System
After=network.target

[Service]
User=${USER}
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python $(pwd)/worker_monitor.py
Environment="PYTHONUNBUFFERED=1"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Installation complete!"
echo ""
echo "To run the worker monitor manually:"
echo "$ source venv/bin/activate"
echo "$ python worker_monitor.py"
echo ""
echo "To install as a service (optional):"
echo "$ sudo cp worker_monitor.service /etc/systemd/system/"
echo "$ sudo systemctl daemon-reload"
echo "$ sudo systemctl enable worker_monitor.service"
echo "$ sudo systemctl start worker_monitor.service"
echo ""
echo "Note: Required model files will be automatically downloaded when the program runs."
