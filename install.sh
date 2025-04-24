#!/bin/bash

# Make sure we have the required packages
echo "Installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv

# Install Python requirements
pip3 install numpy opencv-python

# Create directory for logs
mkdir -p logs

# Make sure the script is executable
chmod +x worker_monitor.py

# Copy the service file to systemd
sudo cp worker_monitor.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable worker_monitor.service

# Start the service
sudo systemctl start worker_monitor.service

echo "Installation complete! Worker monitoring service is now running."
echo "To check status: sudo systemctl status worker_monitor.service"
echo "To stop service: sudo systemctl stop worker_monitor.service"
echo "To start service: sudo systemctl start worker_monitor.service"
echo "Note: Required model files will be automatically downloaded when the service runs."
