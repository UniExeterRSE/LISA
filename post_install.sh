#!/bin/bash

# Check if the operating system is macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing tensorflow-metal for macOS..."
    pip install tensorflow-metal
else
    echo "Skipping tensorflow-metal installation (not macOS)."
fi