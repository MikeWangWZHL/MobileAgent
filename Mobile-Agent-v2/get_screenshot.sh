#!/bin/bash

# Define the ADB path
ADB_PATH="/Users/wangz3/Desktop/vlm_agent_project/platform-tools/adb"

# Define the local filename for the screenshot
LOCAL_FILE="screenshot_$(date +%Y%m%d_%H%M%S).png"

# Define the temporary file path on the Android device
DEVICE_FILE="/sdcard/screenshot.png"

# Check if the specified adb path exists
if [ ! -x "$ADB_PATH" ]; then
    echo "Error: ADB not found or not executable at $ADB_PATH"
    exit 1
fi

# Capture the screenshot on the device
echo "Capturing screenshot on the Android device..."
"$ADB_PATH" shell screencap -p "$DEVICE_FILE"

# Check if the screenshot command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to capture screenshot on the device."
    exit 1
fi

# Pull the screenshot to the local computer
echo "Transferring screenshot to local computer..."
"$ADB_PATH" pull "$DEVICE_FILE" "$LOCAL_FILE"

# Check if the pull command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to transfer screenshot to local computer."
    exit 1
fi

# Remove the screenshot from the device
echo "Removing screenshot from the Android device..."
"$ADB_PATH" shell rm "$DEVICE_FILE"

# Notify the user of success
echo "Screenshot saved to $LOCAL_FILE"