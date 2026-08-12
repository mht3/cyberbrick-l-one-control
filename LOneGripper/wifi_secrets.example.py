AP_SSID = "your-ap-ssid"
AP_PASSWORD = "your-ap-password"
STA_SSID = "your-network-ssid"
STA_PASSWORD = "your-network-password"

# Desktop-side only (virtual_gripper.py / collect_data.py) -- not read by
# the on-device wifi_bridge.py itself.
STA_HOSTNAME = "your-board-hostname-or-ip"  # stable DHCP/DNS hostname (or IP) for STA mode
AP_FIXED_IP = "192.168.4.1"  # ESP32 AP_IF's default address; matches wifi_bridge.py's start_ap()
