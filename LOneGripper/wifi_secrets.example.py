AP_SSID = "your-ap-ssid"
AP_PASSWORD = "your-ap-password"
STA_SSID = "your-network-ssid"
STA_PASSWORD = "your-network-password"

# Read by BOTH sides, deliberately: the desktop connects to this name, and
# wifi_bridge.py claims its first label as the board's DHCP hostname, so the name
# you dial and the name the board registers cannot drift apart.
#
# On a network that publishes DHCP clients in DNS, use the fully qualified name
# (e.g. "mpy-esp32c3.dynamic.ucsd.edu"); the board registers "mpy-esp32c3" and DNS
# appends the rest. A bare IP also works, but then nothing is registered and the
# address will change out from under you -- prefer AP mode if there is no usable DNS.
STA_HOSTNAME = "your-board-hostname-or-ip"
AP_FIXED_IP = "192.168.4.1"  # ESP32 AP_IF's default address; matches wifi_bridge.py's start_ap()
