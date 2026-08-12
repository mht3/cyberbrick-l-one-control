# -*-coding:utf-8-*-
#
# The CyberBrick Codebase License, see the file LICENSE for details.
#
# Copyright (c) 2025 MakerWorld
#
# This file is executed on every boot (including wake-boot from deepsleep)

import bbl_product
import sys
import uos

_PRODUCT_NAME = "RC"
_PRODUCT_VERSION = "01.00.01.01"

bbl_product.set_app_name(_PRODUCT_NAME)
bbl_product.set_app_version(_PRODUCT_VERSION)
del bbl_product
del _PRODUCT_NAME
del _PRODUCT_VERSION

# If /wifi_mode.flag exists, boot straight into wifi_bridge.py instead of the
# stock RC firmware -- this gives wifi_bridge a virgin radio, since rc_main
# (via rc_module/ESP-NOW) never gets a chance to touch it first. The flag is
# set either by virtual_gripper.py over USB, or by rc_main.py itself (see
# _wifi_fallback_watchdog) after NO_PAIRING_FALLBACK_TIMEOUT seconds with no
# transmitter paired -- in both cases followed by a real reset (not a soft
# Ctrl-D reset) so this boot's radio is clean. To go back to normal
# remote-control operation, delete the flag file and reset (wifi_bridge.py's
# RESET command does this automatically).
WIFI_MODE_FLAG = '/wifi_mode.flag'

try:
    uos.stat(WIFI_MODE_FLAG)
    wifi_mode = True
except OSError:
    wifi_mode = False

sys.path.append('/app')  # control.mpy (used by both branches) lives here

if wifi_mode:
    import wifi_bridge
    wifi_bridge.main()
else:
    import rc_main

    rc_main.main()
