#!/bin/bash

bluetoothctl << EOF
power on
connect 2C:FD:B4:19:79:37
exit
EOF