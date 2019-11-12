#!/bin/bash

if pgrep -fla 'python uiface.py'; then
  logger "face-id: stopping"
  sudo pkill -f  'python uiface.py'
fi
