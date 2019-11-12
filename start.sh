#!/bin/bash

# Globals
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VERSION=$($CURDIR/get_ver.sh)
ENABLE_DISPLAY=0
XVFB_DISPLAY_ID=44

function show_usage() {
  progname=$(basename $0)
  cat <<HEREDOC

   Usage: $progname [--enableui]

   optional arguments:
     -h, --help           show this help message and exit
     -e, --enableui       Enables graphical display
HEREDOC
}

function update_app_version_for_ui() {
  # the ui reads the version str from the file
  echo $VERSION >$CURDIR/version
}

function log() {
  # logs to syslog and stderr
  logger -s "face-id: $1"
}

function start_app() {
#  sudo touch "shutdown.txt"
  log "starting " $VERSION
  eval "$(/home/$USER/anaconda3/bin/conda shell.bash hook)"
  conda activate detector
  cd $CURDIR/src
  if [ $ENABLE_DISPLAY = 0 ]; then
    log "disabled display"
    sudo -E xvfb-run -a --listen-tcp --server-num $XVFB_DISPLAY_ID --auth-file /tmp/xvfb.auth -s "-ac -screen 0 1920x1080x24" $CONDA_PREFIX/bin/python uiface.py
  else
    log "enabled display"
    sudo -E $CONDA_PREFIX/bin/python uiface.py
  fi
}

########## main ###########

# parse cli options
while [[ "$#" -gt 0 ]]; do
  case $1 in
  -e | --enableui)
    ENABLE_DISPLAY=1
    ;;
  -h | --help | -help | --h)
    show_usage
    exit
    ;;
  *)
    show_usage
    exit 1
    ;;
  esac
  shift
done

update_app_version_for_ui
start_app
