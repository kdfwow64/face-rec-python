import atexit
import logging
from logging import handlers

import os
import subprocess
from glob import glob

CURDIR = os.path.realpath(os.path.dirname(__file__))
XVFB_DISPLAY_ID = 44  # has to be the same as start.sh

# if you change CAM_REC_OUTPUT, make sure to manually clean the prev folder!
CAM_REC_OUTPUT = os.path.join(CURDIR, 'src/data/cam_records')
CAM_REC_FPS = 12  # reduce quality, we don't need a smooth movie
CAM_REC_LEN_SEC = 30  # approximate lengh ~1M/30sec video file
CAM_REC_MAX_RECORDINGS = 25000  # approximate 25GB total storage: ~208 hours


def _get_segment_start_num():
    # calculates the starting record index, to avoid overriding prev records in case of
    # crash or a restart.
    files = glob(f'{CAM_REC_OUTPUT}/*.mp4')
    if not files:
        return 0
    files.sort(key=lambda x: os.path.getmtime(x))
    last_idx = int(os.path.basename(files[-1]).split('.')[0])

    return int((last_idx + 1) % CAM_REC_MAX_RECORDINGS)


def setup_screen_record():
    logging.info("setting up screen recording")
    os.makedirs(CAM_REC_OUTPUT, exist_ok=True)
    segment_start_number = _get_segment_start_num()
    logging.info(f'calculated segment_start_number: %s', segment_start_number)
    # record the screen output to split files, with circular wraparound
    # TODO(haim): run from tmux, to alllow properly quitting ffmpeg (by sending the 'q' key),
    # and avoiding corrupted video on exit.
    cmd = f'ffmpeg -f x11grab' \
          ' -video_size 1920x1080' \
          f' -i :{XVFB_DISPLAY_ID}' \
          ' -codec:v libx264' \
          f' -r {CAM_REC_FPS}' \
          f' -segment_start_number {segment_start_number}' \
          f' -segment_time {CAM_REC_LEN_SEC}' \
          f' -segment_wrap {CAM_REC_MAX_RECORDINGS}' \
          f' -f segment {CAM_REC_OUTPUT}/%06d.mp4'
    logging.info('invoking: %s', cmd)
    p = subprocess.Popen(cmd,
                         shell=True,
                         close_fds=True,
                         stdin=subprocess.PIPE)
    return p


def die_nicely(proc):
    logging.info('stopping ffmpeg')
    proc.stdin.write(b'q')


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "face-id-record-ui:%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    syslog_handler = handlers.SysLogHandler(address='/dev/log')
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)


def main():
    configure_logging()
    ffmpeg_proc = setup_screen_record()
    # FIXME(haim): systemd apparently kills the child process (ffmpeg, and the sighandler
    # doesn't get invoked for some reason)
    atexit.register(die_nicely, ffmpeg_proc)
    ffmpeg_proc.wait()


if __name__ == '__main__':
    main()
