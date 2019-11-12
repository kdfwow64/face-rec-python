#! /usr/bin/env python

import logging
from logging import handlers
import time
from selenium import webdriver
import socket

REMOTE_SERVER = "www.google.com"
URL = 'http://192.168.1.1/'
PASS = 'admin'
MAX_FAILS = 10
TEST_INTERVAL_SEC = 60


def reboot_modem():
    driver = webdriver.Chrome('/home/user/Downloads/chromedriver')
    try:
        driver.get(URL)
        # settings
        driver.find_element_by_class_name('settings').click()

        # login
        driver.find_element_by_class_name('ipt-login').send_keys(PASS)
        driver.find_element_by_id('f_submit_login').click()
        # reboot
        driver.get(
            f'{URL}/default.html?version=2018-07-20-15-26#settings/systemSetting.html')
        driver.find_element_by_id('btnReboot').click()
        driver.find_element_by_id('btnPopUpOk').click()
        time.sleep(2)
    finally:
        driver.quit()


def is_internet_available(hostname):
    try:
        # see if we can resolve the host name -- tells us if there is
        # a DNS listening
        host = socket.gethostbyname(hostname)
        # connect to the host -- tells us if the host is actually
        # reachable
        s = socket.create_connection((host, 80), timeout=30)
        s.close()
        return True
    except Exception:
        pass
    return False


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "face-id-network-connectivity:%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    syslog_handler = handlers.SysLogHandler(address='/dev/log')
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)


if __name__ == '__main__':
    configure_logging()
    logging.info('started')
    n_tries = 0
    while True:
        try:
            if not is_internet_available(REMOTE_SERVER):
                logging.warning(f'test fail {n_tries}')
                n_tries += 1
            else:
                logging.info('test ok')
                n_tries = 0
            if n_tries == MAX_FAILS:
                n_tries = 0
                logging.error('test max failures, rebooting modem!!')
                reboot_modem()
            time.sleep(TEST_INTERVAL_SEC)
        except Exception:
            pass
