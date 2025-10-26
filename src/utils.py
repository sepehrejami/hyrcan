import time

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
