from datetime import datetime
import time

def time2stamp(ctime):
    try:
        timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
    except:
        try:
            timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S')
        except:
            timeArray = time.strptime(ctime, '%Y-%m-%d')
    return int(time.mktime(timeArray)) * 1000

def coast_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result
    return fun