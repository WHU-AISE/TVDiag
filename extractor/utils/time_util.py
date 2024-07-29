from datetime import datetime
import time


# def time2stamp(cmnttime):   #转时间戳函数
#     cmnttime=datetime.strptime(cmnttime,'%Y-%m-%d %H:%M:%S')
#     stamp=int(datetime.timestamp(cmnttime))
#     return stamp * 1000

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