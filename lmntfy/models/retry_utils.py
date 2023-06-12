import time
from functools import wraps

def retry(n):
    """This decorator can be put on top of a function to let it automatically retry n times on failures"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < n - 1:  # no delay on the last attempt
                        time.sleep(2 ** i if i > 0 else 0)  # instant first retry
                    else:
                        raise e
        return wrapper
    return decorator
