import datetime


def now():
    """return str of current time"""

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_")
