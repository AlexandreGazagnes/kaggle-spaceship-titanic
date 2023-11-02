from src.imports import *


def now():
    """return str of current time"""

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_")


cab_dict = {i: j for j, i in enumerate(string.ascii_uppercase)}


passthrough = "passthrough"
