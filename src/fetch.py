from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from requests.exceptions import Timeout
import time
import traceback
from fake_useragent import UserAgent


# def to_abs_url(base_url="https://commons.wikimedia.org", related_url="/wiki/Category"):
#     return base_url + related_url


ua = UserAgent()
# ua = UserAgent(use_cache_server=False)
header = {"user-agent": ua.chrome}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
def fetch(url):
    try:
        res = requests.get(url, headers=header, timeout=10)
    except Timeout:
        print("Timeout has been raised.")
        return
    except Exception:
        traceback.print_exc()
        return
    time.sleep(1)
    return res


# delete all exif data
# def delete_exif(paths):
#     for path in paths:
#         try:
#             src = Image.open(path)
#             dst = Image.new(src.mode, src.size)
#             dst.putdata(src.getdata())
#             dst.convert("RGB").save(path)
#         except Exception:
#             print(path)
#             traceback.print_exc()
