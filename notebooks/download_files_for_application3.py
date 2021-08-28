import requests
from tqdm import tqdm


url = 'https://www.dropbox.com/s/edhk8j4dmkxurz7/user_age_X.pkl?dl=0'
dst = '../data/application3/user_age_X.pkl'

proxyDict = None
# Please specify your proxy if needed
# proxy = "127.0.0.1:7890"
# proxyDict = { 
#               "http"  : proxy, 
#               "https" : proxy, 
#               "ftp"   : proxy
#             }

headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
r = requests.get(url, stream=True, headers=headers, proxies=proxyDict)
with open(dst, 'wb') as f:
    for chunk in tqdm(r.iter_content(chunk_size=1024), total=317379): 
        if chunk:
            f.write(chunk)