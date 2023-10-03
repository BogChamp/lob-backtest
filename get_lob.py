import json
import urllib.request
import time


link = 'https://api.binance.com/api/v3/depth?symbol=ETHUSDT&limit=1000'
message = urllib.request.urlopen(link).read()
message = json.loads(message)
with open(f'init_lob.json', 'w', encoding='utf-8') as f:
    json.dump(message, f, ensure_ascii=False, indent=1)