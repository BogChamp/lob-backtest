import json
import time
import asyncio
import websockets

async def socket(link, seconds=10):
    lobs = []
    async with websockets.connect(link) as websocket:
        start_time = time.time()
        current_time = start_time
        # number = 1
        while current_time < start_time + seconds:
            #local_ts = time.time() * 10**3
            message = await websocket.recv()
            message = json.loads(message)
            lobs.append(message)
            #message['local_ts'] = local_ts
            # with open(f'diffs/diff{number}.json', 'w', encoding='utf-8') as f:
            #     json.dump(message, f, ensure_ascii=False, indent=1)
            # number += 1
            current_time = time.time()
    return lobs

link = 'https://api.binance.com/api/v3/depth?symbol=ETHUSDT&limit=1000'
loop = asyncio.get_event_loop()
task = socket(link)
result = loop.run_until_complete(asyncio.gather(task))
loop.close()

with open('lobs.json', 'w', encoding='utf-8') as file_out:
    json.dump(result[0], file_out)