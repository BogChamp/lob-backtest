import json
import time
import asyncio
import websockets
import os
import urllib.request

DIFF_SOCKET_URI = "wss://stream.binance.com:9443/ws/ethusdt@depth@100ms"
LOB_API_URI = "https://api.binance.com/api/v3/depth?symbol=ETHUSDT&limit=1000"
TRADES_SOCKET_URI = "wss://stream.binance.com:9443/ws/ethusdt@trade"


async def socket(link, seconds=1000):
    messages = []
    async with websockets.connect(link) as websocket:
        start_time = time.time()
        current_time = start_time
        while current_time < start_time + seconds:
            message = await websocket.recv()
            message = json.loads(message)
            messages.append(message)
            current_time = time.time()
    return messages


async def lob_api_call(link):
    message = urllib.request.urlopen(link).read()
    message = json.loads(message)
    return message


async def main():
    async with asyncio.TaskGroup() as tg:
        diffs = tg.create_task(socket(DIFF_SOCKET_URI))
        trades = tg.create_task(socket(TRADES_SOCKET_URI))
        lob = tg.create_task(lob_api_call(LOB_API_URI))

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    with open("./data/init_lob.json", "w", encoding="utf-8") as file_out:
        json.dump(lob.result(), file_out)

    with open("./data/trades.json", "w", encoding="utf-8") as file_out:
        json.dump(trades.result(), file_out)

    with open("./data/diffs.json", "w", encoding="utf-8") as file_out:
        json.dump(diffs.result(), file_out)


asyncio.run(main())
