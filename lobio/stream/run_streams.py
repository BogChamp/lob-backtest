import json
import time
import asyncio
import websockets
import os
import urllib.request
import sys

DIFF_SOCKET_URI = "wss://fstream.binance.com/ws/ethusdt@depth@0ms"
#TRADES_SOCKET_URI = "wss://stream.binance.com:9443/ws/ethusdt@trade"
AGGTRADES_SOCKET_URI = "wss://fstream.binance.com/ws/ethusdt@aggTrade"
#BOOKTICKER_SOCKET_URI = "wss://stream.binance.com:9443/ws/ethusdt@bookTicker"
LOB_API_URI = "https://fapi.binance.com/fapi/v1/depth?symbol=ETHUSDT&limit=1000"
#url = "wss://stream.binance.com:9443/stream?streams=ethusdt@depth@100ms/ethusdt@trade"

async def socket(link, seconds=1200):
    messages = []
    async with websockets.connect(link) as websocket:
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            message = await websocket.recv()
            message = json.loads(message)
            messages.append(message)
    return messages


async def lob_api_call(link):
    await asyncio.sleep(1)
    message = urllib.request.urlopen(link).read()
    message = json.loads(message)
    return message


async def main(dir_name: str):
    async with asyncio.TaskGroup() as tg:
        #tickers = tg.create_task(socket(BOOKTICKER_SOCKET_URI))
        diffs = tg.create_task(socket(DIFF_SOCKET_URI))
        #trades = tg.create_task(socket(TRADES_SOCKET_URI))
        aggtrades = tg.create_task(socket(AGGTRADES_SOCKET_URI))
        #diffs_trades = tg.create_task(socket(url))
        lob = tg.create_task(lob_api_call(LOB_API_URI))

    dir_path = "./" + dir_name
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    with open(dir_path + "/init_lob.json", "w", encoding="utf-8") as file_out:
        json.dump(lob.result(), file_out)

    # with open(dir_path + "/trades.json", "w", encoding="utf-8") as file_out:
    #     json.dump(trades.result(), file_out)
    
    with open(dir_path + "/aggtrades.json", "w", encoding="utf-8") as file_out:
        json.dump(aggtrades.result(), file_out)

    with open(dir_path + "/diffs.json", "w", encoding="utf-8") as file_out:
        json.dump(diffs.result(), file_out)
    
    # with open(dir_path + "/tickers.json", "w", encoding="utf-8") as file_out:
    #     json.dump(tickers.result(), file_out)
    # with open(dir_path + "/diffs_trades.json", "w", encoding="utf-8") as file_out:
    #     json.dump(diffs_trades.result(), file_out)

if len(sys.argv) > 1:
    dir_name = sys.argv[1]
else:
    dir_name = "data"

asyncio.run(main(dir_name=dir_name))
