"""Capture a screenshot of the Gradio UI after one prediction (headless Edge/Chrome via CDP).

    python scripts/ui_screenshot.py http://127.0.0.1:7861/ui report/figures/ui_screenshot.png
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


async def capture(url, out, port=9333):
    exe = next(b for b in BROWSERS if os.path.exists(b))
    profile = tempfile.mkdtemp(prefix="hg-shot-")
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}", "--disable-gpu",
                             f"--user-data-dir={profile}", "--window-size=1500,1400", "about:blank"])
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            await call("Emulation.setDeviceMetricsOverride", width=1500, height=1400, deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=url)
            await asyncio.sleep(6)                                    # let Gradio render
            await call("Runtime.evaluate", expression=
                       "[...document.querySelectorAll('button')].find(b => b.innerText.includes('Predict')).click()")
            for _ in range(60):                                       # wait for the result markdown
                await asyncio.sleep(1)
                r = await call("Runtime.evaluate", expression="document.body.innerText.includes('chance of readmission')")
                if r.get("result", {}).get("value"):
                    break
            await asyncio.sleep(1.5)
            h = await call("Runtime.evaluate", expression="document.documentElement.scrollHeight")
            height = min(int(h["result"]["value"]), 2600)
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=height, deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(1)
            shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            with open(out, "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            print(f"saved {out} ({height}px tall)")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(capture(sys.argv[1], sys.argv[2]))
