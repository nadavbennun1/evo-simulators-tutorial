#!/usr/bin/env python3
"""Headless Firefox visual smoke check; writes screenshots under /tmp."""
from __future__ import annotations

import functools
import http.server
import threading
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

SITE = Path(__file__).resolve().parents[1]


def main() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    options = Options(); options.add_argument("-headless")
    driver = webdriver.Firefox(options=options, service=Service("/snap/bin/geckodriver"))
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for width, height, label in ((1440, 1000, "desktop"), (768, 1024, "tablet")):
            driver.set_window_size(width, height)
            for page in ("index", "evolution", "sbi"):
                driver.get(f"{base}/{page}.html"); time.sleep(1.5)
                overflow = driver.execute_script("return document.documentElement.scrollWidth > document.documentElement.clientWidth")
                if overflow:
                    offenders = driver.execute_script("return [...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1).slice(0,12).map(e => [e.tagName, e.id, e.className, Math.round(e.getBoundingClientRect().right), Math.round(e.getBoundingClientRect().width)])")
                    raise AssertionError(f"horizontal overflow on {page} at {width}px: {offenders}")
                assert driver.execute_script("return document.querySelectorAll('math').length") > (0 if page != "index" else -1)
                driver.save_screenshot(f"/tmp/workshop-{page}-{label}.png")
        driver.set_window_size(1200, 900); driver.get(f"{base}/sbi.html"); time.sleep(2)
        for station in ("guess-parameter", "training-viewer", "zhou-schedule-designer", "collective-outlier-lab", "ppc-detective"):
            element = driver.find_element("id", station); driver.execute_script("arguments[0].scrollIntoView({block:'start'})", element); time.sleep(.5)
            element.screenshot(f"/tmp/workshop-{station}.png")
        assert driver.execute_script("return document.querySelector('#passage-grid input[value=\"0\"]').disabled")
        assert driver.execute_script("return document.querySelectorAll('#passage-grid input').length") == 13
        assert "ε =" in driver.find_element("id", "abc-summary").text
        driver.execute_script("document.querySelector('#abc-sims').value='10000'")
        driver.find_element("id", "abc-run").click(); time.sleep(.5)
        assert "500/10000 simulations accepted" in driver.find_element("id", "abc-summary").text
        driver.execute_script("document.querySelector('#coll-epsilon').value='-10'; document.querySelector('#coll-epsilon').dispatchEvent(new Event('change'))")
        assert "ε = -10" in driver.find_element("id", "collective-summary").text
        driver.find_element("css selector", 'input[name="diagnosis"][value="well-specified"]').click()
        driver.find_element("id", "ppc-reveal").click()
        assert driver.find_element("id", "ppc-summary").text.startswith("Correct.")
        driver.get(f"{base}/evolution.html"); time.sleep(2)
        assert not driver.find_element("id", "evo-composition").text
        assert "starts empty" in driver.find_element("id", "evo-summary").text
        driver.find_element("id", "evo-play").click(); time.sleep(.4)
        driver.find_element("id", "evo-play").click()
        assert driver.find_element("id", "evo-composition").text
        for station in ("evolution-playground", "dfe-example", "chuong-parameter-challenge", "zhou-model-playground"):
            element = driver.find_element("id", station); driver.execute_script("arguments[0].scrollIntoView({block:'start'})", element); time.sleep(.4)
            element.screenshot(f"/tmp/workshop-{station}.png")
        driver.find_element("id", "chuong-score").click(); time.sleep(.2)
        score = driver.find_element("id", "chuong-score-card").text
        assert "points" in score and "Parameter RMSE" in score
        assert driver.execute_script("return document.querySelectorAll('#chuong-score-card .score-breakdown span').length") == 3
        driver.find_element("id", "chuong-parameter-challenge").screenshot("/tmp/workshop-chuong-scored.png")
        assert driver.execute_script("return document.querySelectorAll('#zhou-model-canvas').length") == 1
        print("Visual smoke check passed; screenshots written to /tmp/workshop-*.png")
    finally:
        driver.quit(); server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == "__main__": main()
