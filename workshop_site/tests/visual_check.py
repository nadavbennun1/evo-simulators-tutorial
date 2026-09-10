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
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

SITE = Path(__file__).resolve().parents[1]


def main() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    options = Options(); options.add_argument("-headless")
    driver = webdriver.Firefox(options=options, service=Service("/snap/bin/geckodriver"))
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for width, height, label in ((1440, 1000, "desktop"), (768, 1024, "tablet"), (390, 844, "mobile")):
            driver.set_window_size(width, height)
            for page in ("index", "evolution", "sbi"):
                driver.get(f"{base}/{page}.html"); time.sleep(1.5)
                if page != "index":
                    WebDriverWait(driver, 8).until(
                        lambda d: d.execute_script("return document.querySelectorAll('mjx-container').length") > 0
                    )
                overflow = driver.execute_script("return document.documentElement.scrollWidth > document.documentElement.clientWidth")
                if overflow:
                    offenders = driver.execute_script("return [...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1).slice(0,12).map(e => [e.tagName, e.id, e.className, Math.round(e.getBoundingClientRect().right), Math.round(e.getBoundingClientRect().width)])")
                    raise AssertionError(f"horizontal overflow on {page} at {width}px: {offenders}")
                assert driver.execute_script("return document.querySelectorAll('math,mjx-container').length") > (0 if page != "index" else -1)
                driver.save_screenshot(f"/tmp/workshop-{page}-{label}.png")
        driver.set_window_size(1200, 900); driver.get(f"{base}/sbi.html"); time.sleep(2)
        driver.find_element("id", "cell-fa1ab176").screenshot("/tmp/workshop-bayes-equation.png")
        abc_framework = driver.find_element("css selector", ".abc-framework-figure")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", abc_framework)
        abc_image = driver.find_element("css selector", ".abc-framework-figure img")
        WebDriverWait(driver, 10).until(lambda d: abc_image.get_property("naturalWidth") > 0)
        assert abc_framework.size["height"] > 180 and abc_image.is_displayed()
        abc_framework.screenshot("/tmp/workshop-abc-framework.png")
        inverse_example = driver.find_element("id", "inverse-example")
        inverse_example.screenshot("/tmp/workshop-inverse-example.png")
        prediction_box = driver.find_element("id", "posterior-predictions")
        diversity_image = driver.find_element("css selector", "#posterior-predictions img")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", diversity_image)
        time.sleep(.5)
        WebDriverWait(driver, 10).until(lambda d: diversity_image.get_property("naturalWidth") > 0)
        prediction_box.screenshot("/tmp/workshop-posterior-predictions.png")
        driver.execute_script("window.scrollTo(0, arguments[0].offsetTop - 90)", prediction_box)
        driver.save_screenshot("/tmp/workshop-posterior-predictions-top.png")
        outline_finale = driver.find_element("css selector", ".chapter-walkthrough .story-slide:nth-child(4)")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", outline_finale)
        outline_finale.screenshot("/tmp/workshop-sbi-outline-finale.png")
        for number in (2, 3):
            slide = driver.find_element("css selector", f".chapter-walkthrough .story-slide:nth-child({number})")
            slide.screenshot(f"/tmp/workshop-sbi-outline-{number}.png")
        for station in ("guess-parameter", "training-viewer", "zhou-schedule-designer", "collective-outlier-lab", "ppc-detective"):
            element = driver.find_element("id", station); driver.execute_script("arguments[0].scrollIntoView({block:'start'})", element); time.sleep(.5)
            element.screenshot(f"/tmp/workshop-{station}.png")
        assert driver.execute_script("return document.querySelector('#passage-grid input[value=\"0\"]').disabled")
        assert driver.execute_script("return document.querySelectorAll('#passage-grid input').length") == 13
        assert driver.execute_script("return document.querySelectorAll('#passage-grid input:checked').length") == 1
        assert driver.find_element("id", "abc-summary").text == "Choose a simulation budget and acceptance quantile, then click Run ABC."
        assert driver.find_element("id", "abc-progress").get_attribute("value") == "0"
        driver.execute_script("document.querySelector('#abc-sims').value='10000'")
        driver.find_element("id", "abc-run").click()
        time.sleep(.4)
        assert not driver.find_element("id", "abc-summary").text.startswith("Complete:")
        WebDriverWait(driver, 15).until(lambda d: d.find_element("id", "abc-summary").text.startswith("Complete:"))
        assert "500/10000 simulations accepted" in driver.find_element("id", "abc-summary").text
        assert driver.find_element("id", "abc-progress").get_attribute("value") == "10000"
        driver.execute_script("document.querySelector('#coll-epsilon').value='auto:0.95'; document.querySelector('#coll-epsilon').dispatchEvent(new Event('change')); document.querySelector('#contam-strength').value='0'; document.querySelector('#contam-strength').dispatchEvent(new Event('input'))")
        neutral_summary = driver.find_element("id", "collective-summary").text
        neutral_posterior = driver.execute_script("return document.querySelector('#collective-posterior-canvas').toDataURL()")
        neutral_trajectory = driver.execute_script("return document.querySelector('#collective-trajectory-canvas').toDataURL()")
        driver.execute_script("document.querySelector('#contam-strength').value='1.5'; document.querySelector('#contam-strength').dispatchEvent(new Event('input'))")
        displaced_summary = driver.find_element("id", "collective-summary").text
        assert driver.find_element("id", "coll-investigate").get_attribute("value") == "6"
        assert "R7 center (-0.90, -5.00, -5.50)" in neutral_summary
        assert "R7 center (-0.27, -3.05, -7.60)" in displaced_summary
        assert neutral_summary != displaced_summary
        assert neutral_posterior != driver.execute_script("return document.querySelector('#collective-posterior-canvas').toDataURL()")
        assert neutral_trajectory != driver.execute_script("return document.querySelector('#collective-trajectory-canvas').toDataURL()")
        driver.execute_script("document.querySelector('#coll-epsilon').value='-10'; document.querySelector('#coll-epsilon').dispatchEvent(new Event('change'))")
        assert "log₁₀ ε = -10.000" in driver.find_element("id", "coll-epsilon-value").text
        assert "Fixed floor." in driver.find_element("id", "collective-summary").text
        driver.find_element("css selector", '#ppc-cases button[data-case="0"]').click()
        driver.find_element("css selector", 'input[name="diagnosis"][value="well-specified"]').click()
        driver.find_element("id", "ppc-reveal").click()
        assert driver.find_element("id", "ppc-summary").text.startswith("Correct.")
        driver.find_element("id", "chuong-score").click(); time.sleep(.2)
        score = driver.find_element("id", "chuong-score-card").text
        assert "points" in score and "Parameter RMSE" in score
        assert driver.execute_script("return document.querySelectorAll('#chuong-score-card .score-breakdown span').length") == 3
        driver.find_element("id", "chuong-parameter-challenge").screenshot("/tmp/workshop-chuong-scored.png")
        driver.get(f"{base}/evolution.html"); time.sleep(2)
        lauer_primer = driver.find_element("css selector", ".experimental-primer-figure")
        lauer_image = driver.find_element("css selector", ".experimental-primer-figure img")
        WebDriverWait(driver, 10).until(lambda d: lauer_image.get_property("naturalWidth") == 1280)
        assert lauer_image.size["width"] <= lauer_primer.size["width"]
        lauer_primer.screenshot("/tmp/workshop-lauer-experimental-primer.png")
        assert driver.execute_script("return document.querySelectorAll('#avecilla-model-builder [data-force-panel]:not([hidden])').length") == 0
        assert driver.execute_script("return document.querySelectorAll('#avecilla-model-builder [data-model-force].active').length") == 0
        driver.find_element("id", "cell-6ec896e6").screenshot("/tmp/workshop-continuous-chemostat-equations.png")
        effective_size = driver.find_element("id", "effective-population-size")
        driver.execute_script("arguments[0].scrollIntoView({block:'start'})", effective_size)
        effective_size.screenshot("/tmp/workshop-effective-population-size.png")
        chemostat_ne = driver.find_element("id", "chemostat-ne-simulator")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", chemostat_ne)
        driver.find_element("id", "chemostat-ne-run").click()
        WebDriverWait(driver, 8).until(lambda d: d.find_element("id", "chemostat-ne-summary").text.startswith("900 of 900"))
        assert "3 · variance match" in driver.find_element("id", "chemostat-ne-calculation").text.lower()
        chemostat_ne.screenshot("/tmp/workshop-chemostat-ne-simulation.png")
        serial_ne = driver.find_element("id", "serial-ne-simulator")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", serial_ne)
        driver.find_element("id", "serial-ne-run").click()
        WebDriverWait(driver, 8).until(lambda d: d.find_element("id", "serial-ne-summary").text.startswith("After 6 doublings"))
        assert driver.execute_script("return document.querySelectorAll('#serial-ne-generations article.revealed').length") == 6
        serial_ne.screenshot("/tmp/workshop-serial-ne-simulation.png")
        assert not driver.find_element("id", "evo-composition").text
        assert "starts empty" in driver.find_element("id", "evo-summary").text
        driver.find_element("id", "evo-play").click(); time.sleep(.4)
        driver.find_element("id", "evo-play").click()
        assert driver.find_element("id", "evo-composition").text
        driver.find_element("css selector", '[data-evo-preset="order"]').click()
        driver.find_element("id", "evo-play").click(); time.sleep(.5)
        driver.find_element("id", "evo-play").click()
        assert "percentage points" in driver.find_element("id", "evo-order-summary").text
        for station in ("evolution-playground", "dfe-example", "chuong-standing-variation", "zhou-model-playground"):
            element = driver.find_element("id", station); driver.execute_script("arguments[0].scrollIntoView({block:'start'})", element); time.sleep(.4)
            if station == "chuong-standing-variation":
                assert "starts empty" in driver.find_element("id", "chuong-phi-summary").text
                driver.find_element("id", "chuong-phi-play").click(); time.sleep(.6)
                driver.find_element("id", "chuong-phi-play").click()
                assert "Generation" in driver.find_element("id", "chuong-phi-summary").text
            if station == "zhou-model-playground":
                assert "starts empty" in driver.find_element("id", "zhou-model-summary").text
                driver.find_element("id", "zhou-model-play").click(); time.sleep(.7)
                driver.find_element("id", "zhou-model-play").click()
                assert "Passage" in driver.find_element("id", "zhou-model-summary").text
            element.screenshot(f"/tmp/workshop-{station}.png")
        driver.find_element("css selector", '[data-model-force="selection"]').click()
        assert driver.find_element("css selector", '[data-force-panel="selection"]').is_displayed()
        driver.find_element("id", "avecilla-model-builder").screenshot("/tmp/workshop-model-builder.png")
        driver.find_element("css selector", 'section[data-cell-id="efcdf8fa"]').screenshot("/tmp/workshop-avecilla-code.png")
        for step in (1, 2, 3): driver.find_element("css selector", f'[data-chuong-step="{step}"]').click()
        assert driver.find_element("css selector", ".chuong-matrix").is_displayed()
        driver.find_element("id", "chuong-equation-exercise").screenshot("/tmp/workshop-chuong-equations.png")
        driver.execute_script("document.querySelectorAll('[data-code-answer]').forEach(input => input.value = input.dataset.codeAnswer)")
        for button in driver.find_elements("css selector", "[data-check-code-line]"): button.click()
        assert "4 of 4" in driver.find_element("id", "chuong-code-summary").text
        driver.find_element("css selector", ".code-fill-exercise").screenshot("/tmp/workshop-chuong-code.png")
        assert driver.execute_script("return document.querySelectorAll('#zhou-model-canvas').length") == 1
        driver.get(f"{base}/index.html"); time.sleep(1)
        driver.find_element("css selector", ".landing-hero").screenshot("/tmp/workshop-landing-hero.png")
        lesson_cards = driver.find_element("css selector", ".chapter-cards")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", lesson_cards)
        lesson_cards.screenshot("/tmp/workshop-lesson-cards.png")

        # Complete both assessment phases at an iPhone-sized viewport using the
        # tap/keyboard path. Pointer dragging is covered by the manual release
        # checklist because synthetic browser drags do not emulate touch reliably.
        driver.set_window_size(390, 844)
        driver.get(f"{base}/assessment/?phase=pre")
        WebDriverWait(driver, 8).until(lambda d: d.find_elements("css selector", "[data-venue]"))
        driver.execute_script("localStorage.clear()")
        driver.refresh()
        WebDriverWait(driver, 8).until(lambda d: d.find_elements("css selector", "[data-venue]"))
        driver.find_element("css selector", '[data-venue="NYU"]').click()
        driver.find_element("id", "participate").click()

        def assert_mobile_fit():
            assert not driver.execute_script("return document.documentElement.scrollWidth > document.documentElement.clientWidth")

        def place(card, target, keyboard=False):
            card_element = driver.find_element("css selector", f'[data-card="{card}"]')
            target_element = driver.find_element("css selector", f'[data-drop="{target}"]')
            if keyboard:
                card_element.send_keys(Keys.ENTER); target_element.send_keys(Keys.ENTER)
            else:
                card_element.click(); target_element.click()

        def complete_knowledge():
            assert_mobile_fit()
            place("mutation", "supplies_variants", keyboard=True)
            place("selection", "changes_contribution")
            place("drift", "stochastic_trajectories")
            driver.find_element("id", "question-next").click()
            for index, card in enumerate(("mutation", "selection", "drift")): place(card, str(index))
            driver.find_element("id", "question-next").click()
            for slot in ("parameters", "simulator", "simulated", "observed", "compare", "posterior"): place(slot, slot)
            driver.find_element("id", "question-next").click()
            driver.find_element("css selector", '[data-choice="uncertainty"]').click()
            driver.find_element("id", "question-next").click()
            for index, card in enumerate(("prior", "simulate", "compare", "keep")): place(card, str(index))
            driver.find_element("id", "question-next").click()
            driver.find_element("css selector", '[data-choice="middle"]').click()
            driver.find_element("id", "question-next").click()
            assert_mobile_fit()

        complete_knowledge()
        driver.find_element("css selector", '[data-rating="confidence_simulator"][data-value="4"]').click()
        driver.find_element("css selector", '[data-rating="confidence_inverse"][data-value="4"]').click()
        driver.find_element("id", "confidence-next").click()
        WebDriverWait(driver, 8).until(lambda d: "we’ll revisit" in d.find_element("css selector", ".assessment-card h1").text)
        assert not driver.find_elements("css selector", ".result-score")
        assert "saved on this device" in driver.find_element("css selector", ".submission-status").text
        participant_id = driver.execute_script("return localStorage.getItem('evoSbiWorkshopParticipantId')")
        assert participant_id
        driver.save_screenshot("/tmp/workshop-assessment-pre-mobile.png")

        driver.get(f"{base}/assessment/?phase=post")
        WebDriverWait(driver, 8).until(lambda d: d.find_elements("css selector", "[data-venue]"))
        driver.find_element("css selector", '[data-venue="NYU"]').click()
        driver.find_element("id", "participate").click()
        complete_knowledge()
        driver.find_element("css selector", '[data-rating="confidence_simulator"][data-value="5"]').click()
        driver.find_element("css selector", '[data-rating="confidence_inverse"][data-value="5"]').click()
        driver.find_element("id", "confidence-next").click()
        driver.find_element("css selector", '[data-rating="impact_models_data"][data-value="5"]').click()
        driver.find_element("css selector", '[data-rating="research_relevance"][data-value="5"]').click()
        driver.find_element("id", "evaluation-submit").click()
        WebDriverWait(driver, 8).until(lambda d: d.find_elements("css selector", ".result-score"))
        assert driver.find_element("css selector", ".assessment-card h1").text == "6 / 6 concepts"
        assert driver.execute_script("return localStorage.getItem('evoSbiWorkshopParticipantId')") == participant_id
        assert_mobile_fit()
        driver.save_screenshot("/tmp/workshop-assessment-post-mobile.png")
        print("Visual smoke check passed; screenshots written to /tmp/workshop-*.png")
    finally:
        driver.quit(); server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == "__main__": main()
