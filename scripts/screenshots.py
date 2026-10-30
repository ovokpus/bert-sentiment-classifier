"""Capture README screenshots from the running stack (docker compose up)."""
from playwright.sync_api import sync_playwright

OUT = "docs/screenshots"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto("http://localhost:8080")
    pg.get_by_text("API online").wait_for(timeout=60000)
    pg.screenshot(path=f"{OUT}/web-home.png", full_page=True)
    pg.get_by_role("button", name="Classify").click()
    pg.locator(".result").first.wait_for(timeout=60000)
    pg.wait_for_timeout(1000)
    pg.screenshot(path=f"{OUT}/web-results.png", full_page=True)
    pg.goto("http://localhost:8000/docs")
    pg.get_by_text("/predict").first.wait_for()
    pg.screenshot(path=f"{OUT}/api-docs.png", full_page=True)
    pg.get_by_text("/predict").first.click()
    pg.get_by_role("button", name="Try it out").click()
    pg.get_by_role("button", name="Execute").click()
    pg.get_by_text("positive_probability").first.wait_for()
    pg.wait_for_timeout(800)
    pg.screenshot(path=f"{OUT}/api-predict.png", full_page=True)
    b.close()
