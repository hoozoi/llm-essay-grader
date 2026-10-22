"""
README용 시연 GIF 녹화 → docs/demo.gif
서버를 먼저 띄운 뒤 실행:  python src/server.py  →  python src/record_demo.py
최종 평가용 글 1개를 붙여넣고 채점하는 장면이며 채점 API를 1회 호출한다.
"""

import io
import json
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8600"
ESSAY_ID = "5376054"  # 최종 평가용, 사람 4점
W, H = 1200, 860


def main():
    essay = next(json.loads(l)["text"] for l in open(ROOT / "data/processed/carfree_test.jsonl", encoding="utf-8")
                 if json.loads(l)["id"] == ESSAY_ID)
    frames = []  # (이미지, 표시 시간 ms)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, color_scheme="light")
        pg.goto(URL)
        pg.wait_for_selector("#essay")
        pg.evaluate("document.fonts.ready")
        time.sleep(1.5)

        def snap(ms):
            frames.append((Image.open(io.BytesIO(pg.screenshot())).convert("RGB"), ms))

        snap(1400)
        pg.fill("#essay", essay)
        pg.dispatch_event("#essay", "input")
        pg.evaluate("document.activeElement.blur(); window.scrollTo(0, 0)")
        time.sleep(0.6)
        snap(1800)

        pg.evaluate("document.getElementById('grade').click()")
        time.sleep(0.3)
        snap(700)
        pg.wait_for_selector("#result:not([hidden])", timeout=60000)
        t0 = time.time()
        while time.time() - t0 < 1.4:  # 점수와 동그라미가 그려지는 구간
            snap(90)
        time.sleep(0.6)
        snap(5000)
        b.close()

    imgs = [f.resize((W * 3 // 4, H * 3 // 4), Image.LANCZOS) for f, _ in frames]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
    out = ROOT / "docs" / "demo.gif"
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True)
    print(f"저장: {out} ({out.stat().st_size / 1e6:.1f}MB, {len(frames)}프레임)")


if __name__ == "__main__":
    main()
