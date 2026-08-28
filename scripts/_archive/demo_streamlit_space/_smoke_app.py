"""用 streamlit.testing 无头验证 demo/app.py 全部交互链路。
运行：py -3 demo/_smoke_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(Path(__file__).resolve().parent / "app.py"), default_timeout=300)
at.run()
print(f"EXCEPTIONS_AFTER_LOAD={len(at.exception)}")
for e in at.exception:
    print("  ", e.value)

# Tab1: 点击"生成并回测因子"按钮
btns = {b.label: b for b in at.button}
print(f"BUTTONS={list(btns.keys())}")
if "🚀 生成并回测因子" in btns:
    btns["🚀 生成并回测因子"].click()
    at.run()
    print(f"EXCEPTIONS_AFTER_TAB1_CLICK={len(at.exception)}")
    for e in at.exception:
        print("  ", e.value)
    errors = [x.value for x in at.error]
    print(f"ST_ERRORS_AFTER_TAB1={len(errors)}")

# Tab4: 点击"运行精炼厂演示"按钮
btns = {b.label: b for b in at.button}
if "⚙️ 运行精炼厂演示（离线模式）" in btns:
    btns["⚙️ 运行精炼厂演示（离线模式）"].click()
    at.run()
    print(f"EXCEPTIONS_AFTER_TAB4_CLICK={len(at.exception)}")
    for e in at.exception:
        print("  ", e.value)
    errors = [x.value for x in at.error]
    print(f"ST_ERRORS_AFTER_TAB4={len(errors)}")

print("APP_SMOKE_DONE")
