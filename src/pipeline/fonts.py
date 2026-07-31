"""CJK 字体探测与注册：保证 PDF / PNG 图表中的中文正常渲染（避免方框/豆腐块）。

matplotlib 在 Windows 上即便系统装有中文字体，也常因字体缓存或 fallback 顺序问题
导致中文回退到 DejaVu Sans Mono 而缺失字形（典型报错：
``Glyph xxxx missing from font(s) DejaVu Sans Mono``）。

本模块显式注册一个可用的中文字体文件，并把 ``sans-serif`` 与 ``monospace`` 两套
fallback 都前置该字体，确保中英文混排（尤其是 ``fig.text(family="monospace")``
这种显式等宽文本）均可见。幂等、可失败降级。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("factor_gpt.fonts")

# (候选字体显示名, 系统文件路径)，按常见度排序；Windows 优先，其次 Linux/macOS
_CJK_FONT_PATHS = [
    ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
    ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
    ("Noto Sans SC", r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    ("Noto Sans SC", r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"),
    ("WenQuanYi Zen Hei", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ("Arial Unicode MS", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]

_REGISTERED_NAME: str = ""


def setup_cjk_font() -> str:
    """注册一个可用的中文字体，返回其字体名（失败返回空串）。幂等可重复调用。"""
    global _REGISTERED_NAME
    if _REGISTERED_NAME:
        return _REGISTERED_NAME
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception as e:  # noqa: BLE001
        logger.debug("matplotlib 不可用，跳过中文字体配置: %s", e)
        return ""

    name = ""
    for disp, path in _CJK_FONT_PATHS:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                fp = font_manager.FontProperties(fname=path)
                real = fp.get_name() or disp
                name = real
                _REGISTERED_NAME = name
                logger.debug("已注册中文字体: %s <- %s", name, path)
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("字体注册失败 %s: %s", path, e)
                continue

    try:
        known = ["Microsoft YaHei", "SimHei", "Noto Sans SC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "Noto Sans CJK SC", "Heiti TC", "Arial Unicode MS"]
        if name:
            known = [name] + known
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = known + ["DejaVu Sans"]
        # 关键：monospace 也前置中文字体，否则 fig.text(family="monospace") 仍会缺字形
        matplotlib.rcParams["font.monospace"] = known + ["DejaVu Sans Mono"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # noqa: BLE001
        logger.debug("设置 rcParams 失败: %s", e)
    return name
