"""配置落盘（`src/ui/config_persistence.py`）。

只做两件事，都是为了「保存配置」这个按钮不至于反噬用户：

1. **密钥不进 ``config.yaml``。** 该文件受版本控制，写进去的明文密钥迟早会
   被 ``git add -A`` 带进提交里。这里把真值写进 ``.env``（已被 .gitignore
   忽略），``config.yaml`` 只留 ``${VAR}`` 占位符——它本来就这么设计的，
   只是 UI 保存时绕过了这条路。
2. **原地打补丁，不重写整个 YAML。** ``yaml.safe_dump`` 会把 258 行注释、
   段落顺序和引号风格整体抹掉，于是「改一个模型名」产生一份没人能审的
   diff。这里只替换命中的那几行，行尾注释一并保留。

本模块刻意不 import streamlit：纯函数，可直接单测。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# 各 provider 的密钥落到哪个环境变量（与 README「Environment Variables」一致）。
LLM_SECRET_ENV_VAR = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}
LLM_SECRET_ENV_VAR_DEFAULT = "FACTORGPT_LLM_API_KEY"

# data 段里同类的敏感字段。
DATA_SECRET_ENV_VAR = {
    "tushare_token": "TUSHARE_TOKEN",
    "ths_api_token": "THS_API_TOKEN",
}

# 这些 provider 的 key 是占位值而非密钥（Ollama 不校验），明文落盘即可。
PLAINTEXT_PROVIDERS = {"ollama"}

# 与 llm.client.unresolved_env_placeholder 同一套判定，此处不 import 是为
# 了让本模块保持零项目内依赖（llm.client 在导入时就会读 config.yaml）。
_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_KEY_RE_TMPL = r"^{indent}([A-Za-z0-9_.\-]+)\s*:(.*)$"


@dataclass
class SaveResult:
    """一次落盘的结果，供 UI 分别渲染成功与警示。"""

    saved: List[str] = field(default_factory=list)
    warned: List[str] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# 占位符 / 标量序列化
# ----------------------------------------------------------------------
def placeholder_var(value: Any) -> Optional[str]:
    """若 ``value`` 是 ``${VAR}`` 形式的占位符，返回变量名；否则 None。"""
    if value is None:
        return None
    m = _PLACEHOLDER_RE.match(str(value).strip())
    return m.group(1) if m else None


def env_var_for_llm(provider: str) -> str:
    """给出该 provider 的密钥应存放的环境变量名。"""
    return LLM_SECRET_ENV_VAR.get((provider or "").strip().lower(), LLM_SECRET_ENV_VAR_DEFAULT)


def secret_target(value: Any, env_var: str, *, plaintext_ok: bool = False) -> Tuple[str, Optional[str]]:
    """把用户填写的密钥拆成「写进 config.yaml 的值」与「写进 .env 的值」。

    第二个返回值为 None 表示不需要写 .env：空值、仍是占位符（用户没给真值，
    照抄成明文只会把 ``${VAR}`` 本身存进 .env），或它压根不是密钥。
    """
    text = "" if value is None else str(value).strip()
    if plaintext_ok:
        return text, None
    if not text:
        return "", None
    if placeholder_var(text):
        return text, None
    return "${%s}" % env_var, text


def yaml_scalar(value: Any) -> str:
    """把一个 Python 值序列化成 YAML 单行标量。

    ``${VAR}`` 会被 PyYAML 判为合法 plain scalar 而原样输出，但它在读者（和
    别的 YAML 实现）眼里都更像流式映射的开头，所以这里显式加引号：config.yaml
    里本来就是 ``"${DEEPSEEK_API_KEY}"`` 的写法。
    """
    if value is None:
        return "''"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    dumped = yaml.safe_dump(text, allow_unicode=True, default_flow_style=True, width=10**6)
    lines = dumped.splitlines()
    scalar = lines[0] if lines else "''"
    if scalar[:1] not in ("'", '"') and any(c in text for c in "${}[]"):
        scalar = "'" + text.replace("'", "''") + "'"
    return scalar


# ----------------------------------------------------------------------
# 原地 YAML 补丁
# ----------------------------------------------------------------------
def patch_yaml(text: str, path: Sequence[str], updates: Dict[str, Any]) -> str:
    """在 YAML 文本中原地更新 ``path`` 指向的映射，其余内容原样保留。

    ``path`` 形如 ``["llm"]`` 或 ``["data", "neodata"]``；``updates`` 里的键
    若已存在则替换（保留行尾注释），否则追加到该映射末尾。
    """
    lines = text.splitlines()
    _patch_mapping(lines, 0, len(lines), list(path), "", updates)
    out = "\n".join(lines)
    return out + "\n" if text.endswith("\n") else out


def _patch_mapping(lines: List[str], start: int, end: int, path: List[str],
                   indent: str, updates: Dict[str, Any]) -> None:
    if not path:
        _apply_updates(lines, start, end, indent, updates)
        return
    head = path[0]
    idx = _find_key(lines, start, end, indent, head)
    if idx is None:
        child = indent + "  "
        lines[end:end] = [f"{indent}{head}:"] + [
            f"{child}{k}: {yaml_scalar(v)}" for k, v in updates.items()
        ]
        return
    sub_end = _section_end(lines, idx + 1, end, indent)
    sub_indent = _child_indent(lines, idx + 1, sub_end, indent)
    _patch_mapping(lines, idx + 1, sub_end, path[1:], sub_indent, updates)


def _apply_updates(lines: List[str], start: int, end: int, indent: str,
                   updates: Dict[str, Any]) -> None:
    child = _child_indent(lines, start, end, indent)
    pat = re.compile(_KEY_RE_TMPL.format(indent=re.escape(child)))
    remaining = dict(updates)
    for j in range(start, end):
        m = pat.match(lines[j])
        if not m or m.group(1) not in remaining:
            continue
        key = m.group(1)
        new_value = remaining.pop(key)
        # 值没变就一个字节都不动：否则「点一次保存」也会在 diff 里留下引号/格式
        # 变化，而这类噪声正是审计时要花时间排除的东西。
        if _same_scalar(m.group(2), new_value):
            continue
        scalar_text = yaml_scalar(new_value)
        comment = ""
        cm = re.search(r"\s+#(.*)$", m.group(2))
        if cm:
            # 注释列尽量保持原位：换了个更短的值不至于让整列注释左移。
            base = len(child) + len(key) + 1          # 冒号后空格的下标（行内）
            gap_start = base + cm.start()             # 值后空白的起点
            hash_col = gap_start + (len(cm.group(0)) - len(cm.group(1)) - 1)
            padding = max(1, hash_col - (len(child) + len(key) + 2 + len(scalar_text)))
            comment = " " * padding + "#" + cm.group(1).rstrip()
        lines[j] = f"{child}{key}: {scalar_text}{comment}"
    if remaining:
        lines[end:end] = [f"{child}{k}: {yaml_scalar(v)}" for k, v in remaining.items()]


def _same_scalar(raw_value_text: str, new_value: Any) -> bool:
    """判断行内已有取值与待写入值是否等价（类型也须一致）。"""
    cm = re.search(r"\s+#", raw_value_text)
    value_part = (raw_value_text[: cm.start()] if cm else raw_value_text).strip()
    if not value_part:
        return False
    try:
        old = yaml.safe_load(value_part)
    except yaml.YAMLError:
        return False
    return type(old) is type(new_value) and old == new_value


def _find_key(lines: List[str], start: int, end: int, indent: str, key: str) -> Optional[int]:
    pat = re.compile(_KEY_RE_TMPL.format(indent=re.escape(indent)))
    for j in range(start, end):
        m = pat.match(lines[j])
        if m and m.group(1) == key:
            return j
    return None


def _section_end(lines: List[str], start: int, end: int, parent_indent: str) -> int:
    """子段结束位置：下一个缩进不超过父段的有效行。"""
    for j in range(start, end):
        ln = lines[j]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        cur = ln[: len(ln) - len(ln.lstrip())]
        if len(cur) <= len(parent_indent):
            return j
    return end


def _child_indent(lines: List[str], start: int, end: int, parent_indent: str) -> str:
    for j in range(start, end):
        ln = lines[j]
        if ln.strip() and not ln.lstrip().startswith("#"):
            return ln[: len(ln) - len(ln.lstrip())]
    return parent_indent + "  "


# ----------------------------------------------------------------------
# .env 写入
# ----------------------------------------------------------------------
def upsert_env(path: Any, var: str, value: str) -> None:
    """在 .env 中写入/更新 ``var=value``，其余行原样保留。"""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(var)}\s*=")
    hit = False
    for i, ln in enumerate(lines):
        if pat.match(ln):
            prefix = "export " if ln.lstrip().startswith("export ") else ""
            lines[i] = f"{prefix}{var}={_env_value(value)}"
            hit = True
            break
    if not hit:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{var}={_env_value(value)}")
    write_text_atomic(p, "\n".join(lines) + "\n")
    try:
        os.chmod(p, 0o600)  # POSIX 上收紧权限；Windows 无操作
    except OSError:
        pass


def _env_value(value: str) -> str:
    if value == "" or any(c in value for c in ' "#\'\n\\'):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def write_text_atomic(path: Any, text: str) -> None:
    """先写临时文件再 ``os.replace``：避免写到一半崩溃留下半个 config.yaml。"""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


# ----------------------------------------------------------------------
# 对外的两个保存入口
# ----------------------------------------------------------------------
def save_llm_settings(config_path: Any, env_path: Any, values: Dict[str, Any]) -> SaveResult:
    """把侧边栏的 llm 段设置落盘，密钥外置到 .env。"""
    res = SaveResult()
    provider = str(values.get("provider") or "deepseek").lower()
    env_var = env_var_for_llm(provider)
    cfg_key, env_value = secret_target(
        values.get("api_key", ""), env_var, plaintext_ok=provider in PLAINTEXT_PROVIDERS
    )

    if env_value:
        upsert_env(env_path, env_var, env_value)
        # 当前进程立即生效，否则要等下次启动 .env 才会被注入。
        os.environ[env_var] = env_value
        res.env_vars.append(env_var)
        res.saved.append(
            f"API Key 已写入 `.env` 的 `{env_var}`（该文件已被 .gitignore 忽略，不会入库）；"
            f"`config.yaml` 只保留占位符 `${{{env_var}}}`。"
        )
    else:
        var = placeholder_var(cfg_key)
        if var:
            res.warned.append(
                f"API Key 仍是未解析的占位符 `${{{var}}}`：请在 `.env` 或环境变量中设置 "
                f"`{var}`，否则下一次运行会因密钥无效而离线兜底。"
            )

    updates = {k: v for k, v in values.items() if k != "api_key"}
    updates["api_key"] = cfg_key
    text = Path(config_path).read_text(encoding="utf-8")
    write_text_atomic(config_path, patch_yaml(text, ["llm"], updates))
    res.saved.append("`config.yaml` 的 llm 段已原地更新（其余段落与注释保持不变）。")
    return res


def save_data_settings(config_path: Any, env_path: Any, values: Dict[str, Any]) -> SaveResult:
    """把侧边栏的 data 段设置落盘：标量原地替换，嵌套段分别打补丁，token 外置。"""
    res = SaveResult()
    scalars: Dict[str, Any] = {}
    nested: Dict[str, Dict[str, Any]] = {}
    for k, v in (values or {}).items():
        if isinstance(v, dict):
            if v:  # 空 dict 说明该子段当前不适用，跳过以免抹掉已有配置
                nested[k] = v
        else:
            scalars[k] = v

    for field_name, env_var in DATA_SECRET_ENV_VAR.items():
        if field_name not in scalars:
            continue
        cfg_value, env_value = secret_target(scalars[field_name], env_var)
        scalars[field_name] = cfg_value
        if env_value:
            upsert_env(env_path, env_var, env_value)
            os.environ[env_var] = env_value
            res.env_vars.append(env_var)
            res.saved.append(
                f"`{field_name}` 已写入 `.env` 的 `{env_var}`，`config.yaml` 只保留占位符。"
            )
        elif placeholder_var(cfg_value):
            res.warned.append(
                f"`{field_name}` 仍是占位符 `${{{placeholder_var(cfg_value)}}}`："
                f"请在 `.env` 或环境变量中设置 `{env_var}`。"
            )

    text = Path(config_path).read_text(encoding="utf-8")
    if scalars:
        text = patch_yaml(text, ["data"], scalars)
    for sub, upd in nested.items():
        text = patch_yaml(text, ["data", sub], upd)
    write_text_atomic(config_path, text)
    res.saved.append("`config.yaml` 的 data 段已原地更新（其余段落与注释保持不变）。")
    return res
