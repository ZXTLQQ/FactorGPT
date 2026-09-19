"""「保存配置」不得反噬用户的两条不变量。

背景：侧边栏的「保存配置」此前把 API Key 明文 ``yaml.safe_dump`` 进
``config.yaml``——一个受版本控制的文件，于是密钥离一次 ``git add -A`` 只差
一步；同一次 dump 还会把文件里几百行注释和段落顺序整体重写，改一个模型名
就产生一份没人能审的 diff。这里把修复后的行为钉住：

1. 密钥真值只进 ``.env``，``config.yaml`` 只留 ``${VAR}``，且插值后仍能取回；
2. 落盘是原地打补丁：注释、其它段落、嵌套子段都不被动。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from llm.client import load_config, unresolved_env_placeholder  # noqa: E402
from ui.config_persistence import (  # noqa: E402
    env_var_for_llm,
    patch_yaml,
    placeholder_var,
    save_data_settings,
    save_llm_settings,
    secret_target,
    upsert_env,
    yaml_scalar,
)

_REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

_SAMPLE = """# 顶层注释
llm:
  # provider 说明
  provider: deepseek               # deepseek / openai / qwen
  api_key: "${DEEPSEEK_API_KEY}"   # 密钥存于 .env
  model: deepseek-chat
  router:
    enabled: false
    critic:
      api_key: "${DEEPSEEK_API_KEY}"
kronos:
  enabled: true
"""

_LLM_VALUES = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "api_key": "sk-test-123456",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.3,
}


# ----------------------------------------------------------------------
# 1) 密钥拆分
# ----------------------------------------------------------------------
def test_secret_target_moves_plaintext_to_env() -> None:
    cfg_value, env_value = secret_target("sk-test-123456", "DEEPSEEK_API_KEY")
    assert cfg_value == "${DEEPSEEK_API_KEY}"
    assert env_value == "sk-test-123456"


def test_secret_target_keeps_placeholder_and_blank() -> None:
    # 用户没给真值：既不能把 "${VAR}" 当明文写进 .env，也不能清空已有占位符
    assert secret_target("${DEEPSEEK_API_KEY}", "DEEPSEEK_API_KEY") == ("${DEEPSEEK_API_KEY}", None)
    assert secret_target("", "DEEPSEEK_API_KEY") == ("", None)
    assert secret_target(None, "DEEPSEEK_API_KEY") == ("", None)


def test_secret_target_allows_plaintext_for_ollama() -> None:
    # Ollama 的 key 是占位值，不是密钥
    assert secret_target("ollama", "FACTORGPT_LLM_API_KEY", plaintext_ok=True) == ("ollama", None)


def test_env_var_per_provider() -> None:
    assert env_var_for_llm("deepseek") == "DEEPSEEK_API_KEY"
    assert env_var_for_llm("openai") == "OPENAI_API_KEY"
    assert env_var_for_llm("qwen") == "DASHSCOPE_API_KEY"
    assert env_var_for_llm("custom") == "FACTORGPT_LLM_API_KEY"
    assert env_var_for_llm("ollama") == "FACTORGPT_LLM_API_KEY"


def test_placeholder_detection_matches_llm_client() -> None:
    # 两处判定必须一致，否则会出现「一边认为是假密钥、一边照常发出去」
    for value in ("${DEEPSEEK_API_KEY}", "  ${A_B_1}  ", "sk-abc", "", None, "pre ${X} post"):
        assert placeholder_var(value) == unresolved_env_placeholder(value)


# ----------------------------------------------------------------------
# 2) 原地 YAML 补丁
# ----------------------------------------------------------------------
def test_patch_updates_value_and_keeps_inline_comment() -> None:
    out = patch_yaml(_SAMPLE, ["llm"], {"provider": "openai"})
    old_line = next(ln for ln in _SAMPLE.splitlines() if ln.strip().startswith("provider:"))
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("provider:"))
    assert line.startswith("  provider: openai")
    # 注释照旧跟着这一行，且换了个更短的值也不该让它左移
    assert line.index("#") == old_line.index("#")


def test_patch_touches_only_the_target_section() -> None:
    out = patch_yaml(_SAMPLE, ["llm"], {"model": "gpt-4o"})
    assert "kronos:" in out and "enabled: true" in out          # 其它顶层段未被动
    assert "# 顶层注释" in out and "# provider 说明" in out      # 注释未被动


def test_patch_does_not_reach_nested_same_named_key() -> None:
    out = patch_yaml(_SAMPLE, ["llm"], {"api_key": "${OPENAI_API_KEY}"})
    doc = yaml.safe_load(out)
    assert doc["llm"]["api_key"] == "${OPENAI_API_KEY}"
    # 缩进更深的 router.critic.api_key 不应被同一个键名误伤
    assert doc["llm"]["router"]["critic"]["api_key"] == "${DEEPSEEK_API_KEY}"


def test_patch_nested_path() -> None:
    out = patch_yaml(_SAMPLE, ["llm", "router"], {"enabled": "true"})
    assert yaml.safe_load(out)["llm"]["router"]["enabled"] == "true"
    assert yaml.safe_load(out)["llm"]["router"]["critic"]["api_key"] == "${DEEPSEEK_API_KEY}"


def test_patch_appends_missing_key() -> None:
    out = patch_yaml(_SAMPLE, ["llm"], {"timeout": 90})
    assert yaml.safe_load(out)["llm"]["timeout"] == 90
    assert out.index("timeout: 90") < out.index("kronos:")  # 追加在本段末尾，未落到别的段落


def test_patch_is_a_noop_when_nothing_changed() -> None:
    out = patch_yaml(_SAMPLE, ["llm"], {"provider": "deepseek", "model": "deepseek-chat"})
    assert out == _SAMPLE


def test_patch_creates_missing_section_at_end() -> None:
    out = patch_yaml(_SAMPLE, ["brand_new"], {"a": 1})
    assert yaml.safe_load(out)["brand_new"] == {"a": 1}


def test_yaml_scalar_forms() -> None:
    assert yaml_scalar("${DEEPSEEK_API_KEY}") == "'${DEEPSEEK_API_KEY}'"
    assert yaml_scalar(0.3) == "0.3"
    assert yaml_scalar(True) == "true"
    assert yaml_scalar("") == "''"


# ----------------------------------------------------------------------
# 3) .env 写入
# ----------------------------------------------------------------------
def test_upsert_env_appends_then_updates_in_place(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("MX_APIKEY=abc\n# 注释行\n", encoding="utf-8")
    upsert_env(env, "DEEPSEEK_API_KEY", "sk-1")
    upsert_env(env, "DEEPSEEK_API_KEY", "sk-2")
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines.count("DEEPSEEK_API_KEY=sk-1") == 0  # 不重复追加
    assert [ln for ln in lines if ln.startswith("DEEPSEEK_API_KEY")] == ["DEEPSEEK_API_KEY=sk-2"]
    assert "MX_APIKEY=abc" in lines and "# 注释行" in lines


def test_upsert_env_quotes_values_with_spaces(tmp_path) -> None:
    env = tmp_path / ".env"
    upsert_env(env, "TUSHARE_TOKEN", "a b#c")
    assert env.read_text(encoding="utf-8").strip() == 'TUSHARE_TOKEN="a b#c"'


# ----------------------------------------------------------------------
# 4) 端到端：保存模型设置
# ----------------------------------------------------------------------
def test_save_llm_settings_keeps_secret_out_of_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    env = tmp_path / ".env"

    res = save_llm_settings(cfg, env, dict(_LLM_VALUES))
    text = cfg.read_text(encoding="utf-8")

    # 不变量一：明文密钥绝不出现在受版本控制的 config.yaml 里
    assert "sk-test-123456" not in text
    assert yaml.safe_load(text)["llm"]["api_key"] == "${DEEPSEEK_API_KEY}"
    # 不变量二：文件没有被重写（行数不变 = 注释与段落顺序都在）
    assert len(text.splitlines()) == len(_REPO_CONFIG.read_text(encoding="utf-8").splitlines())
    assert "── 免费数据源" in text
    # 真值落在 .env，且当前进程立即可用
    assert env.read_text(encoding="utf-8").strip() == "DEEPSEEK_API_KEY=sk-test-123456"
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-123456"
    # 插值后仍能取回真值（这条链断了就等于保存了个不能用的配置）
    assert load_config(str(cfg))["llm"]["api_key"] == "sk-test-123456"
    # 其余字段照常落盘
    assert load_config(str(cfg))["llm"]["model"] == "deepseek-v4-flash"
    assert res.env_vars == ["DEEPSEEK_API_KEY"]
    assert not res.warned


def test_save_llm_settings_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    env = tmp_path / ".env"
    save_llm_settings(cfg, env, dict(_LLM_VALUES))
    save_llm_settings(cfg, env, dict(_LLM_VALUES, model="deepseek-chat"))
    assert [ln for ln in env.read_text(encoding="utf-8").splitlines()
            if ln.startswith("DEEPSEEK_API_KEY")] == ["DEEPSEEK_API_KEY=sk-test-123456"]
    assert load_config(str(cfg))["llm"]["model"] == "deepseek-chat"


def test_saving_unchanged_values_leaves_the_file_byte_identical(tmp_path) -> None:
    """点一次「保存配置」而什么都没改，不该留下任何 diff。"""
    original = _REPO_CONFIG.read_text(encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(original, encoding="utf-8")
    save_llm_settings(cfg, tmp_path / ".env", {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": "${DEEPSEEK_API_KEY}",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.3,
    })
    assert cfg.read_text(encoding="utf-8") == original


def test_save_llm_settings_warns_on_unresolved_placeholder(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    env = tmp_path / ".env"
    res = save_llm_settings(cfg, env, dict(_LLM_VALUES, api_key="${DEEPSEEK_API_KEY}"))
    assert res.warned and "DEEPSEEK_API_KEY" in res.warned[0]
    assert not env.exists()  # 没有真值就不该凭空造出一个 .env


def test_save_llm_settings_keeps_ollama_key_inline(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    env = tmp_path / ".env"
    save_llm_settings(cfg, env, dict(_LLM_VALUES, provider="ollama", api_key="ollama"))
    assert load_config(str(cfg))["llm"]["api_key"] == "ollama"
    assert not env.exists()


# ----------------------------------------------------------------------
# 5) 端到端：保存数据源设置
# ----------------------------------------------------------------------
def test_save_data_settings_externalises_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("THS_API_TOKEN", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    env = tmp_path / ".env"

    neodata = dict(yaml.safe_load(_REPO_CONFIG.read_text(encoding="utf-8"))["data"]["neodata"])
    neodata["base_url"] = "https://example.invalid"

    save_data_settings(cfg, env, {
        "source": "legacy",
        "primary_source": "sina",
        "prefer_sina": True,
        "tushare_token": "tk-plain",
        "ths_api_token": "ths-plain",
        "neodata": neodata,
        "proxy": {"enabled": True, "http": "http://127.0.0.1:7890", "https": ""},
        "offline": {},
    })
    text = cfg.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)

    assert "tk-plain" not in text and "ths-plain" not in text
    assert doc["data"]["tushare_token"] == "${TUSHARE_TOKEN}"
    assert doc["data"]["neodata"]["base_url"] == "https://example.invalid"
    assert doc["data"]["proxy"]["enabled"] is True
    # 顶层 proxy 段与 data.proxy 同名，打补丁不能串台
    assert doc["proxy"]["enabled"] is False
    assert env.read_text(encoding="utf-8").count("TUSHARE_TOKEN=") == 1
    assert load_config(str(cfg))["data"]["tushare_token"] == "tk-plain"
    # 空 dict 不该把已有的 offline 子段抹掉
    assert doc["data"]["offline"]["index"] == "csi800"


def test_save_data_settings_keeps_comments(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    original = _REPO_CONFIG.read_text(encoding="utf-8")
    cfg.write_text(original, encoding="utf-8")
    save_data_settings(cfg, tmp_path / ".env", {"source": "offline", "primary_source": "akshare"})
    assert len(cfg.read_text(encoding="utf-8").splitlines()) == len(original.splitlines())
    assert "请求频率控制" in cfg.read_text(encoding="utf-8")
