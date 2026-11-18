"""示例 01：配置与环境管理 —— 密钥不进代码，配置不靠记忆。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/12_工程化部署/code/ch12_config.py

部署第一课：把散落各处的"魔法值"收敛成一个 AgentConfig：
- 来源分层：环境变量 > 显式传入 > 安全默认值（永远不硬编码密钥）；
- 启动即校验：非法配置在服务启动时炸掉，而不是运行到一半；
- 敏感字段 repr 打码：日志里打印配置对象不泄漏密钥。
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402


class ConfigError(ValueError):
    """配置非法（缺失/类型错/取值越界），必须在启动时暴露。"""


@dataclass
class AgentConfig:
    """Agent 服务的全部配置。"""

    llm_mode: str = "mock"                 # mock / real
    api_key: str = ""                      # 真实模式必填；repr 打码
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_turns: int = 8                     # Agent 循环保险丝
    budget_tokens: int = 100_000           # 预算熔断（第 11 章）
    allowed_injection_mode: str = "reject"  # 输入护栏模式（第 11 章）
    _secrets: set = field(default_factory=set, repr=False)  # 敏感字段名清单

    def __post_init__(self):
        self._secrets = {"api_key"}
        # 校验 1：枚举字段
        if self.llm_mode not in ("mock", "real"):
            raise ConfigError(f"llm_mode 必须是 mock/real，收到 {self.llm_mode!r}")
        if self.allowed_injection_mode not in ("reject", "strip"):
            raise ConfigError(f"allowed_injection_mode 必须是 reject/strip，收到 {self.allowed_injection_mode!r}")
        # 校验 2：数值范围
        if not 1 <= self.max_turns <= 50:
            raise ConfigError(f"max_turns 应在 1~50，收到 {self.max_turns}")
        if self.budget_tokens < 100:
            raise ConfigError(f"budget_tokens 至少 100，收到 {self.budget_tokens}")
        # 校验 3：真实模式必须有密钥；mock 模式不需要
        if self.llm_mode == "real" and not self.api_key:
            raise ConfigError("llm_mode=real 但缺少 api_key（禁止把密钥写进代码，请用环境变量）")
        # 校验 4：base_url 必须是 http(s)
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(f"base_url 非法: {self.base_url!r}")

    def __repr__(self) -> str:
        """敏感字段打码：print(config) 永不泄漏密钥。"""
        fields = []
        for name in self.__dataclass_fields__:
            if name.startswith("_"):
                continue
            value = getattr(self, name)
            if name in self._secrets and value:
                value = f"***({len(str(value))} chars)"
            fields.append(f"{name}={value!r}")
        return f"AgentConfig({', '.join(fields)})"

    @classmethod
    def from_env(cls, env=None) -> "AgentConfig":
        """从环境变量构建配置（env 可注入，测试零污染）。

        环境变量命名约定：LEARNING_LLM_MODE / LEARNING_API_KEY /
        LEARNING_BASE_URL / LEARNING_MODEL / LEARNING_MAX_TURNS / LEARNING_BUDGET_TOKENS
        """
        env = env if env is not None else os.environ

        def get_int(name: str, default: int) -> int:
            raw = env.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ConfigError(f"环境变量 {name} 必须是整数，收到 {raw!r}") from exc

        return cls(
            llm_mode=env.get("LEARNING_LLM_MODE", "mock"),
            api_key=env.get("LEARNING_API_KEY", ""),
            base_url=env.get("LEARNING_BASE_URL", cls.__dataclass_fields__["base_url"].default),
            model=env.get("LEARNING_MODEL", "gpt-4o-mini"),
            max_turns=get_int("LEARNING_MAX_TURNS", 8),
            budget_tokens=get_int("LEARNING_BUDGET_TOKENS", 100_000),
        )


def main() -> None:
    banner("示例 01：配置与环境管理")

    # 实验 1：默认配置直接可用（离线 Mock）
    section("实验 1：默认配置（Mock 模式，零外部依赖）")
    config = AgentConfig()
    print(repr(config))
    verify("默认配置即 Mock 模式", config.llm_mode == "mock")
    verify("repr 打码密钥（即使将来配了也不泄漏）",
           "***" not in repr(config) or config.api_key == "")

    # 实验 2：环境变量覆盖 + 类型错误即时暴露
    section("实验 2：from_env 读取环境变量（可注入，零污染）")
    config2 = AgentConfig.from_env(env={"LEARNING_LLM_MODE": "mock",
                                        "LEARNING_MAX_TURNS": "12",
                                        "LEARNING_BUDGET_TOKENS": "5000"})
    verify("环境变量覆盖默认值", config2.max_turns == 12 and config2.budget_tokens == 5000)
    try:
        AgentConfig.from_env(env={"LEARNING_MAX_TURNS": "八"})
        raise AssertionError("实验 2 失败：非整数应当报错")
    except ConfigError as exc:
        print(f"按预期失败: {exc}")
    verify("非整数环境变量启动即报错", True)

    # 实验 3：真实模式缺密钥 -> 拒绝启动
    section("实验 3：real 模式缺密钥被拒绝（fail-fast）")
    try:
        AgentConfig(llm_mode="real", api_key="")
        raise AssertionError("实验 3 失败：应当拒绝")
    except ConfigError as exc:
        print(f"按预期失败: {exc}")
    verify("real 模式必须带 api_key", True)

    # 实验 4：合法密钥配置 + repr 打码
    section("实验 4：密钥打码")
    config4 = AgentConfig(llm_mode="real", api_key="sk-secret-1234567890abcdef")
    printed = repr(config4)
    print(printed)
    verify("repr 中不出现明文密钥", "sk-secret-1234567890abcdef" not in printed)
    verify("打码信息包含长度提示", "***" in printed and "26 chars" in printed)

    # 实验 5：非法取值全部启动即炸
    section("实验 5：各配置项的边界校验")
    bad_configs = [
        (dict(llm_mode="yolo"), "llm_mode"),
        (dict(max_turns=0), "max_turns"),
        (dict(max_turns=99), "max_turns"),
        (dict(budget_tokens=10), "budget_tokens"),
        (dict(base_url="ftp://x"), "base_url"),
        (dict(allowed_injection_mode="yolo"), "allowed_injection_mode"),
    ]
    for kwargs, field_name in bad_configs:
        try:
            AgentConfig(**kwargs)
            raise AssertionError(f"实验 5 失败：{kwargs} 应当被拒绝")
        except ConfigError as exc:
            print(f"  {field_name}: {exc}")
    verify("6 组非法配置全部启动即炸", True)

    finish("ch12_config")


if __name__ == "__main__":
    main()
