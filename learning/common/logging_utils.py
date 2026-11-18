"""统一日志：教程里的示例都用它打印过程，输出格式一致、便于阅读。"""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    # 环境变量 LEARNING_VERBOSE=1 可打开 DEBUG 级日志，观察每一步细节。
    level = logging.DEBUG if os.environ.get("LEARNING_VERBOSE") == "1" else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger("learning")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """按名字取子 Logger，例如 get_logger("ch01.hello_agent")。"""
    _ensure_configured()
    return logging.getLogger(f"learning.{name}")


def set_verbose(enabled: bool = True) -> None:
    """运行时切换 DEBUG/INFO 级别（tests 里偶尔用到）。"""
    _ensure_configured()
    logging.getLogger("learning").setLevel(logging.DEBUG if enabled else logging.INFO)
