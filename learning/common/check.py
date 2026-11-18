"""示例脚本的自检助手。

教程约定：每个示例脚本最后都有一段"自检区"，用 verify() 逐条断言，
全部通过才算跑通。任何一条失败会立刻抛 AssertionError，退出码非 0，
方便人肉观察也方便测试与 CI。
"""

from __future__ import annotations

import sys

PASS = "✅"
FAIL = "❌"


def banner(title: str) -> None:
    """打印示例的分节标题。"""
    line = "=" * 62
    print(f"\n{line}\n{title}\n{line}")


def section(title: str) -> None:
    """打印小节标题。"""
    print(f"\n--- {title} " + "-" * max(0, 50 - len(title)))


def verify(label: str, condition: bool, detail: str = "") -> bool:
    """断言一条预期：通过打印绿勾；失败打印红叉并抛 AssertionError。"""
    if condition:
        print(f"{PASS} {label}")
        return True
    message = f"{FAIL} {label}" + (f"（{detail}）" if detail else "")
    print(message)
    raise AssertionError(message)


def check_all(label: str, results: list[tuple[str, bool]]) -> None:
    """批量断言：全部通过才打印总结，任何一条失败立刻抛错。"""
    for text, ok in results:
        if not ok:
            raise AssertionError(f"{FAIL} {label}: {text}")
    print(f"{PASS} {label}（共 {len(results)} 条断言全部通过）")


def finish(script_name: str) -> None:
    """示例结尾统一收尾：明确提示自检全部通过。"""
    print(f"\n🎉 [{script_name}] 全部自检通过，零异常退出。")
    sys.stdout.flush()
