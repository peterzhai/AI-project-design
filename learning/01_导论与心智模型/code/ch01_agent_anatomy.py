"""示例 02：把 Agent 拆开看 —— 记忆、工具箱、执行循环三大组件 + LLM 大脑。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/01_导论与心智模型/code/ch01_agent_anatomy.py

第 1 章只告诉你"Agent = LLM + 工具 + 循环"；本示例把这个公式
工程化：每个零件变成一个独立类，后续章节将逐个升级它们。
"""

import ast
import operator
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

logger = get_logger("ch01.anatomy")

# ---------------------------------------------------------------
# 组件 1：Memory —— 最朴素的短期记忆（一个列表）。
# 第 6 章会把它升级为「滑动窗口 + 长期存储 + 摘要压缩」。
# ---------------------------------------------------------------
class Memory:
    """按顺序保存对话消息，render() 输出完整的上下文列表。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, message: dict) -> None:
        """追加一条消息（存副本，避免外部原地修改污染历史）。"""
        self.items.append(dict(message))

    def render(self) -> list[dict]:
        """输出发给 LLM 的完整上下文（每条也是副本）。"""
        return [dict(m) for m in self.items]

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------
# 组件 2：ToolBox —— 工具注册表：名字 -> 函数，统一执行入口。
# 第 4 章会为它补上 JSON Schema 描述与参数校验。
# ---------------------------------------------------------------
class ToolBox:
    """工具注册与执行。

    关键设计：工具执行失败时不抛异常崩溃，而是把错误文本作为
    「观察结果」返回给 LLM —— 让模型自己决定下一步（重试/换路/道歉）。
    """

    def __init__(self) -> None:
        self._tools: dict[str, callable] = {}

    def register(self, name: str, fn: callable) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"错误：未注册的工具「{name}」，可用工具: {self.names()}"
        try:
            return str(self._tools[name](**arguments))
        except Exception as exc:  # 工具报错 -> 作为观察回传，而非让整个 Agent 崩溃
            return f"工具「{name}」执行失败: {exc}"


# ---------------------------------------------------------------
# 组件 3：AgentSkeleton —— 用三大组件组装出的 Agent 骨架。
# 第 5 章（ReAct）会把这里的循环升级为「思考-行动-观察」的显式三段式。
# ---------------------------------------------------------------
class AgentSkeleton:
    """LLM 大脑 + Memory + ToolBox 的标准组装。"""

    def __init__(self, llm, toolbox: ToolBox, memory: Memory, max_turns: int = 8):
        self.llm = llm
        self.toolbox = toolbox
        self.memory = memory
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        self.memory.add({"role": "user", "content": task})
        for turn in range(1, self.max_turns + 1):
            logger.debug("第 %d 轮，上下文 %d 条消息", turn, len(self.memory))
            response = self.llm.chat(self.memory.render(), tools=self.toolbox.names())
            if not response.tool_calls:
                self.memory.add({"role": "assistant", "content": response.content})
                return response.content
            self.memory.add({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })
            for call in response.tool_calls:
                logger.debug("调用工具 %s(%s)", call.name, call.arguments)
                observation = self.toolbox.execute(call.name, call.arguments)
                logger.debug("观察结果: %s", observation)
                self.memory.add({
                    "role": "tool",
                    "name": call.name,
                    "content": observation,
                })
        raise RuntimeError(f"超过最大轮数 {self.max_turns}")


# ---------------------------------------------------------------
# 组件 4：两个真实工具（演示安全算术与备忘录写入）。
# ---------------------------------------------------------------
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculate(expression: str) -> str:
    """计算只含数字与四则运算的表达式（用 ast 白名单解析，拒绝 eval 注入）。"""
    tree = ast.parse(expression, mode="eval")

    def _eval(node) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"表达式包含不允许的元素: {type(node).__name__}")

    return f"{expression} = {_eval(tree.body)}"


def build_note_tools(notes: list[str]) -> dict:
    """返回依赖共享 notes 列表的一组工具（闭包持有状态）。"""
    def write_note(text: str) -> str:
        notes.append(text)
        return f"已写入备忘录（当前共 {len(notes)} 条）"

    def read_notes() -> str:
        return "\n".join(notes) if notes else "（备忘录为空）"

    return {"calculate": calculate, "write_note": write_note, "read_notes": read_notes}


# ---------------------------------------------------------------
# 组装演示：让 Agent「算出 128*4 并记到备忘录」
# ---------------------------------------------------------------
def main() -> None:
    banner("示例 02：Agent 四大组件解剖")

    notes: list[str] = []
    toolbox = ToolBox()
    for name, fn in build_note_tools(notes).items():
        toolbox.register(name, fn)

    llm = MockLLM(scripted=[
        {"tool_calls": [{"name": "calculate", "arguments": {"expression": "128*4"}}]},
        {"tool_calls": [{"name": "write_note", "arguments": {"text": "128*4=512"}}]},
        {"content": "已算出 128*4=512，并写入备忘录。"},
    ])
    agent = AgentSkeleton(llm=llm, toolbox=toolbox, memory=Memory())
    answer = agent.run("帮我算一下 128*4，然后把结果记到备忘录里")
    print(f"最终答案: {answer}")

    section("自检")
    verify("答案是 LLM 的最终总结", answer == "已算出 128*4=512，并写入备忘录。")
    verify("LLM 被调用 3 次（算数 -> 记录 -> 总结）", llm.call_count == 3, f"实际 {llm.call_count}")
    verify("备忘录里确实有结果", notes == ["128*4=512"], f"实际 {notes}")
    verify("记忆有 6 条消息（1用户 + 3助手 + 2工具结果）", len(agent.memory) == 6, f"实际 {len(agent.memory)}")

    # 鲁棒性 1：模型点名了不存在的工具 -> 错误作为观察回传，Agent 不崩溃
    section("鲁棒性演示 1：模型调用未注册的工具")
    llm_bad_tool = MockLLM(scripted=[
        {"tool_calls": [{"name": "send_email", "arguments": {}}]},
        {"content": "抱歉，我还没有发邮件的工具，无法完成该操作。"},
    ])
    agent2 = AgentSkeleton(llm=llm_bad_tool, toolbox=toolbox, memory=Memory())
    answer2 = agent2.run("给老板发邮件")
    verify("未注册工具被拦截并回传观察", "错误：未注册的工具" in agent2.memory.render()[-2]["content"])
    verify("Agent 存活并给出最终回答", answer2.endswith("无法完成该操作。"))

    # 鲁棒性 2：工具内部报错 -> 同样作为观察回传
    section("鲁棒性演示 2：工具执行失败（恶意表达式）")
    llm_bad_args = MockLLM(scripted=[
        {"tool_calls": [{"name": "calculate", "arguments": {"expression": "__import__('os')"}}]},
        {"content": "计算工具拒绝了该表达式，我不会执行危险操作。"},
    ])
    agent3 = AgentSkeleton(llm=llm_bad_args, toolbox=toolbox, memory=Memory())
    agent3.run("帮我执行 __import__('os')")
    observation = agent3.memory.render()[-2]["content"]
    verify("危险表达式被 ast 白名单拒绝", "不允许的元素" in observation, f"实际观察: {observation}")

    finish("ch01_agent_anatomy")


if __name__ == "__main__":
    main()
