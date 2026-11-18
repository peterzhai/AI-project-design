"""示例 03：Agent 间的消息传递与黑板协作 —— 显式协议胜过隐性约定。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/09_多Agent协作/code/ch09_message_passing.py

两种协作机制：
1. 消息传递（Message Passing）：Agent 之间点对点发消息，收件人显式处理 ——
   类似团队里"张三把报告发给李四"；
2. 黑板（Blackboard）：大家往一块共享板上写发现，谁需要谁去读 ——
   类似"把结论贴在白板上"。
两者的共同底线：**协作内容必须显式落盘成数据**，而不是藏在各自上下文里靠猜。
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


@dataclass(frozen=True)
class Message:
    """Agent 间消息：显式的发件人/收件人/内容。"""

    sender: str
    receiver: str
    content: str


class MessageBus:
    """消息总线：按收件人投递队列；记录全部消息供审计。"""

    def __init__(self):
        self._queues: dict[str, deque] = {}
        self.log: list[Message] = []

    def send(self, message: Message) -> None:
        self.log.append(message)
        self._queues.setdefault(message.receiver, deque()).append(message)

    def receive(self, agent_name: str) -> list[Message]:
        """取走该 Agent 的全部未读消息。"""
        inbox = self._queues.get(agent_name, deque())
        messages = list(inbox)
        inbox.clear()
        return messages


class Blackboard:
    """共享黑板：key -> 写入记录（谁在何时写了什么）。"""

    def __init__(self):
        self.entries: dict[str, tuple[str, str]] = {}  # key -> (writer, content)

    def write(self, key: str, writer: str, content: str) -> None:
        self.entries[key] = (writer, content)

    def read(self, key: str) -> str:
        return self.entries[key][1] if key in self.entries else ""


def analyze_data_agent(llm: MockLLM, bus: MessageBus, board: Blackboard):
    """数据分析员：分析完 -> 黑板记录结论 -> 点对点通知报告员。"""

    def handle(task: str) -> str:
        conclusion = llm.chat([{"role": "user", "content": task}]).content
        board.write("销售结论", "数据分析员", conclusion)   # 协作产出落盘到黑板
        bus.send(Message(sender="数据分析员", receiver="报告撰写员",
                         content=f"分析完成，结论已写到黑板「销售结论」，请查阅并撰写报告"))
        return conclusion

    return handle


def report_agent(llm: MockLLM, bus: MessageBus, board: Blackboard):
    """报告撰写员：收消息 -> 读黑板 -> 基于分析结论写报告。"""

    def handle(task: str) -> str:
        inbox = bus.receive("报告撰写员")
        assert inbox, "报告撰写员应当在有通知时才被调用"
        notification = inbox[0]
        conclusion = board.read("销售结论")                 # 从黑板读取协作输入
        report = llm.chat([{
            "role": "user",
            "content": f"任务: {task}\n{notification.sender} 的通知: {notification.content}\n"
                       f"分析结论: {conclusion}\n请撰写最终报告。",
        }]).content
        board.write("最终报告", "报告撰写员", report)
        return report

    return handle


def main() -> None:
    banner("示例 03：消息传递与黑板协作")
    bus = MessageBus()
    board = Blackboard()

    analyzer_llm = MockLLM(handler=lambda m, t: {"content": "Q4 销售额同比增长 15%，华东区贡献最大。"})
    reporter_llm = MockLLM(handler=lambda m, t: {
        "content": "【Q4 销售报告】整体增长 15%，建议明年加大华东区投入。（基于数据分析员结论）"})

    analyze = analyze_data_agent(analyzer_llm, bus, board)
    write_report = report_agent(reporter_llm, bus, board)

    # 协作流水线：分析 -> 通知 -> 写报告
    section("协作过程")
    conclusion = analyze("分析 Q4 销售数据")
    print(f"数据分析员: {conclusion}")
    print(f"消息总线日志: {[(m.sender, m.receiver) for m in bus.log]}")
    report = write_report("撰写 Q4 销售报告")
    print(f"报告撰写员: {report}")

    verify("分析结论先落黑板（显式数据，不是藏在上下文里）",
           board.read("销售结论") == "Q4 销售额同比增长 15%，华东区贡献最大。")
    verify("通知消息显式投递到撰写员收件箱", len(bus.log) == 1 and bus.log[0].receiver == "报告撰写员")
    verify("报告基于黑板上的分析结论（而非自说自话）",
           "15%" in report and "华东区" in report)
    verify("最终报告也落黑板（协作链路全程可审计）", "最终报告" in board.entries)

    section("审计视角：全链路回顾")
    print(f"黑板条目: {list(board.entries)}")
    print(f"消息轨迹: {[(m.sender, m.receiver, m.content[:20]) for m in bus.log]}")
    verify("协作是显式协议（黑板+消息），任何一步都可回放", True)

    finish("ch09_message_passing")


if __name__ == "__main__":
    main()
