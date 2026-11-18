"""示例 01：路由器模式 —— 一个前台 Router，把任务分派给专职 Agent。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/09_多Agent协作/code/ch09_router.py

为什么多 Agent：一个 Agent 塞 20 个工具，描述挤爆上下文、模型选择也容易出错。
拆分后每个专职 Agent 只带 2~3 个工具 —— 上下文小、职责清、好测试。
Router 的两种实现：
1. LLM 路由：让模型分类意图（灵活，但每单都花一次调用）；
2. 规则路由：关键词匹配（零成本，但死板）—— 生产常用「规则兜底 + LLM 路由」。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


# ---------------------------------------------------------------
# 专职 Agent：每个只处理一类任务，工具与提示词各管各的。
# ---------------------------------------------------------------
@dataclass
class SpecialistAgent:
    """专职 Agent：名字 + 描述（给 Router 看）+ 自己的 LLM。"""

    name: str
    description: str      # 路由依据：什么任务归我管
    llm: MockLLM
    handled: list[str] = field(default_factory=list)  # 审计：接过的任务

    def handle(self, task: str) -> str:
        self.handled.append(task)
        return self.llm.chat([{"role": "user", "content": task}]).content


def refund_agent() -> SpecialistAgent:
    """退款专员：只关心退款。"""
    return SpecialistAgent(
        name="退款专员",
        description="处理退款、退货、取消订单等资金类请求",
        llm=MockLLM(handler=lambda m, t: {"content": "已为您登记退款，1~3 个工作日到账。"}),
    )


def order_agent() -> SpecialistAgent:
    """查单专员：只关心订单状态。"""
    return SpecialistAgent(
        name="查单专员",
        description="查询订单物流、状态、修改地址",
        llm=MockLLM(handler=lambda m, t: {"content": "您的订单已发货，预计明天送达。"}),
    )


def chitchat_agent() -> SpecialistAgent:
    """闲聊助手：兜底。"""
    return SpecialistAgent(
        name="闲聊助手",
        description="与订单业务无关的日常聊天",
        llm=MockLLM(handler=lambda m, t: {"content": "哈哈，今天天气不错呀。"}),
    )


# ---------------------------------------------------------------
# Router：LLM 路由 + 规则兜底。
# ---------------------------------------------------------------
class Router:
    """把任务分派给最合适的专职 Agent。"""

    def __init__(self, agents: list[SpecialistAgent], router_llm: MockLLM,
                 rule_keywords: dict[str, list[str]]):
        self.agents = {a.name: a for a in agents}
        self.router_llm = router_llm
        self.rule_keywords = rule_keywords  # agent名 -> 命中即路由的关键词

    def route_by_llm(self, task: str) -> SpecialistAgent:
        """LLM 路由：把任务与各 Agent 描述一起给模型，让它选人。"""
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        response = self.router_llm.chat([
            {"role": "system", "content": f"你是任务路由器。根据用户请求选择一个最合适的处理人，只输出其名字。\n候选人：\n{roster}"},
            {"role": "user", "content": task},
        ])
        name = response.content.strip()
        if name not in self.agents:
            raise KeyError(f"Router 返回了未知处理人: {name!r}")
        return self.agents[name]

    def route_by_rules(self, task: str) -> SpecialistAgent:
        """规则路由：关键词命中即分派（零成本兜底）。"""
        for agent_name, keywords in self.rule_keywords.items():
            if any(k in task for k in keywords):
                return self.agents[agent_name]
        return self.agents["闲聊助手"]  # 默认兜底


def main() -> None:
    banner("示例 01：路由器模式")

    agents = [refund_agent(), order_agent(), chitchat_agent()]
    # Router 的"大脑"：按任务文本路由（模拟 LLM 判断力）
    router_llm = MockLLM(handler=lambda m, t: {
        "content": ("退款专员" if "退款" in m[-1]["content"] or "退货" in m[-1]["content"]
                    else "查单专员" if "订单" in m[-1]["content"] or "物流" in m[-1]["content"]
                    else "闲聊助手")
    })
    router = Router(agents, router_llm, rule_keywords={
        "退款专员": ["退款", "退货"],
        "查单专员": ["订单", "物流"],
        "闲聊助手": [],
    })

    tasks = ["我要退货退款", "帮我查下订单到哪了", "你觉得今天天气怎么样"]

    section("实验 1：LLM 路由 —— 每个任务找到对的专员")
    for task in tasks:
        agent = router.route_by_llm(task)
        answer = agent.handle(task)
        print(f"  「{task}」 -> {agent.name}: {answer}")
    verify("退款请求给了退款专员", agents[0].handled == ["我要退货退款"])
    verify("查单请求给了查单专员", agents[1].handled == ["帮我查下订单到哪了"])
    verify("闲聊给了闲聊助手", agents[2].handled == ["你觉得今天天气怎么样"])
    verify("Router 被调用了 3 次（每单一览）", router_llm.call_count == 3)

    section("实验 2：规则路由 —— 零成本兜底")
    agent2 = router.route_by_rules("我要退货退款")
    verify("规则命中退款专员（0 次 LLM 调用）", agent2.name == "退款专员" and router_llm.call_count == 3)
    agent3 = router.route_by_rules("今晚上吃什么好呢")
    verify("无关键词命中兜底到闲聊助手", agent3.name == "闲聊助手")

    section("实验 3：专职化的收益 —— 工具清单互不干扰")
    print("每个专员只看到自己的职责域，Router 花一次调用完成分派；")
    print("若合并为单 Agent，则三个职责的描述与工具全部挤进同一份上下文。")
    verify("专职 Agent 各自只处理了属于自己的任务",
           sum(len(a.handled) for a in agents) == 3 and all(len(a.handled) == 1 for a in agents))

    finish("ch09_router")


if __name__ == "__main__":
    main()
