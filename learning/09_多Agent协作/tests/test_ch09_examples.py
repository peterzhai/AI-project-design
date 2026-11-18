"""第 9 章测试：路由器、编排者-执行者、消息传递与黑板。"""

import pytest

import ch09_message_passing as message_demo
import ch09_orchestrator as orchestrator_demo
import ch09_router as router_demo
from common.mock_llm import MockLLM
from common.json_utils import JsonParseError


def build_router():
    agents = [router_demo.refund_agent(), router_demo.order_agent(), router_demo.chitchat_agent()]
    router_llm = MockLLM(handler=lambda m, t: {
        "content": ("退款专员" if "退款" in m[-1]["content"] or "退货" in m[-1]["content"]
                    else "查单专员" if "订单" in m[-1]["content"] or "物流" in m[-1]["content"]
                    else "闲聊助手")
    })
    router = router_demo.Router(agents, router_llm, rule_keywords={
        "退款专员": ["退款", "退货"], "查单专员": ["订单", "物流"], "闲聊助手": [],
    })
    return router, router_llm


# ---------- 示例 01：路由器 ----------
def test_router_main_end_to_end():
    router_demo.main()


def test_llm_routing_dispatches_to_specialists():
    router, router_llm = build_router()
    assert router.route_by_llm("我要退货退款").name == "退款专员"
    assert router.route_by_llm("查订单物流").name == "查单专员"
    assert router.route_by_llm("讲个笑话").name == "闲聊助手"
    assert router_llm.call_count == 3


def test_rule_routing_costs_zero_llm_calls():
    router, router_llm = build_router()
    assert router.route_by_rules("退款怎么操作").name == "退款专员"
    assert router.route_by_rules("随便聊聊天").name == "闲聊助手"  # 兜底
    assert router_llm.call_count == 0  # 规则路由零 LLM 调用


def test_unknown_router_output_raises():
    agents = [router_demo.refund_agent()]
    router_llm = MockLLM(scripted=["不存在的人"])
    router = router_demo.Router(agents, router_llm, rule_keywords={})
    with pytest.raises(KeyError, match="未知处理人"):
        router.route_by_llm("任务")


# ---------- 示例 02：编排者-执行者 ----------
def test_orchestrator_main_end_to_end():
    orchestrator_demo.main()


def test_orchestrator_runs_all_workers():
    planner = MockLLM(handler=lambda m, t: {"content": '{"subtasks": ['
        '{"worker": "w1", "instruction": "任务一"},'
        '{"worker": "w2", "instruction": "任务二"}]}'})
    synth = MockLLM(scripted=["汇总完成"])
    w1 = MockLLM(handler=lambda m, t: {"content": "W1 结果"})
    w2 = MockLLM(handler=lambda m, t: {"content": "W2 结果"})
    workers = {"w1": orchestrator_demo.Worker("w1", w1), "w2": orchestrator_demo.Worker("w2", w2)}
    orch = orchestrator_demo.Orchestrator(planner, synth)
    answer, subtasks = orch.run("总任务", workers)
    assert answer == "汇总完成"
    assert [t.output for t in subtasks] == ["W1 结果", "W2 结果"]
    assert w1.call_count == 1 and w2.call_count == 1


def test_unknown_worker_is_skipped_not_crash():
    planner = MockLLM(scripted=[{"content": '{"subtasks": [{"worker": "ghost", "instruction": "x"}]}'}])
    orch = orchestrator_demo.Orchestrator(planner, MockLLM(scripted=["部分完成"]))
    _, subtasks = orch.run("t", {})
    assert "未找到执行者" in subtasks[0].output


def test_planner_bad_json_raises():
    planner = MockLLM(scripted=[{"content": "坏输出"}])
    orch = orchestrator_demo.Orchestrator(planner, MockLLM(scripted=[]))
    with pytest.raises(JsonParseError):
        orch.run("t", {})


# ---------- 示例 03：消息传递与黑板 ----------
def test_message_passing_main_end_to_end():
    message_demo.main()


def test_bus_delivers_and_logs():
    bus = message_demo.MessageBus()
    bus.send(message_demo.Message(sender="A", receiver="B", content="你好"))
    inbox = bus.receive("B")
    assert len(inbox) == 1 and inbox[0].content == "你好"
    assert bus.receive("B") == []          # 取走即清空
    assert bus.receive("C") == []          # 没有收件人也是空
    assert len(bus.log) == 1               # 审计日志保留


def test_blackboard_write_read_overwrite():
    board = message_demo.Blackboard()
    board.write("k", "w1", "v1")
    assert board.read("k") == "v1"
    board.write("k", "w2", "v2")           # 后写覆盖
    assert board.read("k") == "v2"
    assert board.entries["k"][0] == "w2"   # 记录写入者
    assert board.read("missing") == ""
