"""第 5 章测试：ReAct 三段式、轨迹持久化与续跑、终止条件。"""

import pytest

import ch05_react_loop as loop_demo
import ch05_termination as term_demo
import ch05_trace_replay as replay_demo
from common.mock_llm import MockLLM, ToolCall
from common.toolbox import ToolBox


def _weather_box() -> ToolBox:
    return ToolBox().register(loop_demo.get_weather)


# ---------- 示例 01：ReAct 三段式 ----------
def test_react_loop_main_end_to_end():
    loop_demo.main()


def test_react_agent_records_three_part_steps():
    llm = MockLLM(scripted=[
        {"content": "思考A。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "思考B。"},
    ])
    trace = loop_demo.ReActAgent(llm, _weather_box()).run("任务")
    assert len(trace.steps) == 2
    assert trace.steps[0].thought == "思考A。"
    assert trace.steps[0].action == ToolCall(name="get_weather", arguments={"city": "北京"}, id="call_1")
    assert trace.steps[0].observation == "晴，5~12°C"
    assert trace.steps[1].final_answer == "思考B。"
    assert trace.termination.startswith("完成")


def test_thought_enters_context_for_next_turn():
    llm = MockLLM(scripted=[
        {"content": "我要查天气。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "上海"}}]},
        {"content": "好了。"},
    ])
    llm2 = MockLLM(scripted=["占位"])  # 仅供构造，不影响断言
    agent = loop_demo.ReActAgent(llm, _weather_box())
    agent.run("上海天气")
    messages = llm.calls[1]["messages"]
    assert messages[1]["content"] == "我要查天气。"          # 思考显式进入上下文
    assert messages[2]["content"] == "上海：多云"             # 观察回传（ch05 假数据库无上海，走默认值）


def test_max_turns_fuse_in_react_agent():
    llm = MockLLM(handler=lambda m, t: {
        "content": "继续。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}],
    })
    trace = loop_demo.ReActAgent(llm, _weather_box(), max_turns=3).run("死循环任务")
    assert len(trace.steps) == 3
    assert "保险丝" in trace.termination


# ---------- 示例 02：轨迹持久化与续跑 ----------
def test_trace_replay_main_end_to_end():
    replay_demo.main()


def test_trace_json_roundtrip_preserves_everything():
    llm = MockLLM(scripted=[
        {"content": "先查。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "回答。"},
    ])
    trace = loop_demo.ReActAgent(llm, _weather_box()).run("任务X")
    restored = replay_demo.trace_from_json(replay_demo.trace_to_json(trace))
    assert restored.task == trace.task
    assert restored.termination == trace.termination
    assert restored.render() == trace.render()
    assert restored.steps[0].action.id == "call_1"


def test_continue_from_trace_rebuilds_context():
    llm = MockLLM(scripted=[
        {"content": "先查。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "回答。"},
    ])
    trace = loop_demo.ReActAgent(llm, _weather_box()).run("任务X")
    llm2 = MockLLM(scripted=[{"content": "续跑回答。"}])
    resumed = replay_demo.continue_from_trace(
        llm2, _weather_box(), trace, {"get_weather": "晴，5~12°C"})
    assert resumed.steps[0].final_answer == "续跑回答。"
    messages = llm2.calls[0]["messages"]
    assert messages[0]["content"] == "任务X"           # 原任务
    assert messages[1]["content"] == "先查。"          # 历史思考
    assert messages[2]["content"] == "晴，5~12°C"      # 注入的观察


# ---------- 示例 03：终止条件 ----------
def test_termination_main_end_to_end():
    term_demo.main()


def test_repeated_call_detector_canonicalizes_args():
    detector = term_demo.repeated_call_detector(times=2)
    def step(action):
        return loop_demo.Step(turn=1, thought="t", action=action, observation="o")
    trace = loop_demo.Trace(task="t", steps=[
        step(ToolCall(name="f", arguments={"a": 1, "b": 2})),
        step(ToolCall(name="f", arguments={"b": 2, "a": 1})),  # 键序不同，应视为同一调用
    ])
    reason = detector(trace)
    assert reason and "疑似死循环" in reason


def test_different_args_not_flagged():
    detector = term_demo.repeated_call_detector(times=2)
    trace = loop_demo.Trace(task="t", steps=[
        loop_demo.Step(turn=1, thought="t", action=ToolCall(name="f", arguments={"c": "北京"}), observation="o"),
        loop_demo.Step(turn=2, thought="t", action=ToolCall(name="f", arguments={"c": "上海"}), observation="o"),
    ])
    assert detector(trace) is None


def test_guarded_agent_stops_on_repeat_but_not_on_new_args():
    llm_repeat = MockLLM(handler=lambda m, t: {
        "content": "再查。",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}],
    })
    trace = term_demo.GuardedAgent(llm_repeat, _weather_box(), max_turns=10) \
        .add_stop_condition(term_demo.repeated_call_detector(times=2)).run("t")
    assert len(trace.steps) == 2 and "重复调用检测" in trace.termination

    llm_ok = MockLLM(scripted=[
        {"content": "查北京。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "查上海。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "上海"}}]},
        {"content": "完成。"},
    ])
    trace2 = term_demo.GuardedAgent(llm_ok, _weather_box()) \
        .add_stop_condition(term_demo.repeated_call_detector(times=2)).run("t2")
    assert trace2.termination.startswith("完成")
