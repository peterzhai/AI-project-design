"""第 1 章测试：导入三个示例模块，既跑 main() 端到端自检，也对核心函数做精确断言。"""

import pytest

import ch01_agent_anatomy as anatomy
import ch01_agent_vs_pipeline as versus
import ch01_hello_agent as hello


# ---------- 示例 01：最小 Agent ----------
def test_hello_agent_main_end_to_end():
    """main() 内部含多组 verify 断言，零异常退出即全部通过。"""
    hello.main()


def test_minimal_agent_two_turn_tool_loop():
    llm = hello.MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "上海"}}]},
        {"content": "上海有小雨，记得带伞。"},
    ])
    answer = hello.run_minimal_agent(llm, hello.TOOLS, "上海天气怎么样")
    assert answer == "上海有小雨，记得带伞。"
    assert llm.call_count == 2
    # 第二次调用时，模型上下文里应包含工具名与观察结果
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "tool", "name": "get_weather", "content": "小雨，14~18°C"}


def test_minimal_agent_no_tool_needed():
    llm = hello.MockLLM(scripted=["你好，我是助手。"])
    answer = hello.run_minimal_agent(llm, hello.TOOLS, "打个招呼")
    assert answer == "你好，我是助手。"
    assert llm.call_count == 1


def test_minimal_agent_max_turns_fuse():
    llm = hello.MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]} for _ in range(10)
    ])
    with pytest.raises(RuntimeError, match="超过最大轮数"):
        hello.run_minimal_agent(llm, hello.TOOLS, "查天气", max_turns=3)
    assert llm.call_count == 3  # 恰好在保险丝处被截断


# ---------- 示例 02：四大组件 ----------
def test_anatomy_main_end_to_end():
    anatomy.main()


def test_memory_stores_copies_not_references():
    memory = anatomy.Memory()
    message = {"role": "user", "content": "原始"}
    memory.add(message)
    message["content"] = "外部篡改"
    assert memory.render()[0]["content"] == "原始"  # 存的是副本
    assert len(memory) == 1


def test_toolbox_unknown_tool_returns_error_observation():
    toolbox = anatomy.ToolBox()
    observation = toolbox.execute("not_exist", {})
    assert "错误：未注册的工具" in observation
    assert observation.endswith("[]")


def test_toolbox_tool_exception_becomes_observation():
    toolbox = anatomy.ToolBox()
    toolbox.register("boom", lambda: (_ for _ in ()).throw(ValueError("爆炸了")))
    observation = toolbox.execute("boom", {})
    assert "执行失败" in observation and "爆炸了" in observation


def test_calculate_whitelisted_arithmetic_only():
    assert anatomy.calculate("128*4") == "128*4 = 512"
    assert anatomy.calculate("(1+2)*3") == "(1+2)*3 = 9"
    with pytest.raises(ValueError, match="不允许的元素"):
        anatomy.calculate("__import__('os').system('id')")


def test_skeleton_agent_runs_three_turns():
    notes: list[str] = []
    toolbox = anatomy.ToolBox()
    for name, fn in anatomy.build_note_tools(notes).items():
        toolbox.register(name, fn)
    llm = anatomy.MockLLM(scripted=[
        {"tool_calls": [{"name": "calculate", "arguments": {"expression": "9*9"}}]},
        {"tool_calls": [{"name": "write_note", "arguments": {"text": "9*9=81"}}]},
        {"content": "完成：9*9=81 已记录。"},
    ])
    agent = anatomy.AgentSkeleton(llm=llm, toolbox=toolbox, memory=anatomy.Memory())
    assert agent.run("算 9*9 并记录") == "完成：9*9=81 已记录。"
    assert notes == ["9*9=81"]


# ---------- 示例 03：流水线 vs Agent ----------
def test_vs_pipeline_main_end_to_end():
    versus.main()


def test_pipeline_always_fetches_weather_even_for_poem():
    llm = versus.MockLLM(scripted=["诗（还查了天气）", "洗车建议"])
    pipeline = versus.WeatherPipeline(llm)
    pipeline.run("帮我写首诗")
    pipeline.run("今天适合洗车吗")
    assert pipeline.weather_calls == 2  # 写诗也查了天气 —— 流水线的呆板之处


def test_agent_skips_tools_for_poem_but_uses_them_for_weather():
    poem_counter = [0]
    poem_llm = versus.make_agent_llm(poem_counter)
    poem = versus.run_agent_like_loop(poem_llm, "写一首秋天的诗")
    assert poem == "秋风起，黄叶飞，一盏灯下人未归。"
    assert poem_counter[0] == 0  # Agent 判断无需工具

    car_counter = [0]
    car_llm = versus.make_agent_llm(car_counter)
    answer = versus.run_agent_like_loop(car_llm, "今天适合洗车吗？")
    assert car_counter[0] == 1   # Agent 判断需要天气
    assert "晴" in answer        # 建议基于观察结果
