"""示例 03：同样是"让 LLM 做事"，写死步骤的流水线和 Agent 差在哪？

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/01_导论与心智模型/code/ch01_agent_vs_pipeline.py

一句话结论：
- 流水线：步骤在代码里写死，输入再离谱也照跑全套 —— 便宜、可控、笨。
- Agent：  LLM 在运行时决定"要不要做、做什么、做几步" —— 灵活，但需要护栏。
两者不是对立关系：成熟系统常用「Agent 做路由与规划 + 流水线做固定工序」。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

FAKE_WEATHER_DB = {"北京": "晴，5~12°C，西北风3级"}


def get_weather(city: str) -> str:
    """查询城市天气（演示假数据）。"""
    return FAKE_WEATHER_DB.get(city, f"{city}：多云")


# ---------------------------------------------------------------
# 方案 A：传统流水线 —— 步骤写死，第 1 步永远先查天气。
# ---------------------------------------------------------------
class WeatherPipeline:
    """固定两步流水线：查天气 -> 拼提示词让 LLM 回复。"""

    def __init__(self, llm):
        self.llm = llm
        self.weather_calls = 0  # 统计调了几次天气工具，用来暴露流水线的荒谬之处

    def run(self, user_input: str) -> str:
        # 第 1 步（写死）：无论用户问什么，都先查天气
        self.weather_calls += 1
        weather = get_weather("北京")
        # 第 2 步（写死）：把天气塞进提示词
        prompt = f"用户输入：{user_input}\n参考天气：{weather}\n请回复用户。"
        response = self.llm.chat([{"role": "user", "content": prompt}])
        return response.content


# ---------------------------------------------------------------
# 方案 B：Agent —— LLM 看任务决定行动。
# handler 模拟真实 LLM 的判断力：诗题直接写诗；天气题先查再答。
# ---------------------------------------------------------------
def make_agent_llm(weather_counter: list[int]) -> MockLLM:
    """构造一个'有判断力'的假 LLM：按任务内容路由行为。"""

    def handler(messages, tools):
        task = messages[0]["content"]  # 第一条 user 消息就是任务本身
        if "诗" in task:
            return {"content": "秋风起，黄叶飞，一盏灯下人未归。"}  # 无需工具，直接创作
        if any(m.get("role") == "tool" for m in messages):
            return {"content": f"根据天气「{messages[-1]['content']}」，今天天气不错，适合洗车。"}
        weather_counter[0] += 1
        return {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]}

    return MockLLM(handler=handler)


def run_agent_like_loop(llm, task: str, max_turns: int = 5) -> str:
    """与 ch01_hello_agent 相同的最小循环（此处独立成函数便于对比实验）。"""
    messages = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        response = llm.chat(messages, tools=["get_weather"])
        if not response.tool_calls:
            return response.content
        messages.append({"role": "assistant", "content": response.content,
                         "tool_calls": response.tool_calls})
        for call in response.tool_calls:
            result = get_weather(**call.arguments)
            messages.append({"role": "tool", "name": call.name, "content": str(result)})
    raise RuntimeError("超过最大轮数")


def main() -> None:
    banner("示例 03：流水线 vs Agent —— 谁在决定下一步？")

    weather_question = "今天适合洗车吗？"
    poem_request = "帮我写一首关于秋天的短诗"

    # 对照实验 1：流水线处理天气问题 —— 查天气是合理的
    section("实验 1：流水线处理「今天适合洗车吗？」")
    pipeline_llm = MockLLM(scripted=[
        "北京今天晴，5~12°C，没有雨，适合洗车。",
    ])
    pipeline = WeatherPipeline(pipeline_llm)
    print(f"流水线回答: {pipeline.run(weather_question)}")
    verify("流水线查了 1 次天气（合理）", pipeline.weather_calls == 1, f"实际 {pipeline.weather_calls}")

    # 对照实验 2：流水线处理写诗请求 —— 天气照查不误，暴露荒谬性
    section("实验 2：流水线处理「帮我写一首关于秋天的短诗」")
    pipeline_llm2 = MockLLM(scripted=[
        "秋风起，黄叶飞……（同时参考了无关的天气信息）",
    ])
    pipeline2 = WeatherPipeline(pipeline_llm2)
    print(f"流水线回答: {pipeline2.run(poem_request)}")
    verify(
        "写诗任务流水线仍查了天气（步骤写死，无法跳过）",
        pipeline2.weather_calls == 1,
        f"实际 {pipeline2.weather_calls}",
    )

    # 对照实验 3：Agent 处理写诗 —— 不调任何工具，直接创作
    section("实验 3：Agent 处理同一句写诗请求")
    poem_counter = [0]
    agent_llm = make_agent_llm(poem_counter)
    poem = run_agent_like_loop(agent_llm, poem_request)
    print(f"Agent 回答: {poem}")
    verify("Agent 对写诗任务 0 次查天气", poem_counter[0] == 0, f"实际 {poem_counter[0]}")
    verify("Agent 直接输出诗歌（1 次调用）", agent_llm.call_count == 1, f"实际 {agent_llm.call_count}")

    # 对照实验 4：Agent 处理洗车问题 —— 判断需要天气，先查再答
    section("实验 4：Agent 处理「今天适合洗车吗？」")
    car_counter = [0]
    agent_llm2 = make_agent_llm(car_counter)
    car_answer = run_agent_like_loop(agent_llm2, weather_question)
    print(f"Agent 回答: {car_answer}")
    verify("Agent 判断需要天气，查了 1 次", car_counter[0] == 1, f"实际 {car_counter[0]}")
    verify("拿到天气后才给最终建议（2 次调用）", agent_llm2.call_count == 2, f"实际 {agent_llm2.call_count}")
    verify("最终建议基于真实观察结果", "晴" in car_answer)

    finish("ch01_agent_vs_pipeline")


if __name__ == "__main__":
    main()
