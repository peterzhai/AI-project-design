"""示例 02：评估数据集与评估器 —— "感觉不错"不算数，跑分才算数。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/10_评估、追踪与可观测性/code/ch10_evaluators.py

评估三板斧（不需要大模型裁判也能起步）：
1. 包含断言：答案必须包含关键信息（防漏）；
2. 正则断言：答案必须匹配格式（防格式漂移）；
3. 禁含断言：答案不得出现编造词（防幻觉）。
EvalSuite 批量跑 -> 通过率 + 每个失败案例的逐条原因。改了提示词/换模型后重跑，
分数变化就是改进（或退化）的证据 —— 这就是"评估驱动开发"。
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402


@dataclass
class EvalCase:
    """一个评估案例：问题 + 通过条件。"""

    name: str
    question: str
    must_contain: list[str] = field(default_factory=list)  # 答案必须包含
    must_not_contain: list[str] = field(default_factory=list)  # 答案必须不包含
    pattern: str = ""  # 答案必须匹配的正则（空串跳过）


@dataclass
class EvalResult:
    """单个案例的评估结果。"""

    case_name: str
    passed: bool
    answer: str
    failures: list[str] = field(default_factory=list)  # 逐条失败原因


class EvalSuite:
    """评估套件：批量跑案例，产出通过率与失败明细。"""

    def __init__(self, cases: list[EvalCase], agent):
        """agent: callable(question) -> str（被测的 Agent）。"""
        self.cases = cases
        self.agent = agent

    def evaluate_one(self, case: EvalCase) -> EvalResult:
        answer = self.agent(case.question)
        failures: list[str] = []
        for keyword in case.must_contain:
            if keyword not in answer:
                failures.append(f"缺少关键信息「{keyword}」")
        for forbidden in case.must_not_contain:
            if forbidden in answer:
                failures.append(f"出现禁用内容「{forbidden}」（疑似编造）")
        if case.pattern and not re.search(case.pattern, answer):
            failures.append(f"未匹配格式要求: {case.pattern}")
        return EvalResult(case_name=case.name, passed=not failures,
                          answer=answer, failures=failures)

    def run(self) -> tuple[list[EvalResult], float]:
        """跑全部案例，返回 (结果列表, 通过率)。"""
        results = [self.evaluate_one(case) for case in self.cases]
        pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
        return results, pass_rate


def weather_agent(question: str) -> str:
    """被测 Agent（用 Mock 模拟：天气题给固定回答，其他题含糊）。"""
    db = {"北京": "北京今天多云，8~15°C，建议带件外套。"}
    for city, answer in db.items():
        if city in question:
            return answer
    return "抱歉，我不太确定。"  # 兜底


def main() -> None:
    banner("示例 02：评估数据集与评估器")

    cases = [
        EvalCase(name="北京天气-关键信息", question="北京今天天气怎么样？",
                 must_contain=["北京", "多云"], must_not_contain=["晴"]),
        EvalCase(name="北京天气-格式", question="北京天气如何？",
                 pattern=r"北京.*\d+~\d+°C"),
        EvalCase(name="未知城市-诚实拒答", question="拉萨今天天气怎么样？",
                 must_contain=["不确定"], must_not_contain=["晴", "多云"]),
    ]

    suite = EvalSuite(cases=cases, agent=weather_agent)
    results, pass_rate = suite.run()

    section("评估报告（如实展示失败）")
    for r in results:
        mark = "✅" if r.passed else "❌"
        print(f"  {mark} {r.case_name}: {'通过' if r.passed else '失败 ' + str(r.failures)}")
        if not r.passed:
            print(f"     实际答案: {r.answer}")
    print(f"\n通过率: {pass_rate:.0%}（{sum(1 for r in results if r.passed)}/{len(results)}）")

    verify("案例 1 通过（关键信息齐全且未编造'晴'）", results[0].passed)
    verify("案例 2 通过（格式匹配温度区间）", results[1].passed)
    verify("案例 3 失败（兜底回答不含'不确定'，被评估器抓出）",
           not results[2].passed and any("缺少关键信息「不确定」" in f for f in results[2].failures))
    verify("通过率如实计算为 2/3", abs(pass_rate - 2 / 3) < 1e-9)

    section("评估驱动开发：修复后重跑，分数说话")
    def weather_agent_v2(question: str) -> str:
        db = {"北京": "北京今天多云，8~15°C，建议带件外套。"}
        for city, answer in db.items():
            if city in question:
                return answer
        return "抱歉，我不确定，暂时无法回答该城市的天气。"

    suite_v2 = EvalSuite(cases=cases, agent=weather_agent_v2)
    results_v2, pass_rate_v2 = suite_v2.run()
    for r in results_v2:
        print(f"  {'✅' if r.passed else '❌'} {r.case_name}")
    print(f"\nv2 通过率: {pass_rate_v2:.0%}")
    verify("修复兜底话术后通过率提升到 3/3", pass_rate_v2 == 1.0)
    verify("评估是回归测试：老案例依然全部通过", all(r.passed for r in results_v2))

    finish("ch10_evaluators")


if __name__ == "__main__":
    main()
