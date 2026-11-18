"""示例 02：编排者-执行者 —— Orchestrator 拆任务给多个 Worker，再汇总。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/09_多Agent协作/code/ch09_orchestrator.py

与路由器的区别：路由器是"转给一个人"；编排者是"拆成几份、发给多人、
收回结果、汇总成最终答案"。适合可并行拆分的任务（多市场调研、多文档摘要）。
关键设计：子任务的输入输出都是结构化消息 —— 可审计、可并行、可替换。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


@dataclass
class SubTask:
    """编排者拆出的子任务（消息化：谁、做什么、产出什么）。"""

    worker_name: str
    instruction: str
    output: str = ""


@dataclass
class Worker:
    """执行者：只管自己那份指令。"""

    name: str
    llm: MockLLM
    calls: list[str] = field(default_factory=list)

    def work(self, instruction: str) -> str:
        self.calls.append(instruction)
        return self.llm.chat([{"role": "user", "content": instruction}]).content


class Orchestrator:
    """编排者：拆任务（LLM）-> 分派 -> 收结果 -> 汇总（LLM）。"""

    def __init__(self, planner_llm: MockLLM, synthesizer_llm: MockLLM):
        self.planner_llm = planner_llm    # 负责拆分
        self.synthesizer_llm = synthesizer_llm  # 负责汇总

    def plan(self, task: str) -> list[SubTask]:
        """让 Planner LLM 把任务拆成子任务列表（JSON 输出，复用第 3 章约定）。"""
        response = self.planner_llm.chat([
            {"role": "system", "content":
                "你是任务编排者。把任务拆成子任务，只输出 JSON："
                '{"subtasks": [{"worker": "执行者名", "instruction": "具体指令"}]}。'
                "每个子任务必须自包含（不依赖其他子任务的产出）。"},
            {"role": "user", "content": task},
        ])
        from common.json_utils import parse_llm_json
        data = parse_llm_json(response.content)
        return [SubTask(worker_name=item["worker"], instruction=item["instruction"])
                for item in data["subtasks"]]  # type: ignore[index]

    def synthesize(self, task: str, subtasks: list[SubTask]) -> str:
        """把各 Worker 的产出汇总成最终答案。"""
        report = "\n".join(f"【{t.worker_name}】{t.output}" for t in subtasks)
        response = self.synthesizer_llm.chat([
            {"role": "user", "content": f"总任务：{task}\n各执行者汇报：\n{report}\n请汇总成一份最终答案。"},
        ])
        return response.content

    def run(self, task: str, workers: dict[str, Worker]) -> tuple[str, list[SubTask]]:
        """完整编排：拆 -> 派 -> 收 -> 汇。"""
        subtasks = self.plan(task)
        for sub in subtasks:
            if sub.worker_name not in workers:
                sub.output = f"（未找到执行者 {sub.worker_name}，该子任务被跳过）"
                continue
            sub.output = workers[sub.worker_name].work(sub.instruction)
        return self.synthesize(task, subtasks), subtasks


def main() -> None:
    banner("示例 02：编排者-执行者")

    # Worker：市场调研员（各自只懂一个市场）
    cn_worker = Worker(name="中国区调研员", llm=MockLLM(handler=lambda m, t: {
        "content": "中国区：偏好高端智能款，价格敏感度中等，渠道以电商直播为主。"}))
    eu_worker = Worker(name="欧洲区调研员", llm=MockLLM(handler=lambda m, t: {
        "content": "欧洲区：重视隐私合规与能效标签，偏好中端机型。"}))
    us_worker = Worker(name="北美区调研员", llm=MockLLM(handler=lambda m, t: {
        "content": "北美区：运营商渠道占比高，偏好大屏与拍照。"}))
    workers = {w.name: w for w in (cn_worker, eu_worker, us_worker)}

    # Planner：把调研任务拆成三个并行子任务（JSON 计划）
    planner = MockLLM(handler=lambda m, t: {
        "content": '{"subtasks": ['
                   '{"worker": "中国区调研员", "instruction": "调研中国区智能手表市场偏好"},'
                   '{"worker": "欧洲区调研员", "instruction": "调研欧洲区智能手表市场偏好"},'
                   '{"worker": "北美区调研员", "instruction": "调研北美区智能手表市场偏好"}]}'
    })
    synthesizer = MockLLM(handler=lambda m, t: {
        "content": "汇总结论：三区共同点是偏好智能款；差异上中国重直播渠道、欧洲重合规、北美重运营商，建议差异化投放。"
    })
    orchestrator = Orchestrator(planner, synthesizer)

    final_answer, subtasks = orchestrator.run("调研智能手表在三大市场的用户偏好，给出汇总建议", workers)
    print(f"拆出 {len(subtasks)} 个子任务:")
    for t in subtasks:
        print(f"  -> {t.worker_name}: {t.instruction}")
    print(f"\n最终汇总: {final_answer}")

    section("自检")
    verify("拆出 3 个子任务且各归各的执行者", len(subtasks) == 3 and
           {t.worker_name for t in subtasks} == set(workers))
    verify("每个 Worker 恰好被调用 1 次（指令自包含，可并行）",
           all(len(w.calls) == 1 for w in workers.values()))
    verify("子任务产出全部回填", all(t.output for t in subtasks))
    verify("汇总阶段的输入包含各 Worker 的汇报",
           "中国区调研员" in synthesizer.calls[0]["messages"][0]["content"]
           and "欧洲区" in synthesizer.calls[0]["messages"][0]["content"])
    verify("Planner/汇总器职责分离（两个不同 LLM）",
           orchestrator.planner_llm is not orchestrator.synthesizer_llm)

    section("异常路径：分派给不存在的 Worker")
    planner_bad = MockLLM(scripted=[
        {"content": '{"subtasks": [{"worker": "火星区调研员", "instruction": "调研火星市场"}]}'},
    ])
    orchestrator_bad = Orchestrator(planner_bad, MockLLM(scripted=["汇总：仅部分市场有数据，其余缺失。"]))
    answer_bad, subtasks_bad = orchestrator_bad.run("调研火星市场", workers)
    print(f"异常路径答案: {answer_bad}")
    verify("未知 Worker 的子任务被安全跳过（不崩溃）",
           "未找到执行者" in subtasks_bad[0].output)

    finish("ch09_orchestrator")


if __name__ == "__main__":
    main()
