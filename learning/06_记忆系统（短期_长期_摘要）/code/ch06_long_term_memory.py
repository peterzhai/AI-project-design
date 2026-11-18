"""示例 02：长期记忆 —— 被窗口挤出的信息，按条目存起来、按相关性取回来。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/06_记忆系统（短期_长期_摘要）/code/ch06_long_term_memory.py

滑动窗口丢掉了历史，但用户在第 3 轮说过"我对花生过敏"——第 20 轮推荐菜谱时
必须想起来。长期记忆三板斧：
1. 存取：记忆条目（key + 内容 + 元数据）；
2. 检索：按关键词打分的简化版相关性检索（第 7 章升级为向量检索）；
3. 回注：检索到的记忆作为 system 附加段注入上下文。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402


@dataclass
class MemoryItem:
    """一条长期记忆。"""

    key: str            # 简短标题，如 "饮食禁忌"
    content: str        # 完整内容
    turn: int           # 记录时的对话轮次（追溯用）


@dataclass
class LongTermMemory:
    """极简长期记忆：列表存储 + 关键词打分检索。

    教学版检索：对 query 与条目做词级重合计分（Python 标准库即可）。
    生产版会换成向量检索（第 7 章 RAG 讲原理，此处先建立"检索回注"的直觉）。
    """

    items: list[MemoryItem] = field(default_factory=list)
    _next_turn: int = 1

    def remember(self, key: str, content: str) -> None:
        """写入一条记忆（同 key 覆盖旧值 —— 记忆要保鲜，不要堆积矛盾）。"""
        for i, item in enumerate(self.items):
            if item.key == key:
                self.items[i] = MemoryItem(key=key, content=content, turn=self._next_turn)
                self._next_turn += 1
                return
        self.items.append(MemoryItem(key=key, content=content, turn=self._next_turn))
        self._next_turn += 1

    def search(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        """按词级重合度检索最相关的 top_k 条记忆。"""
        query_words = set(query)

        def score(item: MemoryItem) -> int:
            text = item.key + item.content
            return len(query_words & set(text))

        ranked = sorted(self.items, key=score, reverse=True)
        return [item for item in ranked if score(item) > 0][:top_k]

    def recall_for(self, query: str, top_k: int = 3) -> str:
        """把检索结果渲染成可注入 system 的附加段。"""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = [f"- [{item.key}] {item.content}" for item in hits]
        return "以下是关于该用户的已知记忆，回答时必须遵守：\n" + "\n".join(lines)


def main() -> None:
    banner("示例 02：长期记忆")

    memory = LongTermMemory()
    # 模拟多轮对话中"记住"的关键事实
    memory.remember("饮食禁忌", "用户对花生过敏，所有推荐必须避开花生")
    memory.remember("口味偏好", "用户喜欢清淡口味，讨厌香菜")
    memory.remember("预算范围", "晚餐预算人均 80 元以内")
    memory.remember("地址", "用户曾住在朝阳区")

    # 记忆保鲜：同 key 更新覆盖（key 相同才会覆盖，key 不同是两条记忆）
    memory.remember("地址", "用户现住在海淀区")
    print("当前记忆条目:")
    for item in memory.items:
        print(f"  [{item.key}] {item.content}")

    section("实验 1：相关性检索")
    hits = memory.search("帮用户推荐一个晚餐菜谱")
    print(f"query='帮用户推荐一个晚餐菜谱' 命中 {len(hits)} 条: {[h.key for h in hits]}")
    verify("最相关的饮食禁忌排第一（过敏是硬约束）", hits[0].key == "饮食禁忌")
    verify("口味偏好被召回", any(h.key == "口味偏好" for h in hits))
    verify("同 key 更新覆盖（地址只剩一条且是新值）",
           len([item for item in memory.items if item.key == "地址"]) == 1
           and "海淀区" in memory.items[3].content)

    section("实验 2：检索结果回注上下文")
    recall_block = memory.recall_for("推荐晚餐", top_k=2)
    print(recall_block)
    verify("回注块是可注入 system 的文本", recall_block.startswith("以下是关于该用户的已知记忆"))
    verify("过敏约束在回注块中", "花生" in recall_block)

    section("实验 3：与滑动窗口组合（完整记忆架构）")
    # 短期窗口 + 长期记忆 = 上下文 = system + 记忆附加段 + 最近对话
    system_prompt = "你是晚餐推荐助手。\n" + memory.recall_for("推荐晚餐", top_k=2)
    context = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": "今晚吃什么？"}]
    print(f"注入记忆后的 system 提示词共 {len(system_prompt)} 字符，包含 {len(memory.search('推荐晚餐', top_k=2))} 条记忆")
    verify("system 同时包含角色与记忆", "晚餐推荐助手" in system_prompt and "花生" in system_prompt)

    finish("ch06_long_term_memory")


if __name__ == "__main__":
    main()
