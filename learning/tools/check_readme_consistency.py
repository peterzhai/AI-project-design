"""讲义一致性校验工具：README 中的 python 代码块必须逐字存在于仓库某个 .py 文件中。

用法（从仓库根目录）:
    python3 learning/tools/check_readme_consistency.py            # 检查所有章节
    python3 learning/tools/check_readme_consistency.py 01_导论与心智模型  # 检查单章

退出码：0 = 全部一致；1 = 存在脱节（讲义引用与代码不符）。
"""

import re
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[1]


def extract_python_blocks(readme_text: str) -> list[str]:
    """提取 README 中所有 ```python 围栏代码块。"""
    return re.findall(r"```python\n(.*?)```", readme_text, re.S)


def main(chapters: list[Path]) -> int:
    # 索引仓库中所有 .py 文件内容（教程规模小，全量扫描即可）。
    sources = {p: p.read_text(encoding="utf-8") for p in sorted(LEARNING_DIR.rglob("*.py"))}
    checked = failures = 0
    for chapter in chapters:
        readme = chapter / "README.md"
        if not readme.exists():
            continue
        for i, block in enumerate(extract_python_blocks(readme.read_text(encoding="utf-8")), 1):
            checked += 1
            hits = [str(p.relative_to(LEARNING_DIR)) for p, src in sources.items() if block.rstrip() in src]
            if hits:
                print(f"{chapter.name} 代码块{i}: ✅ 逐字匹配 {hits[0]}")
            else:
                failures += 1
                print(f"{chapter.name} 代码块{i}: ❌ 脱节！开头: {block.strip()[:60]!r}")
    print(f"共检查 {checked} 个代码块，失败 {failures} 个。")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [LEARNING_DIR / name for name in sys.argv[1:]]
    else:
        targets = [
            d for d in sorted(LEARNING_DIR.iterdir())
            if d.is_dir() and (d / "README.md").exists()
        ]
    raise SystemExit(main(targets))
