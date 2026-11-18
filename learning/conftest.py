"""pytest 通用配置：把每章的 code/ 目录加入 sys.path，测试即可直接 import 章节模块。

模块命名约定：每章代码文件以 chNN_ 开头（如 ch01_hello_agent.py），
保证跨章节导入时不会发生模块名冲突。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parent
COMMON_DIR = LEARNING_DIR / "common"

# common/ 目录本身也需要可导入（示例脚本中通过 parents[2] 已注入 learning/，
# 这里兜底保证测试进程里 `import common.xxx` 一定可用）。
for _p in (LEARNING_DIR, COMMON_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 把每个章节的 code/ 目录加入 sys.path（排序保证确定性）。
for _code_dir in sorted(LEARNING_DIR.glob("[0-9][0-9]_*/code")):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
for _code_dir in sorted(LEARNING_DIR.glob("capstone_*/code")):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
