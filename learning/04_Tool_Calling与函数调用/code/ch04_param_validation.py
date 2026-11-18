"""示例 02：参数校验 —— 脏参数必须在"进车间"之前被拦下。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/04_Tool_Calling与函数调用/code/ch04_param_validation.py

模型给的参数是"口头订单"，可能缺斤短两（缺必填）、货不对板（类型错）、
甚至指鹿为马（调不存在的工具）。ToolBox 的三层防线：
    1) 工具存在性  ->  2) 参数 schema 校验  ->  3) 执行异常兜底
全部变成可读的"观察"回传给模型，Agent 本体永不崩溃。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.toolbox import ToolBox  # noqa: E402


def build_toolbox() -> tuple[ToolBox, list[str]]:
    """构造带副作用审计的工具箱：audit 列表记录每次真实执行。"""
    audit: list[str] = []

    def cancel_order(order_id: str, reason: str = "用户取消") -> str:
        """取消订单。

        order_id 为订单编号；reason 为取消原因。
        """
        audit.append(f"cancel_order({order_id}, {reason})")
        return f"订单 {order_id} 已取消"

    def refund(amount: float, currency: str = "CNY") -> str:
        """发起退款，金额必须为正数。"""
        if amount <= 0:
            raise ValueError(f"退款金额必须为正数，收到 {amount}")
        audit.append(f"refund({amount}, {currency})")
        return f"已退款 {amount} {currency}"

    box = ToolBox().register(cancel_order).register(refund)  # 链式注册
    return box, audit


def main() -> None:
    banner("示例 02：参数校验与错误分类")

    box, audit = build_toolbox()

    # 防线 1：工具存在性
    section("防线 1：模型点名了不存在的工具")
    obs = box.execute("delete_user", {"user_id": "u1"})
    print(f"观察: {obs}")
    verify("未注册工具被拦截", "未注册的工具" in obs and "cancel_order" in obs)
    verify("没有触发任何真实执行", audit == [])

    # 防线 2a：缺必填参数
    section("防线 2a：缺必填参数（函数没有被执行！）")
    obs2 = box.execute("cancel_order", {})
    print(f"观察: {obs2}")
    verify("缺 order_id 被点名", "缺少必填字段「order_id」" in obs2)
    verify("audit 为空：脏参数没进车间", audit == [])

    # 防线 2b：类型错误
    section("防线 2b：类型错误（order_id 给成了数字）")
    obs3 = box.execute("cancel_order", {"order_id": 12345})
    print(f"观察: {obs3}")
    verify("$.order_id 期望类型 string", "$.order_id: 期望类型 string" in obs3)
    verify("audit 仍为空", audit == [])

    # 防线 2c：可选参数正常省略 + 默认值生效
    section("防线 2c：合法调用（可选参数省略，默认值生效）")
    obs4 = box.execute("cancel_order", {"order_id": "A1001"})
    print(f"观察: {obs4}")
    verify("订单取消成功", obs4 == "订单 A1001 已取消")
    verify("audit 记录了真实执行与默认 reason", audit == ["cancel_order(A1001, 用户取消)"])

    # 防线 3：业务异常兜底（参数合法但业务拒绝）
    section("防线 3：业务规则异常（负数退款）-> 观察，不崩溃")
    obs5 = box.execute("refund", {"amount": -50})
    print(f"观察: {obs5}")
    verify("ValueError 被转为观察文本", "退款金额必须为正数" in obs5)
    verify("失败的调用没有进入 audit", len(audit) == 1)  # 只有刚才那次 cancel 成功执行

    # 正常退款
    obs6 = box.execute("refund", {"amount": 199.5, "currency": "USD"})
    print(f"观察: {obs6}")
    verify("合法退款执行成功", obs6 == "已退款 199.5 USD")
    verify("audit 完整记录两次真实执行", len(audit) == 2)

    print("\n要点回顾：校验在执行之前；错误是'观察'不是'崩溃'；副作用要可审计。")

    finish("ch04_param_validation")


if __name__ == "__main__":
    main()
