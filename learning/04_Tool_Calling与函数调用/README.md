# 第 4 章 · Tool Calling 与函数调用

> 本章目标：把第 1 章的玩具工具箱升级为**正规军** —— 自动生成 JSON Schema 描述、
> 参数校验前置、结构化错误回传，并逐字段还原真实的 Tool-Calling 协议。
>
> 学习时间：约 2~3 小时。前置要求：第 1、3 章。

---

## 1. Tool Calling 是什么：模型"点菜"，代码"做菜"

第 1 章我们用 `tools[call.name](**call.arguments)` 一行就"调了工具"。但真实的
Tool-Calling 是一套严格协议，模型和代码之间隔着两层翻译：

```text
 你的代码                          模型
   │  ① tools 清单（每件工具的      │
   │     JSON Schema 说明书）       │
   ├──────────────────────────────▶│
   │                               │ 读了说明书，决定"点菜"
   │  ② assistant.tool_calls       │
   │    [{id, type, function:{     │
   │        name,                  │
   │        arguments: "JSON字符串" │ ← 注意：线上是字符串！
   │    }}]                        │
   │◀──────────────────────────────┤
   │  ③ 校验参数 → 真正执行         │
   │     → role="tool" 回传结果，   │
   │       tool_call_id 一一对应    │
   ├──────────────────────────────▶│
   │  ④ 模型看到观察，给最终回答    │
   │◀──────────────────────────────┤
```

三件最容易翻车的事，本章逐一解决：

| 翻车点 | 对策 |
| --- | --- |
| 手写 schema 与函数实现脱节 | `function_to_schema` 从签名 + docstring 自动生成 |
| 模型给的脏参数直接打进函数 | `ToolBox.execute` 三层防线，校验在执行之前 |
| 消息组装丢 `tool_call_id` / arguments 忘了转字符串 | 示例 03 逐字段还原协议并断言 |

## 2. 代码走读

### 2.1 示例 01 —— 工具描述自动生成

文件：[code/ch04_schema_generation.py](code/ch04_schema_generation.py)（实现：[common/toolbox.py](../common/toolbox.py)）

工具就是"带类型标注 + docstring 的普通函数"，schema 自动生成。生成规则（摘自 `function_to_schema`，逐字一致）：

```python
    for param_name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"函数 {fn.__name__} 的参数「{param_name}」缺少类型标注，无法生成 schema")
        if annotation not in type_map:
            raise TypeError(
                f"函数 {fn.__name__} 的参数「{param_name}」类型 {annotation} 不受支持"
                f"（支持: int/float/str/bool）"
            )
        properties[param_name] = {"type": type_map[annotation]}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
```

规则一目了然：`int→integer`、`float→number`、`str→string`、`bool→boolean`；带默认值的参数可选；
docstring 第一段就是 description。**缺标注或缺 docstring 直接拒绝注册** —— 不是不支持，
而是强制你写出对模型友好的工具。

示例 1 的核心实验：`function_to_schema(search_flight) == HAND_WRITTEN_SCHEMA` ——
自动生成的结果与手写版**逐字段相等**，证明零信息损失。

### 2.2 示例 02 —— 参数校验：脏参数进不了车间

文件：[code/ch04_param_validation.py](code/ch04_param_validation.py)

`ToolBox.try_execute` 是三层防线（摘自 `common/toolbox.py`，逐字一致；
`execute()` 是它的便捷包装，只取观察文本）：

```python
    def try_execute(self, name: str, arguments: dict) -> tuple[bool, str]:
        """执行并返回 (是否成功, 观察文本)。供程序化判定（如计划执行器重试）。"""
        # 防线 1：工具存在性
        if name not in self._tools:
            return False, f"错误：未注册的工具「{name}」，可用工具: {self.names}"
        tool = self._tools[name]
        # 防线 2：参数 schema 校验（脏参数到不了真实函数）
        param_errors = validate(arguments, tool.schema["function"]["parameters"])
        if param_errors:
            return (False, "错误：参数校验失败，请修正后重试。\n- "
                    + "\n- ".join(param_errors))
        # 防线 3：执行异常
        try:
            return True, str(tool.fn(**arguments))
        except Exception as exc:
            return False, f"工具「{name}」执行失败: {exc}"
```

示例用**副作用审计列表**（`audit`）证明了防线 2 的价值：给 `cancel_order` 传 `order_id=12345`
（类型错）时，`audit == []` —— 真实函数根本没被执行。防线 3 则演示业务异常（负数退款）
同样化为观察文本。

> 复用第 3 章：防线 2 调用的正是 `common/validators.validate`，schema 校验器一处实现两处使用。

### 2.3 示例 03 —— 协议逐字段还原

文件：[code/ch04_tool_protocol.py](code/ch04_tool_protocol.py)

`run_protocol` 把 4 步协议组装成真实消息序列，两个关键细节（摘自源码）：

```python
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),  # 线上是 JSON 字符串！
                    },
                }
                for call in response.tool_calls
            ],
```

```python
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
```

自检逐条断言：`arguments` 在线上是 JSON **字符串**（内存里我们用 dict，上线前必须 dumps）；
`tool_call_id` 与调用的 `id` 一一对应（并行调用时错一个 id，模型就张冠李戴）；
工具清单随**每次**请求发出。进阶实验演示了一个响应并行要两个工具的场景。

> MockLLM 已支持 `tool_call.id`（自动生成 `call_1`、`call_2`，也可在脚本里手工指定）。

## 3. 常见坑

1. **手写 schema 与实现脱节** —— 函数改名了，schema 还是旧的，模型永远调不对。自动生成根治。
2. **校验缺失直接执行** —— `cancel_order(user_id=12345)` 可能真的执行了危险操作。校验前置。
3. **tool 消息不带 `tool_call_id`** —— 并行调用时模型无法区分哪个结果对应哪个调用。
4. **arguments 忘了转 JSON 字符串** —— 内存里 dict 好用，但线上协议要 `json.dumps`，反之解析回 dict。
5. **工具描述含糊**（"处理一下东西"）—— 模型选择工具全靠 description，写得越具体调用越准。
6. **所有工具无脑塞给模型** —— 十几个工具的清单会稀释注意力、增加 token 成本；按场景分组挑选（第 9 章多 Agent 会回到这点）。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch04_schema_generation.py](code/ch04_schema_generation.py) | 示例 01：自动生成 vs 手写 schema |
| [code/ch04_param_validation.py](code/ch04_param_validation.py) | 示例 02：三层防线与副作用审计 |
| [code/ch04_tool_protocol.py](code/ch04_tool_protocol.py) | 示例 03：协议消息逐字段还原 + 并行调用 |
| [tests/test_ch04_examples.py](tests/test_ch04_examples.py) | 9 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

公共层新组件：`common/toolbox.py`（`function_to_schema` + `ToolBox` 三层防线）；
`common/mock_llm.py` 的 `ToolCall` 新增 `id` 字段。

---

> 下一篇：[第 5 章 · ReAct 循环](../05_ReAct循环/README.md) ——
> 把循环升级为显式的"思考 → 行动 → 观察"三段式，让 Agent 的每一步可读、可调、可回放。
