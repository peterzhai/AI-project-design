"""learning 教程公共工具包。

包含四件基础设施，后续所有章节都建立在其之上：
- mock_llm:    离线假 LLM（脚本化响应 / 自定义 handler），无网络也能跑通一切。
- llm_client:  统一客户端工厂，USE_REAL_LLM=1 时切换到真实 OpenAI 兼容接口。
- logging_utils: 统一日志格式。
- check:       示例脚本的自检助手。
"""
