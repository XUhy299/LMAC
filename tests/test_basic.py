"""
测试示例
"""
import pytest


def test_import():
    """测试包能否正常导入"""
    import agentic_rag
    assert agentic_rag.__version__ == "2.0.0"


def test_config():
    """测试配置模块"""
    from agentic_rag import config
    assert hasattr(config, "API_PORT")
    assert hasattr(config, "DATABASE_URL")
