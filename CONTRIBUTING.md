# 贡献指南

感谢你对 Agentic-RAG 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果你发现了 bug 或有功能建议，请通过 GitHub Issues 提交：

1. 检查是否已有相似的 issue
2. 使用相应的 issue 模板
3. 提供详细的描述和复现步骤

### 提交代码

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的修改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

### 开发环境设置

```bash
# 克隆你的 fork
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 代码规范

- 遵循 PEP 8 规范
- 使用 Black 格式化代码：`black agentic_rag/`
- 使用 isort 排序导入：`isort agentic_rag/`
- 添加适当的类型注解
- 编写文档字符串

### 测试

```bash
# 运行测试
pytest tests/ -v

# 运行带覆盖率报告的测试
pytest tests/ --cov=agentic_rag --cov-report=html
```

## 行为准则

请保持友善和尊重，共同维护一个开放的协作环境。
