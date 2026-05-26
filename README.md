# 2N-Guardian

**法律大模型可解释奖励引擎 · Legal LLM Explainable Reward Engine**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/RL-GRPO%2FRLHF-orange)](https://arxiv.org/abs/2402.03300)
[![Domain](https://img.shields.io/badge/Domain-Legal%20AI-green)]()

[简体中文](#简体中文) | [English](#english)

---

## 简体中文

### 🎯 一句话定位

**教大模型像法官一样思考** ——把法官判案时的底层逻辑（事实认定 / 法律适用 / 推理过程）代码化为大模型可训练和评分的三维奖励函数，从对齐层根治通用大模型在民商事场景的"法条幻觉"与"推理黑盒"问题。

> 本项目是**法律场景大模型对齐的基础设施研究**，而非简单的 RAG 检索增强或提示词工程。核心贡献在于把司法三段论（事实 → 法律 → 结论）的裁判逻辑代码化为可执行的 GRPO 奖励信号。

---

### 🔥 核心问题：为什么通用大模型不适合直接用于法律场景？

| 问题 | 表现 | 根因 |
|---|---|---|
| 法条幻觉 | 模型编造不存在的法律条款 | 训练数据未经法律规则校验 |
| 推理黑盒 | 说不清楚为什么这么判 | 无可解释的推理评分机制 |
| 场景泛化弱 | 换案由就不会了 | 缺乏跨案由的通用裁判规则抽象 |
| 合规风险高 | 给出违反执业规范的建议 | 未对齐法律职业伦理边界 |

市面上的方案（RAG、微调）都是**打补丁**。2N-Guardian 的核心判断是：**法律 AI 的瓶颈不在"不知道法条"，在"不知道法官怎么思考"**——必须从训练阶段把司法三段论变成模型能消化的奖惩信号。

---

### 🏗️ 核心架构：三维可解释奖励函数

```
输入：案件材料（事实陈述 + 争议焦点 + 证据列表）
                │
                ▼
    ┌─────────────────────────────────┐
    │         三维奖励函数             │
    │                                 │
    │  Fact Score    Law Score   Process Score
    │  事实认定维度   法律适用维度   推理流程维度
    │  [0.0 ~ 1.0]  [0.0 ~ 1.0]  [0.0 ~ 1.0]
    │                                 │
    │  ← 规则匹配 + 语义融合双通道 →  │
    └─────────────────────────────────┘
                │
                ▼
    GRPO 训练信号（可执行奖惩信号）
                │
                ▼
    可解释推理输出（每步推理可追溯到规则依据）
```

**三个维度各自独立评分**，实现"哪一步推理错了、错在哪里"的精准定位——这是法律 AI 实现可审计、可追溯、可校验的工程基础。

---

### ⚙️ 关键技术模块

**Module A · data_engine —— RLAIF 闭环工程化合成数据管线**
> 搭了一条 AI 训练数据生产流水线：司法规则深度调研 → 黄金种子样本撰写 → 多维度场景泛化 → 奖励函数自动化清洗校验

- 已完成劳动争议赛道 **96 条高质量 IRAC 标注冻结黄金数据集**
- 整套管线可无缝复制至合同纠纷、股权纠纷、民间借贷、知识产权等全民商事案由
- 黄金数据集经民商事生效裁判规则全量校验

**Module B · logic_core —— 核心奖励函数与推理校验引擎**
> 包含两大配套引擎：

- **EvidenceTracker 证据追踪引擎**：30% 规则匹配 + 70% 语义融合双通道架构，自动提取案件事实要素
- **SyllogismEvaluator 三段论校验引擎**：把民商事裁判规则代码化为可执行的逻辑校验链路

**Module C · train_hub —— GRPO 训练脚本**
> 基于 Unsloth + GRPO 的完整训练流程，支持 SiliconFlow API 接入

**Module D · interface —— 可交互工程化验证 Demo**
> 基于 Streamlit 搭建，支持公网安全脱敏版 / 本地完整版双模式切换

---

### 📂 项目结构

```
2N-Guardian/
├── src/
│   ├── data_engine/        # Module A: RLAIF 合成数据管线
│   ├── logic_core/         # Module B: 三维奖励函数 + 推理校验引擎
│   ├── train_hub/          # Module C: GRPO 训练脚本（Unsloth）
│   └── interface/          # Module D: Streamlit Demo
├── data/
│   ├── seeds.jsonl         # 黄金种子样本
│   └── train.jsonl         # 完整训练数据集
├── v2_mvp/
│   ├── src/                # V2 MVP 核心代码
│   ├── rl/                 # RL 配置与脚本
│   └── output/             # 训练数据产物（2n_golden_dataset_v1.jsonl 等）
├── knowledge_base/
│   └── raw_pdfs/           # 20+ 份最高法指导案例与典型案例（公开来源）
├── tests/                  # 测试用例与 runner
├── configs/                # YAML 配置文件
├── scripts/                # PDF 解析、环境检查等工具脚本
├── docs/                   # 项目文档与设计草图
├── app.py                  # 主 Demo 入口
├── requirements.txt        # 依赖列表
└── .env.example            # API Key 配置模板
```

---

### 🚀 快速开始

**1. 克隆仓库**
```bash
git clone https://github.com/YOUR_USERNAME/2N-Guardian.git
cd 2N-Guardian
```

**2. 安装依赖**
```bash
pip install -r requirements.txt
```

**3. 配置 API Key**
```bash
cp .env.example .env
# 编辑 .env，填入你的 SiliconFlow API Key
# SILICONFLOW_API_KEY="sk-your-key-here"
```

**4. 运行 Demo**
```bash
# 方式一：主 Demo（推荐）
streamlit run app.py

# 方式二：完整界面
streamlit run src/interface/main.py
```

**5. 运行数据管线**
```bash
bash run_data_pipeline.sh
```

---

### 📊 数据说明

`knowledge_base/raw_pdfs/` 中的判例 PDF 均来自公开来源：
- **最高人民法院指导性案例**（第 180、181 号等）
- **人力资源社会保障部、最高人民法院联合发布典型案例**
- **各省市中级人民法院发布的涉民生典型案例**

---

### 📄 技术白皮书

完整的架构设计、RLAIF 实现细节、三维奖励函数数学推导见：
> **《2N-Guardian 法律大模型可解释奖励系统技术白皮书》**（PDF，作者提供，欢迎邮件索取）

---

### 🤝 作者

**吴昊雨 (Wu Haoyu)**
- 执业律师（北京市尚公律师事务所·主办律师），6 年民商事诉讼经验
- 独立 AI 研究者，专注法律场景大模型对齐与知识工程
- Email: why1900@163.com

---

## English

### 🎯 One-liner

**Teaching LLMs to reason like judges** — encoding judicial syllogism logic (fact-finding / law application / reasoning process) into a three-dimensional, explainable reward function for GRPO training, solving the root cause of "legal hallucination" and "black-box reasoning" in general-purpose LLMs.

> This project is **foundational alignment infrastructure for legal LLMs**, not simple RAG or prompt engineering. The core contribution is encoding civil/commercial adjudication rules into executable GRPO reward signals.

---

### 🔥 Why general LLMs fail in legal scenarios

| Problem | Symptom | Root Cause |
|---|---|---|
| Legal hallucination | Model fabricates non-existent statutes | Training data lacks legal rule validation |
| Black-box reasoning | Cannot explain why it reached a conclusion | No explainable reasoning scoring mechanism |
| Weak generalization | Breaks down across different case types | Lacks abstraction of cross-domain adjudication rules |
| Compliance risk | Gives advice violating professional ethics | Not aligned with legal professional standards |

RAG and fine-tuning are **patches**. 2N-Guardian's core thesis: **the bottleneck isn't "not knowing the law" — it's "not knowing how judges think"**. The fix must happen at the training alignment layer.

---

### 🏗️ Architecture: 3D Explainable Reward Function

Three independent scoring dimensions — **Fact Score**, **Law Score**, **Process Score** — each producing a [0.0, 1.0] signal that feeds directly into GRPO training. Every step of the model's legal reasoning becomes auditable, traceable, and verifiable.

---

### ⚙️ Key Modules

| Module | Description |
|---|---|
| `data_engine` | RLAIF closed-loop synthetic data pipeline — 96 IRAC-annotated golden samples (labor dispute track), replicable to all civil/commercial case types |
| `logic_core` | Core reward functions + EvidenceTracker (rule-matching + semantic fusion) + SyllogismEvaluator (adjudication rule codification) |
| `train_hub` | GRPO training scripts with Unsloth integration |
| `interface` | Streamlit interactive demo (public sanitized + local full version) |

---

### 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/2N-Guardian.git
cd 2N-Guardian
pip install -r requirements.txt
cp .env.example .env   # Add your SiliconFlow API key
streamlit run app.py
```

---

### 📄 Technical Whitepaper

Full architecture design, RLAIF implementation details, and reward function mathematical derivation:
> **"2N-Guardian Legal LLM Explainable Reward System — Technical Whitepaper"** (available upon request: why1900@163.com)

---

### 🤝 Author

**Wu Haoyu (吴昊雨)**
- Practicing Attorney (Beijing Shanggong Law Firm · Lead Attorney), 6 years civil/commercial litigation experience
- Independent AI Researcher, focusing on legal LLM alignment and knowledge engineering
- Email: why1900@163.com

---

### 📜 License

Apache 2.0 — see [LICENSE](LICENSE)

---

### ⭐ If this helps your research

Please star ⭐ the repo and cite this project in your work. Issues and PRs welcome.
