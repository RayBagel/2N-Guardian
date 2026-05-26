# 2N-Guardian Reward Spec (Step 1 Freeze)

## 1. Goal
构建可复现的法律推理奖励规范，用于 GRPO 训练中对回答进行可解释打分，核心目标是提升：
- 事实识别准确性（Fact）
- 法条匹配准确性（Law）
- 逻辑链完整性（Process）

## 2. Final Formula
所有子分数归一化到 `[0, 1]`：

`Total_Reward = 0.4 * Score_Fact + 0.3 * Score_Law + 0.3 * Score_Process`

## 3. Score Definitions
### 3.1 Score_Fact (40%)
输入：证据提取结果 + 规则匹配信号 + 语义融合信号。

建议计算：
- `combined_signal`（规则/语义融合）作为核心项
- `accepted_evidence_ratio` 作为证据密度项

示例：`Score_Fact = clip01(0.75 * combined_signal + 0.25 * accepted_evidence_ratio)`

### 3.2 Score_Law (30%)
输入：法条抽取结果、2N 场景适用性。

检查项：
- 是否出现明确条号（如 `第39/40/47/87条`）
- 2N 场景下是否混淆补偿金与赔偿金
- 条文与结论是否语义冲突

### 3.3 Score_Process (30%)
输入：逻辑链校验结果。

两环校验：
- Ring1: `事实 -> 法条`
- Ring2: `法条 -> 结论`

软惩罚（当前版本）：
- `fact_law_penalty = 0.3` if Ring1 fail else `1.0`
- `law_conclusion_penalty = 0.5` if Ring2 fail else `1.0`
- `Score_Process = base_logic_score * fact_law_penalty * law_conclusion_penalty`

## 4. Label-Aware Constraints
- `label=lose`：不应奖励“2N必然成立”的结论。
- `label=win`：可奖励“2N/N+1”并要求给出法条与事实支撑。

## 5. Anti-Hacking Rules
- 仅关键词命中不算通过，必须有“事实-法条-结论”一致性。
- 输出结构必须可解析（IRAC 标签完整，分析段长度达标）。

## 6. Reproducibility
- 冻结训练集与评测集后，记录：样本数、标签分布、SHA256。
- 任何训练前改动需更新 manifest。
