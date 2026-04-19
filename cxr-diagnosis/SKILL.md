---
name: 胸部X光诊断
description: 自动化胸部X光图（CXR）诊断工具。当用户请求"诊断胸部X光图"、"诊断CXR"、"分析胸片"、"X光诊断"等类似需求时触发。自动查找最新的胸部X光图片，运行BioViL深度学习模型进行疾病检测，并返回诊断结果。支持的疾病包括：心脏增大、肺不张、胸腔积液、肺炎、气胸、肺水肿、肺实变、肺病变等。
---

# 胸部X光诊断

## 概述

这是一个自动化胸部X光诊断工具，使用BioViL深度学习模型对胸部X光图像进行多标签疾病分类和检测。当用户需要进行胸部X光诊断时，该技能会自动完成以下流程：

1. 从指定目录查找最新的CXR图片
2. 激活conda虚拟环境（torch-gpu）
3. 运行BioViL诊断脚本
4. 解析诊断结果并返回阳性发现

## 工作流程

### 触发条件

用户说出以下任何类似表述时触发：
- "诊断胸部X光图"
- "诊断CXR"
- "分析胸片"
- "X光诊断"
- "帮我看看这张胸片"
- "这张胸部X光有问题吗"

### 执行步骤

**步骤1：查找最新图片**
- 自动扫描 `D:\CXR` 目录
- 按修改时间排序，选择最新的图片文件
- 支持格式：PNG, JPG, JPEG, BMP, TIFF, DICOM

**步骤2：运行诊断**
- 激活conda环境：`torch-gpu`
- 运行诊断脚本：`D:\python_workspace\MedicalClaw\BioViL.py`
- 自动输入图片路径

**步骤3：解析结果**
诊断脚本会返回如下格式的表格：
```
============================================================
Disease Category | Prediction | Pos Sim
============================================================
Enlarged Cardiomediastastinum | Negative | -0.3501
Cardiomegaly | Negative | -0.2513
Lung Lesion | Positive | 0.3435
...
============================================================
```

**步骤4：返回结果**
- 筛选所有 `Prediction = Positive` 的疾病
- 如果有阳性发现，列出疾病名称和相似度分数
- 如果全部为Negative，返回"未发现明显异常"

## 使用方法

### 标准诊断流程

运行诊断脚本：

```bash
python <skill-path>/scripts/diagnose_cxr.py
```

脚本会自动：
1. 查找 `D:\CXR` 目录中最新的图片
2. 激活 `torch-gpu` 环境
3. 运行BioViL诊断
4. 输出格式化的诊断结果

### 自定义参数

如果需要指定不同的目录或环境：

```bash
python <skill-path>/scripts/diagnose_cxr.py \
  --cxr-dir "D:\CustomPath" \
  --biovil-script "D:\CustomPath\BioViL.py" \
  --conda-env "custom-env"
```

## 输出格式

诊断结果包含两个部分：

### 1. 阳性发现摘要（如果有）

```
🔍 **诊断发现异常：**

• **Lung Lesion**: Positive (相似度: 0.3435)
• **Consolidation**: Positive (相似度: 0.0889)
```

### 2. 完整诊断结果表格

```
| 疾病类别 | 预测结果 | 相似度 |
|---------|---------|--------|
| ⚠️ Lung Lesion | Positive | 0.3435 |
| ✓ Cardiomegaly | Negative | -0.2513 |
| ✓ Lung Opacity | Negative | -0.3145 |
...
```

## 支持的疾病类别

BioViL模型可检测以下疾病：

1. **Enlarged Cardiomediastinum** - 纵隔增宽
2. **Cardiomegaly** - 心脏增大
3. **Lung Opacity** - 肺部阴影
4. **Lung Lesion** - 肺部病变
5. **Edema** - 肺水肿
6. **Consolidation** - 肺实变
7. **Pneumonia** - 肺炎
8. **Atelectasis** - 肺不张
9. **Pneumothorax** - 气胸
10. **Pleural Effusion** - 胸腔积液
11. **Pleural Other** - 其他胸膜疾病
12. **Fracture** - 骨折
13. **Support Devices** - 支持设备（如导管等）

## 错误处理

脚本会自动处理以下错误情况：

- **目录不存在**：提示 "目录不存在: <路径>"
- **目录为空**：提示 "在目录 <路径> 中未找到任何图片文件"
- **环境激活失败**：提示 "诊断脚本执行失败"
- **脚本执行错误**：显示详细错误信息

## 技术细节

### 环境要求

- **Conda环境**：torch-gpu（包含PyTorch GPU支持）
- **Python版本**：3.7+
- **依赖**：BioViL模型和相关依赖（已安装在torch-gpu环境中）

### 性能优化

- 使用 `conda run` 命令确保环境正确激活
- 自动查找最新图片，无需手动指定路径
- 解析输出时使用表格格式识别，提高准确性

## 注意事项

1. **图片质量**：确保CXR图片质量良好，低质量图片可能影响诊断准确性
2. **GPU要求**：torch-gpu环境需要GPU支持，确保CUDA可用
3. **路径空格**：脚本会自动处理包含空格的文件路径
4. **诊断建议**：此工具仅供辅助诊断参考，最终诊断应由专业医生确认

## 示例用法

**用户**：诊断胸部X光图

**Codex**：
```
📁 正在查找 D:\CXR 中的最新CXR图片...
✓ 找到图片: D:\CXR\patient_001.png

🔬 正在运行诊断分析...
  环境: torch-gpu
  脚本: D:\python_workspace\MedicalClaw\BioViL.py

🔍 **诊断发现异常：**

• **Lung Lesion**: Positive (相似度: 0.3435)

---
**完整诊断结果：**

| 疾病类别 | 预测结果 | 相似度 |
|---------|---------|--------|
| ⚠️ Lung Lesion | Positive | 0.3435 |
| ✓ Cardiomegaly | Negative | -0.2513 |
| ✓ Lung Opacity | Negative | -0.3145 |
| ✓ Pneumonia | Negative | -0.0235 |
...
```

## 资源文件

### scripts/diagnose_cxr.py

完整的诊断脚本，包含：
- 图片查找功能
- Conda环境激活
- BioViL脚本调用
- 结果解析和格式化
- 错误处理

直接运行即可完成整个诊断流程。
