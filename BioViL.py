import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# 引入 health_multimodal 库
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType

# ==========================================
# 1. 定义 13 种疾病的 正向(Pos) 提示词库
# 移除了负向(Neg)提示词，仅保留正向描述
# ==========================================
CHEXPERT_PROMPTS = {
    "Enlarged Cardiomediastinum": {
        "pos": ["Enlarged cardiomediastinum.", "Widening of the mediastinum.",
                "The cardiomediastinal silhouette is enlarged."]
    },
    "Cardiomegaly": {
        "pos": ["Cardiomegaly is present.", "The heart size is enlarged.", "Enlarged cardiac silhouette."]
    },
    "Lung Opacity": {
        "pos": ["Lung opacity.", "Increased opacity in the lungs.", "Patchy airspace opacities."]
    },
    "Lung Lesion": {
        "pos": ["Lung lesion.", "Pulmonary nodule is present.", "Focal pulmonary mass."]
    },
    "Edema": {
        "pos": ["Pulmonary edema.", "Congestive heart failure with pulmonary edema.", "Interstitial edema."]
    },
    "Consolidation": {
        "pos": ["Consolidation.", "Airspace consolidation.", "Focal lung consolidation."]
    },
    "Pneumonia": {
        "pos": ["Pneumonia.", "Findings consistent with pneumonia.", "Focal infiltrate suggestive of pneumonia."]
    },
    "Atelectasis": {
        "pos": ["Atelectasis.", "Lung collapse or atelectasis.", "Bands of atelectasis."]
    },
    "Pneumothorax": {
        "pos": ["Pneumothorax.", "Air in the pleural space.", "Collapsed lung due to pneumothorax."]
    },
    "Pleural Effusion": {
        "pos": ["Pleural effusion.", "Fluid in the pleural space.", "Blunting of the costophrenic angles."]
    },
    "Pleural Other": {
        "pos": ["Pleural thickening.", "Pleural scarring.", "Pleural abnormality."]
    },
    "Fracture": {
        "pos": ["Fracture.", "Rib fracture.", "Cortical step-off indicating bone fracture."]
    },
    "Support Devices": {
        "pos": ["Support devices are present.", "Endotracheal tube is in place.", "Medical apparatus seen."]
    }
}

# ==========================================
# 2. 初始化模型引擎
# ==========================================
print("Loading BioViL models... This might take a moment.")
text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Models loaded successfully on {device}.")


def get_ensemble_text_embedding(prompts: List[str]) -> torch.Tensor:
    """提取多个文本提示的特征并求平均，以增强 Zero-shot 稳定性"""
    embeddings = []
    with torch.no_grad():
        for prompt in prompts:
            # 获取单句话的全局特征向量
            emb = text_inference.get_embeddings_from_prompt(prompt)
            embeddings.append(emb)

    # 沿着句子维度求平均，并重新进行 L2 归一化 (Cosine similarity 前提)
    ensemble_emb = torch.stack(embeddings).mean(dim=0)
    ensemble_emb = F.normalize(ensemble_emb, p=2, dim=-1)
    return ensemble_emb


def classify_cxr_image(image_path: Path, similarity_threshold: float = 0.3) -> Dict[str, Dict[str, float]]:
    """
    对单张 X 光片进行 13 种疾病的零样本分类
    :param image_path: 图像路径
    :param similarity_threshold: 正向相似度阈值 (默认 0.3，可根据实际情况调整)
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"\nAnalyzing Image: {image_path.name} ...")
    print(f"Using Positive Similarity Threshold: {similarity_threshold}\n")

    results = {}

    # 1. 提取图像的全局特征向量 (只需提取一次)
    with torch.no_grad():
        # 返回形状通常为 (1, Embedding_Dim)
        image_embedding = image_inference.get_projected_global_embedding(image_path)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)

    # 2. 遍历 13 种疾病进行分类判断
    for disease, prompts in tqdm(CHEXPERT_PROMPTS.items(), desc="Diagnosing diseases", unit="disease"):
        # 获取正向的融合文本特征
        pos_emb = get_ensemble_text_embedding(prompts["pos"])

        # 计算图像与正向文本的余弦相似度 [-1, 1]
        sim_pos = F.cosine_similarity(image_embedding, pos_emb, dim=-1).item()

        # 判断结果：仅比较正向相似度是否超过设定阈值
        prediction = "Positive" if sim_pos > similarity_threshold else "Negative"

        results[disease] = {
            "prediction": prediction,
            "sim_pos": sim_pos
        }

    # 3. 检查是否有阳性结果，如果没有则添加 No Finding 为阳性
    has_positive = any(data["prediction"] == "Positive" for data in results.values())
    if not has_positive:
        results["No Finding"] = {
            "prediction": "Positive",
            "sim_pos": 1.0  # 设定为最高相似度
        }

    return results


def main():
    # 替换为你实际的绝对路径
    # 例如：image_path_str = "/absolute/path/to/your/data/NLMCXR/CXR3150_IM-1482-3001.png"
    image_path_str = input("请输入 X 光片图像的绝对路径:").strip()
    image_path = Path(image_path_str)

    # 使用默认阈值 0.3
    threshold = 0.3

    try:
        classification_results = classify_cxr_image(image_path, similarity_threshold=threshold)

        # 打印格式化的分类报告
        print("\n" + "=" * 60)
        print(f"{'Disease Category':<30} | {'Prediction':<10} | {'Pos Sim':<10}")
        print("=" * 60)

        for disease, data in classification_results.items():
            pred = data["prediction"]
            sim = data["sim_pos"]
            # 突出显示阳性预测
            if pred == "Positive":
                print(f"\033[91m{disease:<30} | {pred:<10} | {sim:.4f}\033[0m")
            else:
                print(f"{disease:<30} | {pred:<10} | {sim:.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()