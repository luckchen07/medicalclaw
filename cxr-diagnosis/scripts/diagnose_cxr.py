#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
胸部X光诊断脚本
自动查找最新的CXR图片并运行BioViL诊断
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import subprocess

# 设置UTF-8编码输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def find_latest_cxr_image(cxr_dir: str) -> str:
    """
    查找指定目录中最新的CXR图片
    
    Args:
        cxr_dir: CXR图片目录路径
        
    Returns:
        最新图片的绝对路径
        
    Raises:
        ValueError: 如果目录不存在或没有找到图片
    """
    if not os.path.exists(cxr_dir):
        raise ValueError(f"目录不存在: {cxr_dir}")
    
    # 支持的图片格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.dcm']
    
    # 查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(cxr_dir, ext)))
        image_files.extend(glob.glob(os.path.join(cxr_dir, ext.upper())))
    
    if not image_files:
        raise ValueError(f"在目录 {cxr_dir} 中未找到任何图片文件")
    
    # 按修改时间排序，返回最新的
    latest_image = max(image_files, key=os.path.getmtime)
    return os.path.abspath(latest_image)


def run_biovil_diagnosis(image_path: str, biovil_script: str, conda_env: str = "torch-gpu") -> str:
    """
    运行BioViL诊断脚本
    
    Args:
        image_path: 图片路径
        biovil_script: BioViL.py脚本路径
        conda_env: Conda环境名称
        
    Returns:
        诊断输出结果
    """
    import shutil
    
    # 查找conda环境的Python路径
    # Windows上conda环境路径格式: C:\Users\...\anaconda3\envs\torch-gpu\python.exe
    conda_base = os.environ.get('CONDA_EXE', '')
    if conda_base:
        conda_root = os.path.dirname(os.path.dirname(conda_base))
        python_exe = os.path.join(conda_root, 'envs', conda_env, 'python.exe')
    else:
        # 尝试常见的conda安装路径
        possible_conda_roots = [
            os.path.expanduser('~/anaconda3'),
            os.path.expanduser('~/miniconda3'),
            r'C:\ProgramData\anaconda3',
            r'C:\ProgramData\miniconda3',
        ]
        python_exe = None
        for root in possible_conda_roots:
            candidate = os.path.join(root, 'envs', conda_env, 'python.exe')
            if os.path.exists(candidate):
                python_exe = candidate
                break
        
        if not python_exe:
            raise RuntimeError(f"未找到conda环境: {conda_env}")
    
    if not os.path.exists(python_exe):
        raise RuntimeError(f"Python不存在: {python_exe}")
    
    try:
        # 直接使用conda环境的Python运行脚本
        process = subprocess.Popen(
            [python_exe, biovil_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 输入图片路径
        stdout, stderr = process.communicate(input=f"{image_path}\n", timeout=120)
        
        if process.returncode != 0:
            error_msg = stderr if stderr else stdout
            raise RuntimeError(f"诊断脚本执行失败:\n{error_msg}")
        
        return stdout
        
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("诊断超时（超过120秒）")
    except Exception as e:
        raise RuntimeError(f"执行诊断时出错: {str(e)}")


def clean_ansi_codes(text: str) -> str:
    """移除ANSI转义码"""
    import re
    # 匹配ANSI转义序列
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9]+)m')
    return ansi_escape.sub('', text)


def parse_diagnosis_output(output: str) -> dict:
    """
    解析诊断输出，提取疾病预测结果
    
    Args:
        output: 诊断脚本的输出文本
        
    Returns:
        包含诊断结果的字典 {'positive': [...], 'negative': [...]}
    """
    # 先清理ANSI转义码
    output = clean_ansi_codes(output)
    
    lines = output.strip().split('\n')
    results = {
        'positive': [],
        'negative': [],
        'all': []
    }
    
    parsing = False
    for line in lines:
        # 检测表格开始
        if 'Disease Category' in line and 'Prediction' in line:
            parsing = True
            continue
        
        # 跳过分隔线
        if parsing and '====' in line:
            continue
            
        # 解析数据行
        if parsing and '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 3:
                disease = parts[0].strip()
                prediction = parts[1].strip()
                
                # 提取数值
                try:
                    # 移除可能的颜色标记和其他字符
                    pos_sim_str = parts[2].strip()
                    # 提取数字部分（包括负号和小数点）
                    import re
                    match = re.search(r'-?\d+\.?\d*', pos_sim_str)
                    if match:
                        pos_sim = float(match.group())
                    else:
                        pos_sim = 0.0
                except (ValueError, IndexError):
                    pos_sim = 0.0
                
                result_entry = {
                    'disease': disease,
                    'prediction': prediction,
                    'pos_sim': pos_sim
                }
                
                results['all'].append(result_entry)
                
                if prediction == 'Positive':
                    results['positive'].append(disease)
                else:
                    results['negative'].append(disease)
    
    return results


def format_results(results: dict) -> str:
    """
    格式化诊断结果为用户友好的文本
    
    Args:
        results: 解析后的诊断结果
        
    Returns:
        格式化的结果文本
    """
    if not results['all']:
        return "未能解析诊断结果"
    
    output_lines = []
    
    # 如果有阳性结果
    if results['positive']:
        output_lines.append("🔍 **诊断发现异常：**")
        output_lines.append("")
        for disease in results['positive']:
            # 找到对应的完整信息
            for item in results['all']:
                if item['disease'] == disease:
                    output_lines.append(f"• **{disease}**: Positive (相似度: {item['pos_sim']:.4f})")
                    break
    else:
        output_lines.append("✅ **未发现明显异常**")
    
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("**完整诊断结果：**")
    output_lines.append("")
    
    # 添加完整结果表格
    output_lines.append("| 疾病类别 | 预测结果 | 相似度 |")
    output_lines.append("|---------|---------|--------|")
    for item in results['all']:
        marker = "⚠️" if item['prediction'] == 'Positive' else "✓"
        output_lines.append(f"| {marker} {item['disease']} | {item['prediction']} | {item['pos_sim']:.4f} |")
    
    return '\n'.join(output_lines)


def main(cxr_dir: str = r"D:\CXR", 
         biovil_script: str = r"D:\python_workspace\MedicalClaw\BioViL.py",
         conda_env: str = "torch-gpu"):
    """
    主诊断流程
    
    Args:
        cxr_dir: CXR图片目录
        biovil_script: BioViL脚本路径
        conda_env: Conda环境名称
    """
    try:
        # 1. 查找最新图片
        print(f"📁 正在查找 {cxr_dir} 中的最新CXR图片...")
        image_path = find_latest_cxr_image(cxr_dir)
        print(f"✓ 找到图片: {image_path}")
        print()
        
        # 2. 运行诊断
        print(f"🔬 正在运行诊断分析...")
        print(f"  环境: {conda_env}")
        print(f"  脚本: {biovil_script}")
        print()
        
        output = run_biovil_diagnosis(image_path, biovil_script, conda_env)
        
        # 3. 解析结果
        results = parse_diagnosis_output(output)
        
        # 4. 格式化输出
        formatted = format_results(results)
        print(formatted)
        
        return results
        
    except ValueError as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ 运行时错误: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未预期的错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='胸部X光诊断工具')
    parser.add_argument('--cxr-dir', default=r'D:\CXR', help='CXR图片目录')
    parser.add_argument('--biovil-script', default=r'D:\python_workspace\MedicalClaw\BioViL.py', help='BioViL脚本路径')
    parser.add_argument('--conda-env', default='torch-gpu', help='Conda环境名称')
    
    args = parser.parse_args()
    
    main(args.cxr_dir, args.biovil_script, args.conda_env)
