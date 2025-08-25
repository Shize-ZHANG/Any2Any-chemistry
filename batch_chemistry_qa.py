#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成化学QA对的脚本
模仿batch_image.py的方式，使用GPT-4o生成文+图 -> 文+音的QA对
输入：1张化学图片，输出：文本 + 音频描述
"""

import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv('config.env')

# 从环境变量获取API密钥
API_KEY = os.getenv('OPENAI_API_KEY')

def load_original_data():
    """加载原始JSONL数据"""
    jsonl_file = "test_samples_300.jsonl"
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        return data
    except FileNotFoundError:
        print(f"❌ 找不到文件: {jsonl_file}")
        return []
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return []

def generate_github_url(image_filename: str) -> str:
    """
    根据图片文件名生成GitHub URL
    
    Args:
        image_filename: 图片文件名，如 "img_0001_01.png"
        
    Returns:
        完整的GitHub URL
    """
    base_url = "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-chemistry/main/original_data/images"
    return f"{base_url}/{image_filename}"

def call_openai_api(prompt: str, image_url: str) -> str:
    """
    调用OpenAI API，支持多模态输入
    
    Args:
        prompt: 发送的文本prompt
        image_url: 图片URL
        
    Returns:
        API返回的响应
    """
    client = OpenAI(api_key=API_KEY)
    
    try:
        # 构建消息内容，包含文本和图片
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o 模型
            messages=[
                {
                    "role": "system", 
                    "content": "You are a multimodal expert specialized in chemistry education. Generate structured JSON data for chemistry-related multimodal question-answer pairs. Always respond with properly formatted, multi-line JSON that is easy to read."
                },
                {
                    "role": "user", 
                    "content": content
                }
            ],
            max_tokens=2000,
            response_format={"type": "json_object"}  # 强制输出JSON格式
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")
        return None

def create_prompt(image_url: str, text_description: str, data_id: str) -> str:
    """
    创建prompt，用于生成文+图到文+音的化学QA对
    
    Args:
        image_url: 图片URL
        text_description: 文本描述（作为音频内容）
        data_id: 数据ID
        
    Returns:
        完整的prompt
    """
    original_data = {
        "image1": image_url,
        "audio1": text_description
    }
    
    prompt = f"""You are a multimodal expert. Based on the following original data, please construct a data (Question-Answer pair) entry that strictly conforms to the JSON format below.
Please design a multimodal interleaved Question-Answer pair. You can place different pieces of information from the original data into the input or output of the Question-Answer pair.

[Original data]
{json.dumps(original_data, indent=2, ensure_ascii=False)}

[Question-Answer pair JSON template]
This Question-Answer pair must adhere to the following structure in the following JSON template and don't generate additional information.
{{
    "domain": "natural_science",
    "subdomain": "chemistry",
    "id": "{data_id}",
    "input": {{
        "modal": {{
            "image1": "url"
        }},
        "content": "Interleave <image1> tag at the appropriate position in the text and clearly indicate that the answer must include audio content to support or illustrate the explanation."
    }},
    "output": {{
        "modal": {{
            "audio1": "text"
        }},
        "content": "This is the golden annotation answer that the model is expected to generate. Interleave <audio1> tag at suitable position within the text."
    }}
}}

[Construction requirements]
1 You need to design appropriate question-answer pair and clearly indicate in the question which specific modalities other than text are required to be included in the answer. 
2 The content of the input is the entire input fed into the model. The question-answer pair should be open-world QA. 
3 The content of the input is the entire input fed into the model and the content of the output is the golden output of the model. You should design the input content and output content based on the original data.
4 Give the JSON directly, no additional output information.
5 The <> tags should be the components of the text sentence, not just a single word. For example, the <> tags can serve as the subject, object, or other components of the sentence.
6 Please note that the <> tags of the input should not appear in the output.
7 The <audio1> is a textual description of the <image1> for chemistry education purposes.

[IMPORTANT REQUIREMENTS]
- The input should contain exactly one image (<image1>) showing a chemistry-related diagram, structure, or concept
- The output should contain exactly one audio description (<audio1>) explaining the chemistry content
- The audio1 text should be the provided text description, which explains the chemistry concepts shown in the image
- The question should ask for a detailed audio explanation of the chemistry concepts, structures, or processes shown in the image
- The answer should naturally reference the audio description using <audio1> tag
- Focus on educational value and scientific accuracy in chemistry
- The question should be suitable for chemistry students or educators
- The audio description should help someone understand the chemical concepts without seeing the image

[Example of good structure]
Input: Question about chemistry concepts with <image1> asking for audio explanation
Output: Answer providing <audio1> with detailed chemistry explanation

Please respond with only the JSON, no additional text or explanation.
"""

    return prompt

def process_single_item(data_item: dict, data_id: str, delay: int = 2) -> bool:
    """
    处理单个数据项的QA对生成
    
    Args:
        data_item: 包含image_path和text的数据项
        data_id: 数据ID
        delay: API调用间隔延迟(秒)
        
    Returns:
        是否成功
    """
    print(f"\n🔄 处理ID: {data_id}")
    
    # 提取图片文件名和生成URL
    image_path = data_item['image_path']
    image_filename = os.path.basename(image_path)
    image_url = generate_github_url(image_filename)
    text_description = data_item['text']
    
    print(f"   📷 图片URL: {image_url}")
    print(f"   📝 文本长度: {len(text_description)} 字符")
    
    # 创建prompt
    prompt = create_prompt(image_url, text_description, data_id)
    
    # 调用API
    print(f"   📤 调用OpenAI API...")
    response = call_openai_api(prompt, image_url)
    
    if response:
        try:
            # 清理响应
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            qa_pair = json.loads(cleaned_response)
            
            # 保存到JSONL文件（格式化的多行JSON）
            output_file = "chemistry_qa_pairs.jsonl"
            with open(output_file, "a", encoding="utf-8") as f:
                # 写入格式化的JSON，每个case占多行，便于阅读
                formatted_json = json.dumps(qa_pair, ensure_ascii=False, indent=2)
                f.write(formatted_json + "\n")
            
            print(f"   ✅ 成功生成化学QA对，已保存")
            return True
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON解析失败: {e}")
            # 保存原始响应到错误日志
            with open("chemistry_error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"ID: {data_id}\nError: {e}\nResponse: {response}\n{'='*50}\n")
            return False
            
    else:
        print(f"   ❌ API调用失败")
        # 记录错误
        with open("chemistry_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"ID: {data_id}\nError: API调用失败\n{'='*50}\n")
        return False

def batch_process(start_index: int = 0, end_index: int = 10, delay: int = 2):
    """
    批量处理多个化学数据项
    
    Args:
        start_index: 起始索引
        end_index: 结束索引
        delay: API调用间隔延迟(秒)
    """
    print("🧪 化学QA对批量生成脚本")
    print("=" * 50)
    print(f"🎯 处理范围: 索引 {start_index} 到 {end_index}")
    print(f"⏱️  API延迟: {delay}秒")
    print(f"📥 输入模式: 1张化学图片 + 文本描述")
    print(f"📤 输出模式: 文本问题 + 音频描述")
    print("=" * 50)
    
    # 检查API密钥
    if not API_KEY:
        print("❌ 请在config.env文件中设置OPENAI_API_KEY环境变量")
        return
    
    # 加载原始数据
    original_data = load_original_data()
    if not original_data:
        print("❌ 无法加载原始数据")
        return
    
    print(f"📊 加载了 {len(original_data)} 条原始数据")
    
    # 调整结束索引
    end_index = min(end_index, len(original_data) - 1)
    
    success_count = 0
    error_count = 0
    
    # 批量处理
    for i in range(start_index, end_index + 1):
        data_id = str(i + 1)  # ID从1开始
        data_item = original_data[i]
        
        success = process_single_item(data_item, data_id, delay)
        if success:
            success_count += 1
        else:
            error_count += 1
        
        # 添加延迟以避免API限制（除了最后一个）
        if i < end_index:
            print(f"   ⏳ 等待{delay}秒后继续...")
            time.sleep(delay)
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"📊 处理完成!")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {error_count} 个")
    print(f"📁 输出文件: chemistry_qa_pairs.jsonl")
    if error_count > 0:
        print(f"📝 错误日志: chemistry_error_log.txt")
    print(f"{'='*50}")

def generate_demo_data():
    """生成演示数据（前5个样本，不调用API）"""
    print("🧪 生成演示化学QA对数据")
    print("=" * 50)
    
    # 加载原始数据
    original_data = load_original_data()
    if not original_data:
        print("❌ 无法加载原始数据")
        return
    
    demo_data = original_data[:5]  # 只取前5个样本
    output_file = "demo_chemistry_qa_pairs.jsonl"
    
    print(f"🎯 生成前5个样本的演示数据...")
    
    for i, data_item in enumerate(demo_data):
        # 提取数据
        image_path = data_item['image_path']
        image_filename = os.path.basename(image_path)
        image_url = generate_github_url(image_filename)
        text_description = data_item['text']
        
        # 创建演示QA对
        qa_pair = {
            "domain": "natural_science",
            "subdomain": "chemistry",
            "id": str(i + 1),
            "input": {
                "modal": {
                    "image1": image_url
                },
                "content": f"Please examine the chemical diagram shown in <image1> and provide a comprehensive audio explanation of the scientific concepts, molecular structures, and chemical processes illustrated in this chemistry-related image."
            },
            "output": {
                "modal": {
                    "audio1": text_description
                },
                "content": f"Based on my analysis of the chemical diagram, here is the detailed audio explanation: <audio1>"
            }
        }
        
        # 保存到文件（格式化的多行JSON）
        with open(output_file, "a", encoding="utf-8") as f:
            formatted_json = json.dumps(qa_pair, ensure_ascii=False, indent=2)
            f.write(formatted_json + "\n")
        
        print(f"   ✅ 生成样本 {i + 1}/5: {image_filename}")
    
    print(f"💾 演示数据已保存到: {output_file}")
    print(f"{'='*50}")

def validate_generated_data(filename: str = "chemistry_qa_pairs.jsonl"):
    """验证生成的数据格式"""
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return
    
    print(f"🔍 验证生成的数据: {filename}")
    print("=" * 50)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        valid_count = 0
        error_count = 0
        
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                
                # 检查必要字段
                required_fields = ['domain', 'subdomain', 'id', 'input', 'output']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"   ❌ 第{i+1}行缺少字段: {missing_fields}")
                    error_count += 1
                    continue
                
                # 检查input结构
                if 'modal' not in data['input'] or 'image1' not in data['input']['modal']:
                    print(f"   ❌ 第{i+1}行input缺少image1")
                    error_count += 1
                    continue
                
                # 检查output结构
                if 'modal' not in data['output'] or 'audio1' not in data['output']['modal']:
                    print(f"   ❌ 第{i+1}行output缺少audio1")
                    error_count += 1
                    continue
                
                # 检查标签使用
                input_content = data['input'].get('content', '')
                output_content = data['output'].get('content', '')
                
                if '<image1>' not in input_content:
                    print(f"   ❌ 第{i+1}行input content缺少<image1>标签")
                    error_count += 1
                    continue
                
                if '<audio1>' not in output_content:
                    print(f"   ❌ 第{i+1}行output content缺少<audio1>标签")
                    error_count += 1
                    continue
                
                valid_count += 1
                
            except json.JSONDecodeError:
                print(f"   ❌ 第{i+1}行JSON格式错误")
                error_count += 1
        
        print(f"✅ 有效记录: {valid_count}")
        print(f"❌ 错误记录: {error_count}")
        print(f"📊 总记录数: {len(lines)}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")

def main():
    """主函数"""
    print("🧪 化学QA对批量生成脚本")
    print("=" * 50)
    print("📥 输入模式: 1张化学图片 + 文本描述")
    print("📤 输出模式: 文本问题 + 音频描述")
    print("=" * 50)
    print("请选择模式:")
    print("1. 单次处理 (测试用)")
    print("2. 批量处理 (使用GPT-4o)")
    print("3. 生成演示数据 (无需API)")
    print("4. 验证生成的数据")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        # 单次处理模式
        print("\n🎯 单次处理模式")
        
        # 加载数据
        original_data = load_original_data()
        if not original_data:
            print("❌ 无法加载原始数据")
            return
        
        index = input(f"请输入要处理的索引 (0-{len(original_data)-1}, 默认: 0): ").strip()
        index = int(index) if index.isdigit() else 0
        
        if 0 <= index < len(original_data):
            process_single_item(original_data[index], str(index + 1))
        else:
            print("❌ 索引超出范围")
        
    elif choice == "2":
        # 批量处理模式
        print("\n📊 批量处理模式")
        start_index = input("请输入起始索引 (默认: 0): ").strip()
        end_index = input("请输入结束索引 (默认: 9): ").strip()
        delay = input("请输入API延迟秒数 (默认: 2): ").strip()
        
        # 设置默认值
        start_index = int(start_index) if start_index.isdigit() else 0
        end_index = int(end_index) if end_index.isdigit() else 9
        delay = int(delay) if delay.isdigit() else 2
        
        batch_process(start_index, end_index, delay)
        
    elif choice == "3":
        # 生成演示数据
        generate_demo_data()
        
    elif choice == "4":
        # 验证数据
        filename = input("请输入要验证的文件名 (默认: chemistry_qa_pairs.jsonl): ").strip()
        if not filename:
            filename = "chemistry_qa_pairs.jsonl"
        validate_generated_data(filename)
        
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
