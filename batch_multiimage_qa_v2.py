#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成多图化学QA对的脚本 (基于映射文件版本)
文+图 -> 文+图 (多图输入，多图输出)
"""

import json
import os
import time
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv('config.env')

# 从环境变量获取API密钥
API_KEY = os.getenv('OPENAI_API_KEY')

class MultiImageQAGeneratorV2:
    def __init__(self, mapping_file="images_301_900.jsonl"):
        self.mapping_file = mapping_file
        self.github_base_url = "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-chemistry/main/original_data"
        self.client = OpenAI(api_key=API_KEY)
        
    def load_image_mapping(self):
        """加载图片映射文件，按ID分组"""
        try:
            image_groups = defaultdict(list)
            
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    query_id = data['id']
                    image_path = data['image_path']
                    # 提取文件名 (去掉images/前缀)
                    image_filename = os.path.basename(image_path)
                    image_groups[query_id].append(image_filename)
            
            # 转换为普通字典并排序图片
            result = {}
            for query_id, images in image_groups.items():
                result[query_id] = sorted(images)
            
            return result
            
        except FileNotFoundError:
            print(f"❌ 找不到映射文件: {self.mapping_file}")
            return {}
        except Exception as e:
            print(f"❌ 加载映射文件失败: {e}")
            return {}
    
    def generate_github_url(self, image_filename: str) -> str:
        """根据图片文件名生成GitHub URL"""
        return f"{self.github_base_url}/images/{image_filename}"
    
    def split_images_for_qa(self, images: list) -> tuple:
        """
        将图片列表分成两部分用于输入和输出
        确保两部分都至少有一张图片，且保持原始顺序
        """
        total_images = len(images)
        if total_images < 2:
            raise ValueError(f"至少需要2张图片才能分割，当前只有{total_images}张")
        
        # 随机选择分割点（确保两部分都有至少1张图片）
        split_point = random.randint(1, total_images - 1)
        
        input_images = images[:split_point]
        output_images = images[split_point:]
        
        return input_images, output_images
    
    def create_prompt(self, input_images: list, output_images: list, data_id: str) -> str:
        """创建用于生成多图QA对的prompt"""
        
        # 构建原始数据部分
        original_data = "{\n"
        for i, img in enumerate(input_images + output_images, 1):
            url = self.generate_github_url(img)
            original_data += f'    "image{i}": "{url}",\n'
        original_data = original_data.rstrip(',\n') + "\n}"
        
        prompt = f"""You are a multimodal expert. Based on the following original data, please construct a data (Question-Answer pair) entry that strictly conforms to the JSON format below.
Please design a multimodal interleaved Question-Answer pair. You can place different pieces of information from the original data into the input or output of the Question-Answer pair.

[Original data]
{original_data}

[Question-Answer pair JSON template]
This Question-Answer pair must adhere to the following structure in the following JSON template and don't generate additional information.
{{
    "domain": "natural_science",
    "subdomain": "chemistry",
    "id": "{data_id}",
    "input": {{
        "modal": {{
            "image1": "url",
            ...
        }},
        "content": "Interleave <image1>, <image2>, etc. tags at the appropriate positions in the text and CLEARLY indicate that the answer must include the number of images in the output to support or illustrate the explanation. For example, the answer must include n images in the output to support or illustrate the explanation, where n is the number of images in the output."
    }},
    "output": {{
        "modal": {{
            "image1": "url",
            ...
        }},
        "content": "This is the golden annotation answer that the model is expected to generate. Interleave <image1>, <image2>, etc. tags at suitable positions within the text."
    }}
}}

[Construction requirements]
1 You need to design appropriate question-answer pair and clearly indicate in the question which specific modalities other than text are required to be included in the answer.
2 The content of the input is the entire input fed into the model. The question-answer pair should be open-world QA.
3 The content of the input is the entire input fed into the model and the content of the output is the golden output of the model. You should design the input content and output content based on the original data.
4 Give the JSON directly, no additional output information.
5 The <imageN> tags should be the components of the text sentence, not just a single word. For example, the <imageN> tags can serve as the subject, object, or other components of the sentence. Use specific numbered tags like <image1>, <image2>, etc.
6 Please note that the <imageN> tags of the input should not appear in the output.
7 IMPORTANT: Input content must contain all tags of input images and NOT contain any tags that refer to output images, and output content must contain all tags of output images and NOT contain any tags that refer to input images. Each part can only reference its own images.
8 You need to divide the images in the original data into two parts, making sure not to change their original order. Both parts must contain at least one image. Place the first part in the input and the second part in the output.
9 CRITICAL: The question-answer pair MUST be chemically and scientifically relevant. The input question should logically connect to the output answer through chemical concepts, molecular structures, reactions, or properties shown in the images. Avoid generic or unrelated questions.

Input images (first {len(input_images)} images): {[self.generate_github_url(img) for img in input_images]}
Output images (remaining {len(output_images)} images): {[self.generate_github_url(img) for img in output_images]}

[Tag Usage Rules]
- Input content can ONLY use tags for input images: {', '.join([f'<image{i+1}>' for i in range(len(input_images))])}
- Output content can ONLY use tags for output images: {', '.join([f'<image{i+len(input_images)+1}>' for i in range(len(output_images))])}
- Cross-referencing between input and output images is strictly forbidden
"""
        return prompt
    
    def call_openai_api(self, prompt: str, input_images: list) -> str:
        """调用OpenAI API生成QA对"""
        try:
            # 构建消息内容，包含文本和多张图片
            content = [{"type": "text", "text": prompt}]
            
            # 添加输入图片到API调用中
            for img in input_images:
                img_url = self.generate_github_url(img)
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": img_url}
                })
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a multimodal expert specialized in chemistry education. Generate structured JSON data for chemistry-related multimodal question-answer pairs with multiple images. Always respond with properly formatted JSON."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ API调用失败: {str(e)}")
            return None
    
    def generate_single_qa(self, query_id: str, images: list) -> dict:
        """生成单个QA对"""
        try:
            # 分割图片
            input_images, output_images = self.split_images_for_qa(images)
            
            print(f"   📊 查询 {query_id}: {len(input_images)}张输入图片, {len(output_images)}张输出图片")
            
            # 创建prompt
            prompt = self.create_prompt(input_images, output_images, query_id)
            
            # 调用API
            response = self.call_openai_api(prompt, input_images)
            
            if response:
                try:
                    qa_data = json.loads(response)
                    
                    # 验证生成的数据结构
                    if self.validate_qa_structure(qa_data, input_images, output_images):
                        # 立即保存到文件
                        self.save_single_qa(qa_data)
                        print(f"   ✅ 成功生成QA对，已保存")
                        return qa_data
                    else:
                        print(f"   ❌ 查询 {query_id}: 生成的数据结构不符合要求")
                        return None
                except json.JSONDecodeError as e:
                    print(f"   ❌ 查询 {query_id}: JSON解析失败: {e}")
                    return None
            else:
                print(f"   ❌ 查询 {query_id}: API调用失败")
                return None
                
        except Exception as e:
            print(f"   ❌ 查询 {query_id}: 处理失败: {e}")
            return None
    
    def validate_qa_structure(self, qa_data: dict, input_images: list, output_images: list) -> bool:
        """验证生成的QA数据结构"""
        try:
            # 检查必需的字段
            required_fields = ['domain', 'subdomain', 'id', 'input', 'output']
            for field in required_fields:
                if field not in qa_data:
                    return False
            
            # 检查input结构
            if 'modal' not in qa_data['input'] or 'content' not in qa_data['input']:
                return False
            
            # 检查output结构
            if 'modal' not in qa_data['output'] or 'content' not in qa_data['output']:
                return False
            
            # 检查图片数量是否匹配
            input_modal_count = len(qa_data['input']['modal'])
            output_modal_count = len(qa_data['output']['modal'])
            
            if input_modal_count != len(input_images) or output_modal_count != len(output_images):
                print(f"   ⚠️  图片数量不匹配: 输入期望{len(input_images)}实际{input_modal_count}, 输出期望{len(output_images)}实际{output_modal_count}")
                return False
            
            return True
            
        except Exception:
            return False
    
    def generate_qa_by_ids(self, query_ids: list) -> list:
        """根据指定的查询ID列表生成QA对"""
        print(f"🚀 开始生成 {len(query_ids)} 个多图化学QA对...")
        
        # 加载图片映射
        image_mapping = self.load_image_mapping()
        if not image_mapping:
            print("❌ 无法加载图片映射")
            return []
        
        print(f"✅ 加载了 {len(image_mapping)} 个查询的图片映射")
        
        qa_pairs = []
        successful_count = 0
        failed_count = 0
        
        for i, query_id in enumerate(query_ids):
            if query_id not in image_mapping:
                print(f"\n📋 处理 {i+1}/{len(query_ids)}: 查询 {query_id}")
                print(f"   ❌ 查询 {query_id}: 在映射文件中未找到")
                failed_count += 1
                continue
            
            images = image_mapping[query_id]
            if len(images) < 2:
                print(f"\n📋 处理 {i+1}/{len(query_ids)}: 查询 {query_id}")
                print(f"   ❌ 查询 {query_id}: 图片数量不足({len(images)}张)，至少需要2张")
                failed_count += 1
                continue
            
            print(f"\n📋 处理 {i+1}/{len(query_ids)}: 查询 {query_id} (共{len(images)}张图片)")
            
            # 生成QA对
            qa_data = self.generate_single_qa(query_id, images)
            
            if qa_data:
                qa_pairs.append(qa_data)
                successful_count += 1
                print(f"   ✅ 成功生成QA对")
            else:
                failed_count += 1
                print(f"   ❌ 生成失败")
            
            # 添加延迟避免API限制
            if i < len(query_ids) - 1:
                delay_seconds = 8  # 可以调整这个值
                print(f"   ⏳ 等待{delay_seconds}秒后继续...")
                time.sleep(delay_seconds)
        
        print(f"\n📊 生成完成统计:")
        print(f"   ✅ 成功: {successful_count}")
        print(f"   ❌ 失败: {failed_count}")
        print(f"   📈 成功率: {successful_count/len(query_ids)*100:.1f}%")
        
        return qa_pairs
    
    def save_single_qa(self, qa_data: dict, output_file: str = "chemistry_qa_pairs.jsonl"):
        """立即保存单个QA对到JSONL文件（追加模式，格式化JSON）"""
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                # 写入格式化的JSON，每个case占多行，便于阅读
                formatted_json = json.dumps(qa_data, ensure_ascii=False, indent=2)
                f.write(formatted_json + '\n')
            
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    
    def save_qa_pairs(self, qa_pairs: list, output_file: str = "chemistry_qa_pairs.jsonl"):
        """保存QA对到JSONL文件（追加模式，格式化JSON）"""
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    # 写入格式化的JSON，每个case占多行，便于阅读
                    formatted_json = json.dumps(qa_pair, ensure_ascii=False, indent=2)
                    f.write(formatted_json + '\n')
            
            print(f"💾 QA对已追加保存到: {output_file}")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

def main():
    """主函数"""
    print("🧪 多图化学QA对生成器 v2 (基于映射文件)")
    print("=" * 60)
    
    # 检查API密钥
    if not API_KEY:
        print("❌ 未找到OpenAI API密钥，请检查config.env文件")
        return
    
    # 创建生成器
    generator = MultiImageQAGeneratorV2()
    
    # 获取用户输入
    print("请输入要生成QA对的查询ID:")
    print("格式: 0301,0302,0303 或 0301-0305 或 0301")
    id_input = input("查询ID: ").strip()
    
    if not id_input:
        print("❌ 未输入查询ID")
        return
    
    # 解析查询ID
    query_ids = []
    try:
        if '-' in id_input:
            # 范围格式: 0301-0305
            start_id, end_id = id_input.split('-')
            start_num = int(start_id)
            end_num = int(end_id)
            query_ids = [f"{i:04d}" for i in range(start_num, end_num + 1)]
        elif ',' in id_input:
            # 列表格式: 0301,0302,0303
            query_ids = [id.strip() for id in id_input.split(',')]
        else:
            # 单个ID: 0301
            query_ids = [id_input]
    except ValueError:
        print("❌ 查询ID格式错误")
        return
    
    print(f"\n🎯 配置信息:")
    print(f"   - 查询ID数量: {len(query_ids)}")
    print(f"   - 查询ID列表: {query_ids[:10]}{'...' if len(query_ids) > 10 else ''}")
    print(f"   - 映射文件: {generator.mapping_file}")
    print(f"   - GitHub基础URL: {generator.github_base_url}")
    
    confirm = input(f"\n确认开始生成? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消生成")
        return
    
    # 生成QA对
    start_time = time.time()
    qa_pairs = generator.generate_qa_by_ids(query_ids)
    end_time = time.time()
    
    if qa_pairs:
        # 显示统计信息
        print(f"\n✨ 生成完成!")
        print(f"   ⏱️  总耗时: {end_time - start_time:.1f} 秒")
        print(f"   📊 生成数量: {len(qa_pairs)}")
        print(f"   💾 保存文件: chemistry_qa_pairs.jsonl")
        print(f"   📝 每个QA对生成后已立即保存")
    else:
        print("\n❌ 没有成功生成任何QA对")

if __name__ == "__main__":
    main()
