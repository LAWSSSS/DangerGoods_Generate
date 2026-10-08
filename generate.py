#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云百炼版 - 废钢异物合成图像生成脚本
使用 wan2.7-image 模型，DashScope 原生格式
自动压缩图片到 2K 以内，保持原图比例
"""

import argparse
import base64
import json
import os
import random
import re
import time
import ssl
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from PIL import Image, ImageDraw
import urllib3

# ============ 强制禁用SSL验证 ============
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ============ 自定义SSL适配器 ============

class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def get_session_with_ssl_bypass():
    session = requests.Session()
    adapter = SSLAdapter()
    session.mount('https://', adapter)
    session.verify = False
    
    retry_strategy = urllib3.Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session


# ============ Prompt分析器类 ============

class PromptAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.good_prompts_file = output_dir / "good_prompts.json"
        self.bad_prompts_file = output_dir / "bad_prompts.json"
        self.prompt_stats_file = output_dir / "prompt_stats.json"
        
        self.good_prompts: List[Dict] = []
        self.bad_prompts: List[Dict] = []
        
        self._load_data()
    
    def _load_data(self):
        if self.good_prompts_file.exists():
            try:
                with open(self.good_prompts_file, 'r', encoding='utf-8') as f:
                    self.good_prompts = json.load(f)
            except:
                self.good_prompts = []
        
        if self.bad_prompts_file.exists():
            try:
                with open(self.bad_prompts_file, 'r', encoding='utf-8') as f:
                    self.bad_prompts = json.load(f)
            except:
                self.bad_prompts = []
    
    def add_prompt_record(self, prompt: str, image_id: str, object_class: str, 
                          size_judgment: str, confidence: float, 
                          quality_score: float, fusion_quality: str,
                          round_num: int = 0,
                          full_prompt: str = ""):
        record = {
            "image_id": image_id,
            "object_class": object_class,
            "size_judgment": size_judgment,
            "confidence": confidence,
            "quality_score": quality_score,
            "fusion_quality": fusion_quality,
            "round_num": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_text": prompt,
            "full_prompt": full_prompt if full_prompt else prompt,
            "prompt_length": len(prompt)
        }
        
        is_good = (
            size_judgment == "acceptable" and 
            confidence >= 0.5 and 
            quality_score >= 0.5
        )
        
        if is_good:
            self.good_prompts.append(record)
            print(f"  [Prompt分析] ✅ 好Prompt已保存 (第{len(self.good_prompts)}个)")
        else:
            self.bad_prompts.append(record)
            print(f"  [Prompt分析] ❌ 差Prompt已保存 (第{len(self.bad_prompts)}个)")
        
        self._save_prompts()
    
    def _save_prompts(self):
        with open(self.good_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.good_prompts, f, ensure_ascii=False, indent=2)
        
        with open(self.bad_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.bad_prompts, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self.good_prompts) + len(self.bad_prompts),
            "good_count": len(self.good_prompts),
            "bad_count": len(self.bad_prompts),
            "good_prompts_file": str(self.good_prompts_file),
            "bad_prompts_file": str(self.bad_prompts_file),
            "last_update": datetime.now(timezone.utc).isoformat()
        }
    
    def save_stats(self):
        stats = self.get_stats()
        with open(self.prompt_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 Prompt分析统计")
        print("="*60)
        print(f"总记录数: {stats['total_records']}")
        print(f"好Prompt: {stats['good_count']}")
        print(f"差Prompt: {stats['bad_count']}")
        
        if self.good_prompts:
            print(f"\n✅ 好的Prompt示例 (最新的3条):")
            for i, p in enumerate(self.good_prompts[-3:]):
                print(f"  {i+1}. [{p['object_class']}] {p['prompt_text'][:150]}...")
        
        if self.bad_prompts:
            print(f"\n❌ 差的Prompt示例 (最新的3条):")
            for i, p in enumerate(self.bad_prompts[-3:]):
                print(f"  {i+1}. [{p['object_class']}] {p['prompt_text'][:150]}...")
        print("="*60)


# ============ 配置类 ============

@dataclass
class AppConfig:
    api_key: str
    model: str
    size: str
    output_format: str
    watermark: bool
    max_images: int
    timeout_seconds: int
    sleep_seconds: float
    total_num: int
    seed: Optional[int]
    
    object_resize_scale: float
    object_canvas_scale: float
    object_opacity: float
    object_anchor: str
    
    backgrounds_dir: Path
    objects_dir: Path
    prompt_templates_path: Path
    negative_prompt_path: Path
    output_images_dir: Path
    output_final_images_dir: Path
    output_metadata_dir: Path
    output_failed_dir: Path
    output_bbox_dir: Path
    output_analysis_dir: Path
    
    vlm_api_url: str
    vlm_model: str
    enable_vlm_optimization: bool = True
    max_optimization_rounds: int = 3
    size_tolerance: str = "very_small"
    
    vlm_evaluator_prompt: str = field(default_factory=lambda: (
        "请仔细分析这张合成图片，判断其中添加的异物尺寸是否合适。\n"
        "核心要求：异物必须非常小，几乎难以察觉，需要仔细寻找才能发现。\n"
        "判定标准（从严判定）：\n"
        "1. 异物是否第一眼就能被注意到？如果是，则判定为'太大'\n"
        "2. 异物是否比周围大多数废钢块更大或相当？如果是，则判定为'太大'\n"
        "3. 异物是否超过画面宽度的1.5%？如果是，则判定为'太大'\n"
        "4. 异物是否几乎难以察觉，需要仔细寻找才能发现，且尺寸远小于周围废钢块？则判定为'合格'\n"
        "5. 异物是否完全看不见或几乎不存在？则判定为'太小'\n"
        "注意：宁可偏小也不要偏大，异物越小越好。\n"
        "请以JSON格式输出：{\"size_judgment\": \"too_large|acceptable|too_small\", "
        "\"reason\": \"判断理由\", \"improvement_suggestion\": \"如果太大，如何修改Prompt\", "
        "\"confidence\": 0.0-1.0, "
        "\"quality_score\": 0.0-1.0, "
        "\"fusion_quality\": \"bad|ok|good\"}"
    ))


# ============ 尺寸计算工具 ============

def calculate_optimal_size(background_path: Path, object_path: Path = None) -> Dict[str, Any]:
    bg_img = Image.open(background_path)
    bg_w, bg_h = bg_img.size
    
    if bg_w >= 3840:
        target_ratio = random.uniform(0.004, 0.010)
    elif bg_w >= 1920:
        target_ratio = random.uniform(0.006, 0.014)
    else:
        target_ratio = random.uniform(0.008, 0.018)
    
    target_width = int(bg_w * target_ratio)
    target_height = int(bg_h * target_ratio)
    
    min_size = max(16, int(min(bg_w, bg_h) * 0.004))
    target_width = max(target_width, min_size)
    target_height = max(target_height, min_size)
    
    max_size = int(min(bg_w, bg_h) * 0.025)
    target_width = min(target_width, max_size)
    target_height = min(target_height, max_size)
    
    return {
        "bg_size": (bg_w, bg_h),
        "target_width": target_width,
        "target_height": target_height,
        "target_ratio": target_ratio,
        "description": f"约 {target_width}x{target_height} 像素（画面宽度的{target_ratio*100:.1f}%）"
    }


# ============ Prompt增强工具 ============

class PromptEnhancer:
    @staticmethod
    def get_position_descriptions(gaps: str) -> List[str]:
        base_positions = [
            "废钢堆叠的缝隙深处",
            "两块钢板之间的夹缝中",
            "齿轮与型钢的交接处",
            "废钢堆的底部空隙",
            "大型废钢件的下方阴影处",
            "杂乱废钢的中间层",
            "废钢堆的角落凹陷处",
            "管状废钢与板材的夹角中",
        ]
        if gaps and len(gaps) > 5:
            custom_positions = [
                f"{gaps}的深处",
                f"{gaps}的隐蔽角落",
                f"{gaps}的夹层中",
                f"{gaps}的下方阴影处",
            ]
            return custom_positions + base_positions
        return base_positions
    
    @staticmethod
    def get_occlusion_descriptions() -> List[str]:
        return [
            "被废钢层层覆盖，仅露出微小边缘",
            "被上方的废钢件压住，大部分被遮挡",
            "嵌在废钢缝隙中，周围被废钢包围",
            "被周围的废钢挤压，只露出极小部分",
            "藏在废钢堆深处，被多层废钢遮挡",
            "被一块较大的废钢板遮挡了大半",
            "塞在废钢件的夹缝里，只露出一角",
            "被废钢堆掩埋，仅表面微露",
        ]
    
    @staticmethod
    def get_weathering_descriptions(object_class: str) -> List[str]:
        base = [
            "表面覆盖着厚厚的铁锈和灰尘",
            "带有明显的磨损痕迹和划痕",
            "布满油污和泥垢，显得陈旧",
            "锈迹斑斑，颜色暗淡",
            "表面有凹坑和腐蚀痕迹",
            "沾染了废钢的锈色和污渍",
            "部分油漆剥落，露出锈蚀的金属",
            "表面粗糙，有明显的老化迹象",
        ]
        specific = {
            "hydraulic_cylinder": [
                "缸体表面有油渍和磨损痕迹，活塞杆有轻微锈蚀",
                "表面有磕碰凹坑，油泥包裹",
                "锈迹与油污混合，呈现工业废件的典型质感",
            ],
            "lpg_tank": [
                "罐体表面有大面积锈蚀，阀门处有陈旧破损",
                "底部有凹陷，表面覆盖着泥土和锈迹",
                "油漆大部分脱落，露出锈蚀的金属表面",
            ],
            "axle": [
                "轴表面有磨损痕迹和锈斑，端部有磕碰",
                "布满油污和铁锈，显得非常陈旧",
                "表面有严重的腐蚀和磨损痕迹",
            ],
            "wire_rope_drum": [
                "卷筒表面有锈迹，缠绕着断裂的钢丝",
                "法兰盘锈蚀严重，表面有油污",
                "钢丝绳残留物和锈迹混合，呈现废置状态",
            ],
            "shell": [
                "弹体表面有铁锈和泥土，尾部有磨损",
                "锈蚀严重，表面有划痕和凹坑",
                "沾满泥土和铁锈，颜色暗淡",
            ],
            "fire_extinguisher": [
                "表面有划痕和掉漆，锈迹斑斑",
                "标签模糊不清，瓶体有磕碰痕迹",
                "表面布满灰尘和污渍，显得十分陈旧",
            ],
        }
        if object_class in specific:
            return specific[object_class] + base
        return base
    
    @staticmethod
    def get_lighting_descriptions(lighting: str) -> List[str]:
        if lighting and len(lighting) > 3:
            return [
                f"光照{lighting}，与周围环境一致",
                f"受{lighting}影响，阴影自然",
                f"光影与{lighting}匹配",
            ]
        return [
            "光照自然，与周围废钢的光影一致",
            "光线从上方斜射，阴影落在废钢堆上",
            "漫反射光线，无明显定向光源",
            "顶部照明，废钢堆形成丰富阴影",
        ]
    
    @staticmethod
    def get_tone_descriptions(tone: str) -> List[str]:
        if tone and len(tone) > 3:
            return [
                f"色调{tone}，与废钢环境融合",
                f"色彩{tone}，呈现工业废料的质感",
                f"整体偏向{tone}，和谐统一",
            ]
        return [
            "色调偏暗，呈现工业废料的沉稳质感",
            "以铁锈红、深灰、暗褐色为主",
            "色彩饱和度低，呈现陈旧感",
        ]
    
    @staticmethod
    def get_scenario_descriptions() -> List[str]:
        return [
            "废钢堆积的工业场景",
            "废料堆场的典型环境",
            "工业废钢回收场的一角",
            "重型废钢的杂乱堆叠区域",
        ]
    
    @staticmethod
    def get_embedding_verbs() -> List[str]:
        return [
            "嵌入", "塞入", "放置", "卡在", "夹在", 
            "隐藏在", "埋在", "藏在", "陷在", "挤在"
        ]


# ============ 物体图片加载器 ============

def get_all_object_images_with_classes(object_root: Path, prompt_templates: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    object_items = []
    
    if not object_root.exists():
        return object_items
    
    all_images = []
    for img in object_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            all_images.append(img)
    
    print(f"[DEBUG] 共找到 {len(all_images)} 张物体图片")
    
    for img_path in all_images:
        filename = img_path.stem.lower()
        
        matched_class = None
        for cls in prompt_templates.keys():
            if cls in filename:
                matched_class = cls
                break
        
        if not matched_class:
            for cls in prompt_templates.keys():
                cls_variants = [
                    cls,
                    cls.replace("_", ""),
                    cls.replace("_", "-"),
                ]
                for variant in cls_variants:
                    if variant in filename:
                        matched_class = cls
                        break
                if matched_class:
                    break
        
        if not matched_class:
            print(f"[WARNING] 无法识别类别: {img_path.name}")
            continue
        
        # ===== 提高"罐/瓶/缸"类别的权重 =====
        # 键名必须与 prompt_templates.json 中的类别名称完全一致（中文）
        boosted_classes = {
            '密封胶罐': 4.0,       # 密封胶罐：1张，低权重 → 最高优先级
            '储罐': 3.0,           # 储罐：3张
            '杀虫剂罐': 3.0,       # 杀虫剂罐：3张
            '气瓶': 3.0,           # 气瓶：3张
            '气缸': 3.0,           # 气缸：3张
            '油缸': 3.0,           # 油缸：2张
            '灭火器': 3.0,         # 灭火器：3张
            '自喷漆罐': 4.0,       # 自喷漆罐：1张（最高优先级）
        }

        if matched_class in boosted_classes:
            weight = boosted_classes[matched_class]
            print(f"[DEBUG] ⭐ {img_path.name} 类别 '{matched_class}' 权重提高到 {weight}")
        elif filename.startswith('a'):
            weight = 0.1
            print(f"[DEBUG] {img_path.name} 以 'a' 开头，权重降低到 0.1")
        else:
            weight = 1.0
        
        object_items.append({
            "path": img_path,
            "class": matched_class,
            "weight": weight,
            "filename": img_path.name
        })
    
    print(f"[DEBUG] 有效物体图片: {len(object_items)} 张")
    
    class_counts = {}
    for item in object_items:
        cls = item["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"[DEBUG] 类别分布: {class_counts}")
    
    weight_counts = {"低权重(以a开头)": 0, "正常权重": 0}
    for item in object_items:
        if item["weight"] < 1.0:
            weight_counts["低权重(以a开头)"] += 1
        else:
            weight_counts["正常权重"] += 1
    print(f"[DEBUG] 权重分布: {weight_counts}")
    
    return object_items


def choose_weighted_object(object_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_items:
        return None
    
    weights = [item["weight"] for item in object_items]
    total_weight = sum(weights)
    
    if total_weight == len(object_items):
        return random.choice(object_items)
    
    r = random.random() * total_weight
    cumsum = 0
    for item, weight in zip(object_items, weights):
        cumsum += weight
        if r <= cumsum:
            return item
    
    return object_items[-1]


# ============ VLM分析器类 ============

class VLMAnalyzer:
    def __init__(self, cfg: AppConfig, prompt_analyzer: Optional[PromptAnalyzer] = None):
        self.cfg = cfg
        self.optimization_history: List[Dict] = []
        self.prompt_enhancer = PromptEnhancer()
        self.prompt_analyzer = prompt_analyzer
        
    def _call_vlm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        images: List[Path],
        max_tokens: int = 8192,
        expect_json: bool = True
    ) -> Any:
        api_key = self.cfg.api_key
        print(f"[DEBUG] VLM 使用 API Key: {api_key[:30]}...")
        print(f"[DEBUG] VLM 请求 URL: {self.cfg.vlm_api_url}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        content = []
        content.append({"type": "text", "text": full_prompt})
        for img_path in images:
            data_url = image_to_data_url_vlm(img_path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        
        payload = {
            "model": self.cfg.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        
        session = get_session_with_ssl_bypass()
        
        try:
            response = session.post(
                self.cfg.vlm_api_url, 
                headers=headers, 
                json=payload, 
                timeout=self.cfg.timeout_seconds
            )
            response.raise_for_status()
            result = response.json()
            
            try:
                content_text = result["choices"][0]["message"]["content"]
                if expect_json:
                    return self._extract_json(content_text)
                return content_text
            except (KeyError, IndexError) as e:
                raise ValueError(f"VLM返回格式异常: {e}, 原始响应: {result}")
        except requests.exceptions.SSLError as e:
            print(f"  [SSL Error] {e}，尝试使用更宽松的SSL配置...")
            session = requests.Session()
            session.verify = False
            try:
                response = session.post(
                    self.cfg.vlm_api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.cfg.timeout_seconds
                )
                response.raise_for_status()
                result = response.json()
                content_text = result["choices"][0]["message"]["content"]
                if expect_json:
                    return self._extract_json(content_text)
                return content_text
            except Exception as e2:
                print(f"  [重试失败] {e2}")
                return self._get_default_evaluation()
        except Exception as e:
            print(f"  [VLM API Error] {e}")
            return self._get_default_evaluation()
    
    def _call_vlm_with_data_url(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int = 4096,
        expect_json: bool = True
    ) -> Any:
        """
        使用已编码的 data URL 调用 VLM（用于评估压缩后的图片）
        """
        api_key = self.cfg.api_key
        print(f"[DEBUG] VLM 使用 API Key: {api_key[:30]}...")
        print(f"[DEBUG] VLM 请求 URL: {self.cfg.vlm_api_url}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        content = [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
        
        payload = {
            "model": self.cfg.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        
        session = get_session_with_ssl_bypass()
        
        try:
            response = session.post(
                self.cfg.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=self.cfg.timeout_seconds
            )
            response.raise_for_status()
            result = response.json()
            
            try:
                content_text = result["choices"][0]["message"]["content"]
                if expect_json:
                    return self._extract_json(content_text)
                return content_text
            except (KeyError, IndexError) as e:
                raise ValueError(f"VLM返回格式异常: {e}, 原始响应: {result}")
        except Exception as e:
            print(f"  [VLM API Error] {e}")
            return self._get_default_evaluation()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "object": {
                "category": "unknown",
                "color": "金属色",
                "shape": "不规则",
                "material": "金属",
                "texture": "粗糙",
                "key_features": ""
            },
            "background": {
                "lighting": "自然光",
                "shadows": "废钢阴影",
                "stacking": "杂乱堆叠",
                "gaps": "废钢缝隙",
                "tone": "暗沉",
                "texture_style": "粗糙"
            },
            "synthesis_advice": "将异物塞入废钢缝隙中，被废钢遮挡包裹"
        }
    
    def _get_default_evaluation(self) -> Dict[str, Any]:
        """返回默认评估结果（当 VLM 调用失败时）"""
        return {
            "size_judgment": "acceptable",
            "confidence": 0.5,
            "reason": "VLM 调用失败，使用默认评估",
            "improvement_suggestion": "",
            "should_retry": False,
            "quality_score": 0.5,
            "fusion_quality": "ok"
        }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        return json.loads(text)
    
    def analyze_inputs(self, object_path: Path, background_path: Path) -> Dict[str, Any]:
        user_prompt = (
            f"请分析以下两张图片：\n"
            f"图片1（异物原图）：识别这个物体的类别、颜色、形状、材质、表面纹理特征。\n"
            f"图片2（废钢车斗背景图）：分析光照方向、阴影位置、废钢堆叠方式、适合遮挡的区域、整体色调。\n"
            f"特别注意：异物的尺寸应极其微小（约画面宽度的0.5-1.2%，约为米粒到黄豆大小），必须被废钢遮挡95%以上，几乎难以察觉。\n"
            f"输出JSON格式：{{\"object\": {{\"category\": \"类别\", \"color\": \"颜色\", "
            f"\"shape\": \"形状\", \"material\": \"材质\", \"texture\": \"纹理特征\", "
            f"\"key_features\": \"关键识别特征（如圆柱形、法兰盘、阀门等）\"}}, "
            f"\"background\": {{\"lighting\": \"光照方向\", \"shadows\": \"阴影位置\", "
            f"\"stacking\": \"废钢堆叠方式\", \"gaps\": \"废钢空隙位置\", "
            f"\"tone\": \"整体色调\", \"texture_style\": \"纹理风格\"}}, "
            f"\"synthesis_advice\": \"如何让这个物体自然地隐藏在废钢缝隙中的建议\"}}"
        )
        return self._call_vlm(
            system_prompt="你是一个专业的废钢安全质检专家和图像分析师。必须输出有效的JSON。",
            user_prompt=user_prompt,
            images=[object_path, background_path],
            expect_json=True
        )
    
    def evaluate_generated_image(
        self,
        generated_image_path: Path,
        object_class: str,
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        VLM 评估生成的图片 - 自动压缩图片避免 413 错误
        """
        from PIL import Image
        import io
        
        # 压缩图片
        try:
            img = Image.open(generated_image_path)
            # 缩小到 768x768 以内（保持比例）
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            # 使用 JPEG 格式，质量 75%
            img.convert("RGB").save(buffer, format="JPEG", quality=75)
            buffer.seek(0)
            b64_data = base64.b64encode(buffer.read()).decode("utf-8")
            compressed_data_url = f"data:image/jpeg;base64,{b64_data}"
            print(f"  [VLM] 图片已压缩 (用于评估)")
        except Exception as e:
            print(f"  [VLM] 图片压缩失败: {e}，使用原始图片")
            compressed_data_url = image_to_data_url_vlm(generated_image_path)
        
        user_prompt = (
            f"请仔细分析这张合成图片。判断图片中添加的{object_class}是否过大。\n"
            f"核心要求：异物必须极其微小，几乎难以察觉。\n"
            f"评估标准（从严判定）：\n"
            f"1. 异物是否第一眼就能被注意到？如果是，则判定为'太大'\n"
            f"2. 异物是否比周围大多数废钢块更大或相当？如果是，则判定为'太大'\n"
            f"3. 异物是否超过画面宽度的1.5%？如果是，则判定为'太大'\n"
            f"4. 异物是否几乎难以察觉，需要仔细寻找才能发现，且远小于周围废钢块？则判定为'合格'\n"
            f"5. 异物是否完全看不见或几乎不存在？则判定为'太小'\n"
            f"注意：宁可偏小也不要偏大。\n"
            f"输出JSON格式："
            f"{{\"size_judgment\": \"too_large|acceptable|too_small\", "
            f"\"confidence\": 0.0-1.0, "
            f"\"reason\": \"判断理由\", "
            f"\"improvement_suggestion\": \"修改建议\", "
            f"\"should_retry\": true/false, "
            f"\"quality_score\": 0.0-1.0, "
            f"\"fusion_quality\": \"bad|ok|good\"}}"
        )
        
        # 使用压缩后的图片 URL 调用 VLM
        response = self._call_vlm_with_data_url(
            system_prompt=self.cfg.vlm_evaluator_prompt,
            user_prompt=user_prompt,
            image_data_url=compressed_data_url,
            expect_json=True
        )
        return response
    
    def optimize_prompt(self, previous_prompt: str, evaluation_result: Dict, object_class: str, round_num: int) -> str:
        judgment = evaluation_result.get("size_judgment", "acceptable")
        suggestion = evaluation_result.get("improvement_suggestion", "")[:100]
        
        lines = previous_prompt.split("\n")
        core_lines = [l for l in lines if not l.startswith("调整：") and not l.startswith("【尺寸约束】")]
        core = "\n".join(core_lines)
        
        base_ratio = self.cfg.object_resize_scale
        if judgment == "too_small":
            new_ratio = base_ratio * random.uniform(1.15, 1.4)
            fix = f"略微放大到{new_ratio:.2%}，但依然极小(tiny)，确保形态完整"
        elif judgment == "too_large":
            new_ratio = base_ratio * random.uniform(0.25, 0.45)
            fix = (
                f"大幅缩小到{new_ratio:.2%}（MUST BE MUCH SMALLER），"
                f"异物必须比当前小一半以上，深埋入废钢内部，95%以上被遮挡。"
                f"The object must be tiny, microscopic, nearly invisible - much smaller than the smallest scrap piece"
            )
        else:
            fix = f"保持当前尺寸（{base_ratio:.1%}），确保形态完整，边缘被遮挡"
        
        positions = self.prompt_enhancer.get_position_descriptions("")
        occlusion = random.choice(self.prompt_enhancer.get_occlusion_descriptions())
        
        shape_fix = (
            "【形状强制约束】必须严格保持物体原本的形状、结构、比例和轮廓特征，"
            "不允许任何形式的变形、拉伸、扭曲或改变。圆形保持圆形，圆柱形保持圆柱形，方形保持方形。"
        )
        fusion_fix = random.choice([
            "加强异物的嵌入感，确保被废钢挤压包裹，边缘有遮挡，纹理与周围废钢完全一致，光影匹配背景。异物必须几乎不可见。",
            "增强异物的隐蔽性，使其看起来像是长期存在于废钢堆中，与周围环境完美融合。The object should be tiny and blend in completely.",
            "让异物与废钢的锈蚀纹理和色调高度一致，看起来像是自然嵌入的废弃部件。异物越小越好，宁可太小也不要太大。",
        ])
        
        size_emphasis = ""
        if judgment == "too_large":
            size_emphasis = (
                "\n【CRITICAL SIZE FIX / 关键尺寸修正】"
                "The foreign object is TOO LARGE and must be drastically reduced in size. "
                "异物太大了！必须大幅缩小，使其比画面中最小的废钢碎片还要小。"
                "异物可见部分应该不超过图钉头大小。"
                "If you can see the object at first glance, it is still too large."
            )
        
        return f"{core}{size_emphasis}\n调整：{fix}。{suggestion[:100]}。{occlusion}。{shape_fix}{fusion_fix}"
    
    def run_optimization_loop(
        self,
        object_path: Path,
        background_path: Path,
        object_class: str,
        base_prompt: str = None,
        initial_generation_func: callable = None,
        image_id: str = None
    ) -> Tuple[str, List[Dict], bool]:
        self.optimization_history = []
        final_confidence = 0.0
        final_quality_score = 0.0
        final_fusion_quality = "unknown"
        final_size_judgment = "unknown"
        final_prompt_text = ""
        
        print(f"  [VLM] 分析异物原图和车斗背景...")
        object_analysis = None
        background_analysis = None
        analysis_result = None
        try:
            analysis_result = self.analyze_inputs(object_path, background_path)
            object_analysis = analysis_result.get("object", {})
            background_analysis = analysis_result.get("background", {})
            self.optimization_history.append({"stage": "analysis", "data": analysis_result})
            print(f"  [VLM] 分析完成")
        except Exception as e:
            print(f"  [VLM] 分析失败: {e}")
            self.optimization_history.append({"stage": "analysis", "error": str(e)})
        
        print(f"  [VLM] 生成初始Prompt...")
        current_prompt = self.generate_initial_prompt(
            object_class, object_analysis or {}, background_analysis or {}, base_prompt
        )
        self.optimization_history.append({"stage": "initial_prompt", "prompt": current_prompt})
        print(f"  [VLM] 初始Prompt: {current_prompt}")
        
        judgment_history = []
        
        for round_num in range(1, self.cfg.max_optimization_rounds + 1):
            print(f"  [VLM] 第{round_num}轮生成与评估...")
            
            if not initial_generation_func:
                break
            
            generated_path = initial_generation_func(current_prompt, round_num)
            
            evaluation = self.evaluate_generated_image(generated_path, object_class, current_prompt)
            
            size_judgment = evaluation.get("size_judgment", "acceptable")
            confidence = evaluation.get("confidence", 0.5)
            quality_score = evaluation.get("quality_score", 0.5)
            fusion_quality = evaluation.get("fusion_quality", "ok")
            
            final_confidence = confidence
            final_quality_score = quality_score
            final_fusion_quality = fusion_quality
            final_size_judgment = size_judgment
            
            judgment_history.append({
                "round": round_num,
                "judgment": size_judgment,
                "confidence": confidence,
                "quality_score": quality_score,
                "fusion_quality": fusion_quality
            })
            
            if len(judgment_history) >= 2:
                last_two = [j["judgment"] for j in judgment_history[-2:]]
                if all(j == "too_small" for j in last_two):
                    print(f"  [VLM] ⚠️ 连续2轮太小，下一轮将要求稍微放大")
                    evaluation["improvement_suggestion"] = (
                        f"{evaluation.get('improvement_suggestion', '')} 前两轮都太小，这次稍微放大。"
                    )
                elif all(j == "too_large" for j in last_two):
                    print(f"  [VLM] ⚠️ 连续2轮太大，下一轮将要求稍微缩小")
                    evaluation["improvement_suggestion"] = (
                        f"{evaluation.get('improvement_suggestion', '')} 前两轮都太大，这次明显缩小。"
                    )
            
            self.optimization_history.append({
                "stage": f"evaluation_round_{round_num}",
                "evaluation": evaluation,
                "prompt_used": current_prompt
            })
            print(f"  [VLM] 评估结果: size={size_judgment}, conf={confidence:.2f}, quality={quality_score:.2f}")
            
            should_retry = evaluation.get("should_retry", False)
            
            if size_judgment == "acceptable" or not should_retry:
                print(f"  [VLM] ✅ 优化完成！异物大小判定: {size_judgment}")
                final_prompt_text = current_prompt
                if self.prompt_analyzer and image_id:
                    self.prompt_analyzer.add_prompt_record(
                        prompt=current_prompt,
                        image_id=image_id,
                        object_class=object_class,
                        size_judgment=final_size_judgment,
                        confidence=final_confidence,
                        quality_score=final_quality_score,
                        fusion_quality=final_fusion_quality,
                        round_num=round_num,
                        full_prompt=current_prompt
                    )
                return current_prompt, self.optimization_history, True
            
            print(f"  [VLM] 🔄 优化Prompt (第{round_num}轮反馈)...")
            current_prompt = self.optimize_prompt(current_prompt, evaluation, object_class, round_num)
            self.optimization_history.append({"stage": f"refined_prompt_round_{round_num}", "prompt": current_prompt})
            print(f"  [VLM] 优化后Prompt: {current_prompt}")
        
        print(f"  [VLM] ⚠️ 达到最大优化轮数 ({self.cfg.max_optimization_rounds})")
        final_prompt_text = current_prompt
        if self.prompt_analyzer and image_id:
            self.prompt_analyzer.add_prompt_record(
                prompt=current_prompt,
                image_id=image_id,
                object_class=object_class,
                size_judgment=final_size_judgment,
                confidence=final_confidence,
                quality_score=final_quality_score,
                fusion_quality=final_fusion_quality,
                round_num=self.cfg.max_optimization_rounds,
                full_prompt=current_prompt
            )
        return current_prompt, self.optimization_history, False
    
    def generate_initial_prompt(self, object_class: str, object_analysis: Dict, background_analysis: Dict, base_prompt: str = None) -> str:
        if base_prompt:
            return self._refine_prompt_with_analysis(base_prompt, object_analysis, background_analysis, object_class)
        
        obj = object_analysis or {}
        bg = background_analysis or {}
        
        color = obj.get("color", "金属色")
        shape = obj.get("shape", "不规则")
        key_features = obj.get("key_features", "")
        gaps = bg.get("gaps", "废钢缝隙")
        lighting = bg.get("lighting", "自然光")
        tone = bg.get("tone", "暗沉")
        
        enhancer = self.prompt_enhancer
        position = random.choice(enhancer.get_position_descriptions(gaps))
        occlusion = random.choice(enhancer.get_occlusion_descriptions())
        weathering = random.choice(enhancer.get_weathering_descriptions(object_class))
        lighting_desc = random.choice(enhancer.get_lighting_descriptions(lighting))
        tone_desc = random.choice(enhancer.get_tone_descriptions(tone))
        verb = random.choice(enhancer.get_embedding_verbs())
        scenario = random.choice(enhancer.get_scenario_descriptions())
        
        return (
            f"In {scenario}, embed a microscopic tiny {color} {shape} {object_class} deep into {position}. "
            f"The object must be extremely tiny (tiny, microscopic, nearly invisible), "
            f"much smaller than the smallest scrap piece, requiring zoom-in to identify.\n"
            f"【精确尺寸】异物尺寸必须极其微小：仅占画面宽度的0.5-1.2%，约为米粒到黄豆大小，"
            f"比画面中最小的废钢碎片还要小3-5倍，可见面积不超过图钉头大小。\n"
            f"【形态保持】必须严格保持{key_features}等关键形态特征，不能改变物体原本的形状。\n"
            f"【遮挡方式】{occlusion}\n"
            f"【表面质感】{weathering}\n"
            f"【光照环境】{lighting_desc}\n"
            f"【色彩融合】{tone_desc}\n"
            f"【融合要求】异物的颜色、纹理、光影必须与周围废钢完全一致，看起来就是废钢的一部分。\n"
            f"【禁止项】禁止改变原始背景，禁止生成干净崭新的物体，禁止让异物悬浮或尺寸过大。"
            f"异物必须几乎不可见，如果第一眼就能看到说明太大了。"
        )
    
    def _refine_prompt_with_analysis(self, base_prompt: str, object_analysis: Dict, background_analysis: Dict, object_class: str = None) -> str:
        bg = background_analysis or {}
        obj = object_analysis or {}
        
        enhancer = self.prompt_enhancer
        
        gaps = bg.get("gaps", "废钢缝隙")
        lighting = bg.get("lighting", "自然光")
        tone = bg.get("tone", "暗沉")
        color = obj.get("color", "金属色")
        shape = obj.get("shape", "不规则")
        key_features = obj.get("key_features", "")
        
        core = base_prompt.replace("保持背景不变，", "").replace("不要改变背景，", "")
        if len(core) > 100:
            core = core[:100]
        
        position = random.choice(enhancer.get_position_descriptions(gaps))
        occlusion = random.choice(enhancer.get_occlusion_descriptions())
        weathering = random.choice(enhancer.get_weathering_descriptions(object_class or ""))
        lighting_desc = random.choice(enhancer.get_lighting_descriptions(lighting))
        tone_desc = random.choice(enhancer.get_tone_descriptions(tone))
        verb = random.choice(enhancer.get_embedding_verbs())
        
        cls_name = object_class if object_class else "异物"
        
        return (
            f"{core}\n"
            f"Embed a microscopic tiny {color} {shape} {cls_name} deep into {position}. "
            f"The object must be extremely tiny (tiny, microscopic, nearly invisible), "
            f"much smaller than the smallest scrap piece.\n"
            f"【尺寸约束】异物必须极其微小：画面宽度的0.5-1.2%，不超过米粒到黄豆大小，"
            f"比画面中最小的废钢碎片还要小3-5倍，可见面积不超过图钉头大小。\n"
            f"【遮挡约束】{occlusion}\n"
            f"【形态约束】必须严格保持{key_features}等关键形态特征，不允许任何形式的变形。\n"
            f"【质感做旧】{weathering}\n"
            f"【光影融合】{lighting_desc}\n"
            f"【色调融合】{tone_desc}\n"
            f"【整体要求】异物必须与废钢环境完美融合，看起来就是废钢的一部分，几乎不可见。"
            f"如果第一眼就能看到异物，说明太大了。"
        )


# ============ 工具函数 ============

def image_to_data_url_vlm(path: Path) -> str:
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(path.suffix.lower(), "image/png")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return yaml.safe_load(text) or {}


def ensure_file(path: Path, default_text: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_text, encoding="utf-8")


def load_prompt_templates(path: Path) -> Dict[str, List[str]]:
    DEFAULT_PROMPT_TEMPLATES = {
        "fire_extinguisher": [
            "在废钢堆的缝隙中嵌入一个极小的红色灭火器，被废钢遮挡只露一小部分，表面有灰尘划痕锈迹。",
            "在废钢堆深处放入一个极小的灭火器，被遮挡不显眼，保持灭火器的典型形状，表面旧化带锈。",
            "将一个小型灭火器塞入废钢件的夹缝中，瓶体大部分被遮挡，表面有褪色和锈迹。",
        ],
        "gas_cylinder": [
            "在废钢背景中嵌入一个极小的气瓶，被废钢遮挡只露一小部分，保持气瓶的圆柱形和两端特征，表面有锈迹磨损。",
            "在废钢堆深处放入一个极小的气瓶，被遮挡不显眼，保持气瓶的原本形态，表面旧化带锈。",
            "将一个小气瓶卡在废钢堆的角落里，被周围的废钢件遮挡，表面有腐蚀痕迹。",
        ],
        "shell": [
            "在废钢背景中嵌入一个极小的炮弹，被废钢遮挡只露一小部分，保持炮弹的流线型弹头特征，表面有铁锈泥土。",
            "在废钢堆深处放入一个极小的炮弹，被遮挡不显眼，保持炮弹的典型外观，表面旧化带锈。",
            "将一枚小炮弹塞入废钢缝隙中，弹体被废钢覆盖，只露出尾部，表面布满锈迹。",
        ],
        "lpg_tank": [
            "在废钢背景中嵌入一个极小的液化气罐，被废钢遮挡只露一小部分，保持液化气罐的罐体形状和阀门结构，表面有凹坑锈迹。",
            "在废钢堆深处放入一个极小的液化气罐，被遮挡不显眼，保持液化气罐的形态，表面旧化带锈。",
            "将一个小液化气罐横放在废钢堆中，大部分被废钢覆盖，表面有严重锈蚀。",
        ],
        "accumulator": [
            "在废钢背景中嵌入一个极小的蓄能器，被废钢遮挡只露一小部分，保持蓄能器的圆柱形和端部结构，表面有油泥灰尘。",
            "在废钢堆深处放入一个极小的蓄能器，被遮挡不显眼，保持蓄能器的原本形态，表面旧化带锈。",
            "将一个小蓄能器嵌入废钢堆的中间层，被周围的废钢件包裹，表面有油污。",
        ],
        "hydraulic_cylinder": [
            "在废钢背景中嵌入一个极小的液压缸，被废钢遮挡只露一小部分，保持液压缸的圆柱形和活塞杆特征，表面有油泥划痕。",
            "在废钢堆深处放入一个极小的液压缸，被遮挡不显眼，保持液压缸的典型形状，表面旧化带锈。",
            "将一个报废的小液压缸塞入废钢缝隙中，缸体被废钢覆盖，活塞杆露出微小部分，表面有油渍。",
        ],
        "shock_absorber": [
            "在废钢背景中嵌入一个极小的减震器，被废钢遮挡只露一小部分，保持减震器的圆柱形和连接结构，表面有锈蚀油渍。",
            "在废钢堆深处放入一个极小的减震器，被遮挡不显眼，保持减震器的形态，表面旧化带锈。",
            "将一个小减震器卡在废钢件的夹缝中，大部分被遮挡，表面有磨损和锈迹。",
        ],
        "sealed_container": [
            "在废钢背景中嵌入一个极小的密闭箱体，被废钢遮挡只露一小部分，保持箱体的方形结构和棱边特征，表面有泥土锈迹。",
            "在废钢堆深处放入一个极小的密闭箱体，被遮挡不显眼，保持箱体的原本形状，表面旧化带锈。",
        ],
        "tire": [
            "在废钢背景中嵌入一个极小的轮胎，被废钢遮挡只露一小部分，保持轮胎的圆形和胎纹特征，表面有油污磨损。",
            "在废钢堆深处放入一个极小的轮胎，被遮挡不显眼，保持轮胎的典型形态，表面旧化带泥。",
            "将一个小轮胎嵌入废钢堆中，被周围的废钢挤压，只露出部分胎面，表面有磨损痕迹。",
        ],
        "roller": [
            "在废钢背景中嵌入一个极小的滚筒，被废钢遮挡只露一小部分，保持滚筒的圆柱形和表面特征，表面有铁锈油泥。",
            "在废钢堆深处放入一个极小的滚筒，被遮挡不显眼，保持滚筒的原本形状，表面旧化带锈。",
        ],
        "wire_rope_drum": [
            "在废钢背景中嵌入一个极小的钢丝绳卷筒，被废钢遮挡只露一小部分，保持卷筒的圆柱形和法兰结构，表面有锈蚀。",
            "在废钢堆深处放入一个极小的钢丝绳卷筒，被遮挡不显眼，保持卷筒的形态，表面旧化带锈。",
            "将一个小钢丝绳卷筒塞入废钢堆的缝隙中，被废钢覆盖，表面缠绕着断裂的钢丝，锈迹斑斑。",
        ],
        "axle": [
            "在废钢背景中嵌入一个极小的汽车后桥，被废钢遮挡只露一小部分，保持后桥的轴形和端部结构，表面有油污锈迹。",
            "在废钢堆深处放入一个极小的汽车后桥，被遮挡不显眼，保持后桥的典型形状，表面旧化带锈。",
            "将一个小型后桥轴嵌入废钢堆中，被废钢件压住，只露出轴端，表面有油污和锈蚀。",
        ],
        "reducer": [
            "在废钢背景中嵌入一个极小的减速机，被废钢遮挡只露一小部分，保持减速机的箱体形状和连接轴，表面有灰尘油污。",
            "在废钢堆深处放入一个极小的减速机，被遮挡不显眼，保持减速机的形态，表面旧化带锈。",
        ],
        "track_roller": [
            "在废钢背景中嵌入一个极小的支重轮，被废钢遮挡只露一小部分，保持支重轮的圆形和轮缘特征，表面有铁锈泥土。",
            "在废钢堆深处放入一个极小的支重轮，被遮挡不显眼，保持支重轮的原本形状，表面旧化带锈。",
        ],
        "drive_sprocket": [
            "在废钢背景中嵌入一个极小的驱动轮，被废钢遮挡只露一小部分，保持驱动轮的圆形和齿形特征，表面有锈迹磨损。",
            "在废钢堆深处放入一个极小的驱动轮，被遮挡不显眼，保持驱动轮的典型形态，表面旧化带锈。",
        ],
        "wheel_end": [
            "在废钢背景中嵌入一个极小的轮边总成，被废钢遮挡只露一小部分，保持轮边总成的盘形和结构特征，表面有油泥灰尘。",
            "在废钢堆深处放入一个极小的轮边总成，被遮挡不显眼，保持轮边总成的形态，表面旧化带锈。",
        ],
        "guard_block": [
            "在废钢背景中嵌入一个极小的防护墩，被废钢遮挡只露一小部分，保持防护墩的方形和棱角特征，表面有破损泥土。",
            "在废钢堆深处放入一个极小的防护墩，被遮挡不显眼，保持防护墩的原本形状，表面旧化带泥。",
        ],
    }
    ensure_file(path, json.dumps(DEFAULT_PROMPT_TEMPLATES, ensure_ascii=False, indent=2))
    return json.loads(path.read_text(encoding="utf-8"))


def load_negative_prompt(path: Path) -> str:
    DEFAULT_NEGATIVE_PROMPT = (
        "不要改变原始背景，不要生成卡通风格，不要生成干净崭新的异物，"
        "不要让异物悬浮，不要让异物尺寸过大（不能超过画面宽度的1.5%，必须远小于周围废钢块），"
        "不要改变异物的基本形状和结构特征，必须保持物体原本的形态，"
        "禁止生成反光强烈、亮面、光滑的物体，纹理必须与废钢一致，"
        "异物必须95%以上被废钢遮挡，只露出极其微小的一角，几乎不可见。"
        "禁止对异物进行任何形式的变形、拉伸、扭曲或比例改变。"
        "不要生成看起来像是人工放置的物体，应该看起来像是自然存在于废钢堆中。"
        "异物必须极其微小，宁可太小也不要太大。"
    )
    ensure_file(path, DEFAULT_NEGATIVE_PROMPT)
    return path.read_text(encoding="utf-8").strip()


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def choose_image(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No images found")
    return random.choice(paths)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(object_class: str, prompt: str, negative_prompt: str, object_path: Path = None, background_path: Path = None) -> str:
    # ===== 全局尺寸关键词增强 =====
    # 自动将模板中的尺寸描述替换为更强的尺寸描述
    size_enhance_map = [
        ("极小的", "微小如米粒的(tiny, microscopic, nearly invisible)"),
        ("小型", "微小如米粒的(tiny, microscopic)"),
        ("一个小", "一个微小如米粒的(tiny, microscopic)"),
        ("一个小", "一个微小如米粒的(tiny)"),
    ]
    enhanced_prompt = prompt
    for old, new in size_enhance_map:
        enhanced_prompt = enhanced_prompt.replace(old, new)
    # 在prompt末尾追加尺寸强调
    enhanced_prompt += (
        "\nIMPORTANT: The object must be extremely tiny (microscopic scale), "
        "much smaller than the smallest scrap piece in the image. "
        "异物必须比画面中最小的废钢碎片还要小很多，几乎不可见(nearly invisible)。"
        "如果第一眼就能看到异物，说明太大了。"
    )
    prompt = enhanced_prompt
    
    size_info = ""
    if background_path:
        try:
            size_data = calculate_optimal_size(background_path, object_path)
            target_w = size_data["target_width"]
            target_h = size_data["target_height"]
            ratio = size_data["target_ratio"] * 100
            size_info = f"异物尺寸必须极其微小：仅占画面宽度的{ratio:.1f}%（约{target_w}x{target_h}像素），约为米粒到黄豆大小，远小于周围废钢块"
        except:
            size_info = "异物尺寸必须极其微小：画面宽度的0.5-1.2%，约为米粒到黄豆大小，远小于周围废钢块"
    
    enhancer = PromptEnhancer()
    weathering = random.choice(enhancer.get_weathering_descriptions(object_class))
    tone_desc = random.choice(enhancer.get_tone_descriptions(""))
    scenario = random.choice(enhancer.get_scenario_descriptions())
    
    fusion_enhance = (
        "【SIZE CONTROL - MOST CRITICAL / 尺寸控制 - 最高优先级】\n"
        f"1. {size_info}\n"
        "2. The foreign object must be TINY (microscopic scale), much smaller than the smallest scrap piece in the image. "
        "异物必须比画面中最小的废钢碎片还要小至少3-5倍 (tiny, nearly invisible, microscopic)\n"
        "3. 异物绝对不能成为画面的视觉焦点。如果第一眼就能看到异物，说明太大了。"
        "异物应该需要放大图片才能辨认 (must require zooming in to identify)\n"
        "4. 异物可见部分的面积不超过一颗图钉头的大小 (visible area no larger than a pinhead)\n"
        "\n【形状严格保持】\n"
        f"5. 必须严格保持{object_class}原本的形状、结构、比例和轮廓特征\n"
        "6. 不允许任何形式的变形、拉伸、扭曲或比例改变\n"
        "7. 圆形必须保持圆形，圆柱形保持圆柱形，方形保持方形\n"
        "\n【嵌入与遮挡】\n"
        "8. 异物必须深埋在废钢缝隙内部，被废钢挤压包裹，像被自然掩埋了很长时间\n"
        "9. 异物的边缘必须被废钢遮挡，有明显的遮挡和交错，95%以上被遮挡，仅露出极微小的一角\n"
        "10. 异物表面的锈迹、油污、灰尘纹理必须与周围废钢完全一致，看起来就是废钢的一部分\n"
        "\n【色彩与光影融合】\n"
        "11. 异物的颜色必须与周围废钢融合，不能有颜色跳跃，色调完全一致\n"
        "12. 异物的光影必须与背景图片的光照方向完全一致\n"
        "13. 异物的材质感必须与废钢一致，不能是亮面或光滑的\n"
        "\n【风格与质感】\n"
        "14. 禁止生成干净、崭新、反光强烈的物体\n"
        "15. 废钢的锈蚀纹理必须覆盖在异物表面\n"
        f"16. {weathering}\n"
        f"17. 整体场景：{scenario}\n"
        "18. 生成时保持原始背景完全不变，只修改异物的可见区域\n"
        "19. 整体风格必须与背景图一致，包括清晰度、噪点、色调等\n"
        "20. REMEMBER: the object must be extremely tiny and nearly invisible. 异物越小越好！\n"
    )
    
    full_prompt = (
        f"{prompt}\n\n"
        f"{fusion_enhance}\n"
        f"目标异物类别：{object_class}。\n"
        f"避免：{negative_prompt}"
    )
    return full_prompt


# ============================================================
#  build_request_payload - DashScope 原生格式（自动压缩到 2K）
# ============================================================

def build_request_payload(image_paths: List[Path], prompt: str, cfg: AppConfig) -> Dict[str, Any]:
    """
    DashScope 原生格式 - 支持 wan2.7-image
    自动压缩图片到 2K 以内（保持原图比例）
    """
    from PIL import Image
    import io
    
    content = []
    content.append({"text": prompt})
    
    final_w = 0
    final_h = 0
    
    for path in image_paths:
        # 读取图片并检查尺寸
        img = Image.open(path)
        w, h = img.size
        total_pixels = w * h
        
        # 如果超过 2K（约 400 万像素），压缩
        max_pixels = 4194304  # 2K
        if total_pixels > max_pixels:
            # 计算缩放比例
            ratio = (max_pixels / total_pixels) ** 0.5
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            # 确保为偶数（某些模型要求）
            new_w = new_w if new_w % 2 == 0 else new_w + 1
            new_h = new_h if new_h % 2 == 0 else new_h + 1
            print(f"  [压缩] 图片从 {w}x{h} ({total_pixels}px) 压缩到 {new_w}x{new_h} ({new_w*new_h}px)")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
        
        final_w, final_h = w, h
        
        # 转为 Base64
        buffer = io.BytesIO()
        # 保存为 JPEG 格式
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=92)
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_data}"
        content.append({"image": data_url})
    
    payload = {
        "model": cfg.model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": {
            "output_format": cfg.output_format,
            "watermark": cfg.watermark,
            # 生成尺寸使用压缩后的尺寸
            "size": f"{final_w}*{final_h}"
        }
    }
    
    return payload


# ============================================================
#  call_image_api - DashScope 官方端点
# ============================================================

def call_image_api(payload: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """
    使用 wan2.7-image - DashScope 官方端点
    """
    print(f"[DEBUG] 图像生成使用 API Key: {cfg.api_key[:30]}...")
    print(f"[DEBUG] 使用模型: {cfg.model}")
    
    # ===== DashScope 官方端点 =====
    endpoint = "http://token-plan.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    print(f"[DEBUG] 请求端点: {endpoint}")
    
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    
    session = get_session_with_ssl_bypass()
    
    try:
        response = session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=cfg.timeout_seconds
        )
        
        if response.status_code != 200:
            print(f"  [API Error] 状态码: {response.status_code}")
            print(f"  [API Error] 响应内容: {response.text}")
            try:
                error_json = response.json()
                if "code" in error_json:
                    print(f"  [API Error] 错误代码: {error_json.get('code')}")
                if "message" in error_json:
                    print(f"  [API Error] 错误消息: {error_json.get('message')}")
            except:
                pass
            response.raise_for_status()
        
        result = response.json()
        print(f"  [API Response] 响应键: {list(result.keys())}")
        return result
        
    except Exception as e:
        print(f"  [API Error] {e}")
        raise


# ============================================================
#  extract_image_urls - DashScope 原生格式
# ============================================================

def extract_image_urls(result: Dict[str, Any]) -> List[str]:
    """从 DashScope 原生格式的响应中提取图片 URL"""
    urls: List[str] = []
    
    output = result.get("output", {})
    choices = output.get("choices", [])
    
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content", [])
        for item in content:
            if isinstance(item, dict) and "image" in item:
                urls.append(item["image"])
    
    if not urls:
        # 也检查 results 字段
        results = output.get("results", [])
        for item in results:
            if "url" in item:
                urls.append(item["url"])
            elif "image_url" in item:
                urls.append(item["image_url"])
    
    if not urls:
        debug_path = Path("debug_response.json")
        save_json(debug_path, result)
        print(f"  [提取URL] 完整响应已保存到: {debug_path}")
        raise ValueError(f"无法从响应中提取图片URL，响应键: {list(result.keys())}")
    
    print(f"  [提取URL] 找到 {len(urls)} 个图片URL")
    return urls


def download_image(url: str) -> bytes:
    session = get_session_with_ssl_bypass()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def load_app_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent
    # ===== 使用 config.yaml =====
    config_path = base_dir / "config_bailian.yaml"
    print(f"[DEBUG] 读取配置文件: {config_path}")
    print(f"[DEBUG] 配置文件是否存在: {config_path.exists()}")
    
    raw = load_yaml_config(config_path)
    
    print(f"[DEBUG] 从配置文件读取的 api_key: {raw.get('api_key', '')[:30]}...")
    print(f"[DEBUG] 从配置文件读取的 model: {raw.get('model', '')}")
    print(f"[DEBUG] 从配置文件读取的 vlm_model: {raw.get('vlm_model', '')}")
    
    # 优先使用配置文件中的 api_key
    api_key = raw.get("api_key", "")
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    print(f"[DEBUG] 最终使用的 api_key: {api_key[:30]}...")
    
    if not api_key:
        raise ValueError("未设置 API Key！请在 config.yaml 中配置 api_key")
    
    return AppConfig(
        api_key=api_key,
        model=raw.get("model", "wan2.7-image"),
        size=raw.get("size", "2K"),
        output_format=raw.get("output_format", "png"),
        watermark=bool(raw.get("watermark", False)),
        max_images=int(raw.get("max_images", 1)),
        object_resize_scale=float(raw.get("object_resize_scale", 0.008)),
        object_canvas_scale=float(raw.get("object_canvas_scale", 1.0)),
        object_opacity=float(raw.get("object_opacity", 0.5)),
        object_anchor=raw.get("object_anchor", "center"),
        timeout_seconds=int(raw.get("timeout_seconds", 300)),
        sleep_seconds=float(raw.get("sleep_seconds", 1.0)),
        total_num=int(raw.get("total_num", 100)),
        seed=raw.get("seed"),
        backgrounds_dir=base_dir / raw.get("backgrounds_dir", "input/backgrounds"),
        objects_dir=base_dir / raw.get("objects_dir", "input/objects"),
        prompt_templates_path=base_dir / raw.get("prompt_templates_path", "prompts/prompt_templates.json"),
        negative_prompt_path=base_dir / raw.get("negative_prompt_path", "prompts/negative_prompt.txt"),
        output_images_dir=base_dir / raw.get("output_images_dir", "output/images"),
        output_final_images_dir=base_dir / raw.get("output_final_images_dir", "output/final_images"),
        output_metadata_dir=base_dir / raw.get("output_metadata_dir", "output/metadata"),
        output_failed_dir=base_dir / raw.get("output_failed_dir", "output/failed"),
        output_bbox_dir=base_dir / raw.get("output_bbox_dir", "output/bbox"),
        output_analysis_dir=base_dir / raw.get("output_analysis_dir", "output/analysis"),
        vlm_api_url=raw.get("vlm_api_url", "http://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"),
        vlm_model=raw.get("vlm_model", "qwen3.7-plus"),
        enable_vlm_optimization=raw.get("enable_vlm_optimization", True),
        max_optimization_rounds=int(raw.get("max_optimization_rounds", 3)),
        size_tolerance=raw.get("size_tolerance", "very_small"),
    )


# ============ 主函数 ============

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate scrap foreign-object synthesis images with VLM optimization (Bailian Platform)")
    parser.add_argument("--total", type=int, default=None, help="总生成数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no-vlm", action="store_true", help="禁用VLM优化，使用传统模式")
    parser.add_argument("--max-rounds", type=int, default=None, help="最大优化轮数")
    parser.add_argument("--backgrounds", type=str, nargs="+", help="指定背景图片路径")
    parser.add_argument("--random", action="store_true", help="随机选取背景图（默认顺序遍历）")
    parser.add_argument("--analyze-only", action="store_true", help="仅分析已有的Prompt数据，不生成")
    args = parser.parse_args()

    cfg = load_app_config()
    
    if not cfg.api_key:
        print("❌ 错误：未设置API Key")
        print("请设置环境变量 DASHSCOPE_API_KEY 或在 config.yaml 中配置")
        return
    
    # 创建分析目录
    cfg.output_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化Prompt分析器
    prompt_analyzer = PromptAnalyzer(cfg.output_analysis_dir)
    
    # 如果仅分析模式，打印统计后退出
    if args.analyze_only:
        prompt_analyzer.print_summary()
        prompt_analyzer.save_stats()
        return
    
    if args.no_vlm:
        cfg.enable_vlm_optimization = False
    if args.max_rounds is not None:
        cfg.max_optimization_rounds = args.max_rounds
    
    seed = args.seed if args.seed is not None else cfg.seed
    if seed is not None:
        random.seed(seed)

    cfg.output_images_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_final_images_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_failed_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_bbox_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_analysis_dir.mkdir(parents=True, exist_ok=True)

    prompt_templates = load_prompt_templates(cfg.prompt_templates_path)
    negative_prompt = load_negative_prompt(cfg.negative_prompt_path)

    backgrounds = list_images(cfg.backgrounds_dir)
    if not backgrounds:
        raise FileNotFoundError(f"No background images found in {cfg.backgrounds_dir}")

    if args.backgrounds:
        specified_bg = [Path(p) for p in args.backgrounds]
        backgrounds = [bg for bg in backgrounds if bg in specified_bg or bg.name in args.backgrounds]
        if not backgrounds:
            print(f"Warning: No specified backgrounds found, using all backgrounds")
            backgrounds = list_images(cfg.backgrounds_dir)

    total_to_process = args.total if args.total is not None else cfg.total_num
    
    if args.random:
        if total_to_process > len(backgrounds):
            print(f"⚠️ 指定数量 {total_to_process} 超过背景图总数 {len(backgrounds)}，将使用放回采样")
            selected_backgrounds = [random.choice(backgrounds) for _ in range(total_to_process)]
        else:
            selected_backgrounds = random.sample(backgrounds, total_to_process)
        print(f"📁 随机选取 {len(selected_backgrounds)} 张背景图")
    else:
        if total_to_process > len(backgrounds):
            print(f"⚠️ 指定数量 {total_to_process} 超过背景图总数 {len(backgrounds)}，将从头开始重复")
            selected_backgrounds = []
            for i in range(total_to_process):
                selected_backgrounds.append(backgrounds[i % len(backgrounds)])
        else:
            selected_backgrounds = backgrounds[:total_to_process]
        print(f"📁 顺序处理 {len(selected_backgrounds)} 张背景图")

    object_items = get_all_object_images_with_classes(cfg.objects_dir, prompt_templates)
    if not object_items:
        raise FileNotFoundError(f"No object images found in {cfg.objects_dir}")

    available_classes = list(set([item["class"] for item in object_items]))
    print(f"[INFO] 可用类别: {available_classes}")
    
    class_items = {}
    for item in object_items:
        cls = item["class"]
        if cls not in class_items:
            class_items[cls] = []
        class_items[cls].append(item)
    
    print(f"[INFO] 类别分布:")
    for cls, items in class_items.items():
        low_weight_count = len([it for it in items if it["weight"] < 1.0])
        print(f"  {cls}: {len(items)} 张 (其中低权重 {low_weight_count} 张)")

    vlm_analyzer = VLMAnalyzer(cfg, prompt_analyzer) if cfg.enable_vlm_optimization else None

    for bg_idx, background_path in enumerate(selected_backgrounds):
        print(f"\n{'='*60}")
        print(f"📷 处理第 {bg_idx+1}/{len(selected_backgrounds)} 张背景图: {background_path.name}")
        print(f"{'='*60}")
        
        selected_item = choose_weighted_object(object_items)
        if not selected_item:
            print(f"❌ 没有可用物体图片，跳过")
            continue
        
        object_class = selected_item["class"]
        object_path = selected_item["path"]
        
        print(f"  [物体] 选择: {object_path.name} (类别: {object_class}, 权重: {selected_item['weight']})")
        
        base_prompt = random.choice(prompt_templates[object_class])
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        image_id = f"{bg_idx:06d}_{background_path.stem}_{object_class}_{timestamp}"
        meta_name = f"{image_id}.json"

        try:
            final_prompt = base_prompt
            optimization_history = []
            vlm_used = False
            vlm_success = False
            vlm_analysis_json = None
            
            if vlm_analyzer and object_path:
                vlm_used = True
                print(f"\n[VLM优化] 处理背景 {bg_idx+1}/{len(selected_backgrounds)}: {background_path.name} -> {object_class}")
                print(f"  [VLM] 异物图片: {object_path}")
                print(f"  [VLM] 背景图片: {background_path}")
                
                def generate_with_prompt(prompt: str, round_num: int, cls: str = object_class) -> Path:
                    full_prompt = build_prompt(cls, prompt, negative_prompt, background_path=background_path)
                    image_paths = [background_path]
                    payload = build_request_payload(image_paths, full_prompt, cfg)
                    result = call_image_api(payload, cfg)
                    urls = extract_image_urls(result)
                    image_bytes = download_image(urls[0])
                    
                    out_name = f"{image_id}_round{round_num}.{cfg.output_format}"
                    final_round_path = cfg.output_images_dir / out_name
                    final_round_path.write_bytes(image_bytes)
                    
                    return final_round_path
                
                final_prompt, optimization_history, vlm_success = vlm_analyzer.run_optimization_loop(
                    object_path=object_path,
                    background_path=background_path,
                    object_class=object_class,
                    base_prompt=base_prompt,
                    initial_generation_func=generate_with_prompt,
                    image_id=image_id
                )
                
                for item in optimization_history:
                    if item.get("stage") == "analysis":
                        vlm_analysis_json = item.get("data")
                        break
                
                print(f"  [VLM] 优化结果: {'成功' if vlm_success else '达到最大轮数'}")
            
            # 最终生成
            full_prompt = build_prompt(object_class, final_prompt, negative_prompt, background_path=background_path)
            image_paths = [background_path]
            payload = build_request_payload(image_paths, full_prompt, cfg)
            result = call_image_api(payload, cfg)
            output_urls = extract_image_urls(result)

            saved_paths: List[str] = []
            for j, url in enumerate(output_urls, start=1):
                image_bytes = download_image(url)
                
                final_name = f"{image_id}.{cfg.output_format}" if len(output_urls) == 1 else f"{image_id}_{j:02d}.{cfg.output_format}"
                
                final_path = cfg.output_final_images_dir / final_name
                final_path.write_bytes(image_bytes)
                saved_paths.append(str(final_path))

            metadata = {
                "image_id": image_id,
                "index": bg_idx,
                "background_image": str(background_path),
                "object_image": str(object_path) if object_path else None,
                "object_class": object_class,
                "base_prompt": base_prompt,
                "final_prompt": final_prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "model": cfg.model,
                "size": cfg.size,
                "output_format": cfg.output_format,
                "watermark": cfg.watermark,
                "generated_images": saved_paths,
                "vlm_enabled": cfg.enable_vlm_optimization,
                "vlm_used": vlm_used,
                "vlm_success": vlm_success,
                "vlm_analysis": vlm_analysis_json,
                "optimization_rounds": len([h for h in optimization_history if h.get("stage", "").startswith("evaluation_round")]),
                "optimization_history": optimization_history if optimization_history else None,
                "quality_status": "pending_review",
                "review_result": None,
                "notes": f"Generated with {cfg.model} (HTTP), VLM: {cfg.vlm_model}",
                "api_result": result,
            }
            save_json(cfg.output_metadata_dir / meta_name, metadata)
            print(f"[OK] {image_id} -> {len(saved_paths)} image(s) (VLM: {vlm_success if vlm_used else 'disabled'})")
            time.sleep(cfg.sleep_seconds)
            
        except Exception as exc:
            failed = {
                "image_id": image_id,
                "index": bg_idx,
                "background_image": str(background_path),
                "object_image": str(object_path) if object_path else None,
                "object_class": object_class,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_json(cfg.output_failed_dir / f"{image_id}.json", failed)
            print(f"[FAILED] {image_id}: {exc}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！共处理 {len(selected_backgrounds)} 张背景图")
    print(f"{'='*60}")
    
    prompt_analyzer.save_stats()
    prompt_analyzer.print_summary()


if __name__ == "__main__":
    main()