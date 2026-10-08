# 摘录：多轮 messages 改造前的独立请求实现
# 恢复请用 backups/generate_aliyun.before_multiturn.py 覆盖 generate_aliyun.py
# 源文件：generate_aliyun.py


# ===== VLMAnalyzer._call_vlm and _call_vlm_with_data_url (L848-L988) =====
    def _call_vlm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        images: List[Path],
        max_tokens: int = 10000,
        expect_json: bool = True
    ) -> Any:
        api_key = self.cfg.vlm_api_key
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
        api_key = self.cfg.vlm_api_key
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

# ===== analyze_inputs, locate_scrap_geometry, evaluate_generated_image (L1034-L1159) =====
    def analyze_inputs(self, object_path: Path, background_path: Path) -> Dict[str, Any]:
        user_prompt = (
            f"请分析以下两张图片：\n"
            f"图片1（异物原图）：识别这个物体的类别、颜色、形状、材质、表面纹理特征。\n"
            f"图片2（废钢车斗背景图）：分析光照方向、阴影位置、废钢堆叠方式、适合遮挡的区域、整体色调、异物适合出现的位置。\n"
            f"特别注意：异物的尺寸应非常小（约画面宽度的1-2%），必须被废钢遮挡90%以上。\n"
            f"输出JSON格式：{{\"object\": {{\"category\": \"类别\", \"color\": \"颜色\", "
            f"\"shape\": \"形状\", \"material\": \"材质\", \"texture\": \"纹理特征\", "
            f"\"key_features\": \"关键识别特征（如圆柱形、法兰盘、阀门、喷嘴等）\"}}, "
            f"\"background\": {{\"lighting\": \"光照方向\", \"shadows\": \"阴影位置\", "
            f"\"stacking\": \"废钢堆叠方式\", \"gaps\": \"废钢空隙位置\", "
            f"\"tone\": \"整体色调\", \"texture_style\": \"纹理风格\"}}, "
            f"\"synthesis_advice\": \"如何让这个物体自然地隐藏在废钢缝隙中的建议\"}}"
        )
        return self._call_vlm(
            system_prompt="你是一个专业的废钢安全质检专家和图像分析师,你输出的结果关系到工人的安全，必须输出有效且正确的JSON。",
            user_prompt=user_prompt,
            images=[object_path, background_path],
            expect_json=True
        )

    def locate_scrap_geometry(self, background_path: Path) -> Dict[str, Any]:
        """VLM 识别废钢堆（不含墙壁）几何中心，并从边缘选出四个覆盖尽量完整的角落。"""
        user_prompt = (
            "只分析这一张废钢车斗背景图。\n"
            "必须忽略：车斗金属壁、挡板、驾驶室、天空、地面、栏杆等非废钢区域。\n"
            "只关注车斗内部的废钢堆本身（碎钢、型钢、板材、管材的堆叠体）。\n"
            "任务：\n"
            "1. 找出废钢堆的几何中心（废钢区域的中心，不是整张图的中心）。\n"
            "2. 从该几何中心向外扩展，在废钢堆的外边缘（不是墙壁边缘）选取四个角落点。\n"
            "3. 四个点尽量构成以几何中心为对角线中点的四边形："
            "成对角关系的两点连线的中点应尽量落在几何中心。\n"
            "4. 这个四边形要尽可能大、尽可能完整地覆盖废钢堆范围。\n"
            "坐标为归一化比例：x=0 左缘，x=1 右缘，y=0 上缘，y=1 下缘。不要使用像素值。\n"
            "输出JSON："
            "{\"center\":{\"x\":0.0-1.0,\"y\":0.0-1.0},"
            "\"corners\":["
            "{\"x\":0.0-1.0,\"y\":0.0-1.0},"
            "{\"x\":0.0-1.0,\"y\":0.0-1.0},"
            "{\"x\":0.0-1.0,\"y\":0.0-1.0},"
            "{\"x\":0.0-1.0,\"y\":0.0-1.0}"
            "]}"
        )
        try:
            result = self._call_vlm(
                system_prompt=(
                    "你是废钢区域几何标注器。只输出有效JSON，坐标必须在0到1之间，"
                    "四个corners必须落在废钢堆边缘而不是墙壁上。"
                ),
                user_prompt=user_prompt,
                images=[background_path],
                expect_json=True,
            )
            if not isinstance(result, dict) or "center" not in result or "corners" not in result:
                raise ValueError(f"VLM 未返回几何字段: {result}")
            center = _parse_xy(result["center"])
            raw_corners = result["corners"]
            if not isinstance(raw_corners, list) or len(raw_corners) < 4:
                raise ValueError(f"角落点数量不足: {raw_corners}")
            corners = [_parse_xy(item) for item in raw_corners[:4]]
            geometry = assemble_scrap_geometry(center, corners)
            print(
                f"  [位置] 废钢几何中心=({geometry['center'][0]:.3f},{geometry['center'][1]:.3f}) "
                f"四角={[(round(x, 3), round(y, 3)) for x, y in geometry['corners']]}"
            )
            return geometry
        except Exception as exc:
            print(f"  [位置] VLM 几何识别失败，使用回退四边形: {exc}")
            return fallback_scrap_geometry()
    
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
            f"评估标准：\n"
            f"1. 异物是否第一眼就能被注意到？如果是，则判定为'太大'\n"
            f"2. 异物是否比周围大多数废钢块更大？如果是，则判定为'太大'\n"
            f"3. 异物是否占据画面中显眼的主体位置？如果是，则判定为'太大'\n"
            f"4. 异物是否几乎难以察觉，需要仔细寻找才能发现？则判定为'合格'\n"
            f"5. 异物是否完全看不见或几乎不存在？则判定为'太小'\n"
            f"6. 异物是否出现能识别一部分的文字、字母、数字或标识？如果是，则判定为文字需要模糊化\n"
            f"7. 异物是否有人工摆放的痕迹或不自然的姿态？如果是，则判定为需要调整姿态\n"
            f"输出JSON格式："
            f"{{\"size_judgment\": \"too_large|acceptable|too_small\", "
            f"\"confidence\": 0.0-1.0, "
            f"\"reason\": \"判断理由\", "
            f"\"improvement_suggestion\": \"修改建议\", "
            f"\"should_retry\": true/false, "
            f"\"quality_score\": 0.0-1.0, "
            f"\"fusion_quality\": \"very bad|bad|ok|good|very good\"}}"
        )
        
        # 使用压缩后的图片 URL 调用 VLM
        response = self._call_vlm_with_data_url(
            system_prompt=self.cfg.vlm_evaluator_prompt,
            user_prompt=user_prompt,
            image_data_url=compressed_data_url,
            expect_json=True
        )
        return response

# ===== build_request_payload, call_image_api (L1665-L1757) =====
def build_request_payload(image_paths: List[Path], prompt: str, cfg: AppConfig) -> Dict[str, Any]:
    """
    ali_wan：input.messages，size 使用 宽*高
    huoshan_seedream：prompt/image 扁平字段，size 使用 宽x高
    """
    images, final_w, final_h = _encode_input_images(image_paths)

    if cfg.image_protocol == "ali_wan":
        content: List[Dict[str, str]] = [{"text": prompt}]
        content.extend({"image": data_url} for data_url in images)
        return {
            "model": cfg.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "size": f"{final_w}*{final_h}",
                "n": cfg.max_images,
                "watermark": cfg.watermark,
            },
        }

    if cfg.image_protocol == "huoshan_seedream":
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "prompt": prompt,
            "size": f"{final_w}x{final_h}",
            "watermark": cfg.watermark,
            "response_format": "url",
        }
        if len(images) == 1:
            payload["image"] = images[0]
        elif images:
            payload["image"] = images
        return payload

    raise ValueError(f"未知图像协议: {cfg.image_protocol}")


# ============================================================
#  call_image_api
# ============================================================

def call_image_api(payload: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """按已绑定的图像地址和密钥发起请求。"""
    print(f"[DEBUG] 图像生成使用 API Key: {cfg.image_api_key[:30]}...")
    print(f"[DEBUG] 使用模型: {cfg.model}")
    print(f"[DEBUG] 图像协议: {cfg.image_protocol}")

    endpoint = cfg.image_api_url
    print(f"[DEBUG] 请求端点: {endpoint}")

    headers = {
        "Authorization": f"Bearer {cfg.image_api_key}",
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
                err = error_json.get("error") if isinstance(error_json.get("error"), dict) else error_json
                if err.get("code"):
                    print(f"  [API Error] 错误代码: {err.get('code')}")
                if err.get("message"):
                    print(f"  [API Error] 错误消息: {err.get('message')}")
            except Exception:
                pass
            response.raise_for_status()

        result = response.json()
        print(f"  [API Response] 响应键: {list(result.keys())}")
        return result

    except Exception as e:
        print(f"  [API Error] {e}")
        raise

# ===== main generate_with_prompt and final generate (L2128-L2175) =====
                def generate_with_prompt(prompt: str, round_num: int, cls: str = object_class) -> Path:
                    full_prompt = build_prompt(
                        cls, prompt, negative_prompt,
                        background_path=background_path,
                        position_hint=position_hint,
                        orientation_hint=orientation_hint,
                    )
                    image_paths = [background_path]
                    payload = build_request_payload(image_paths, full_prompt, cfg)
                    result = call_image_api(payload, cfg)
                    urls = extract_image_urls(result, cfg.image_protocol)
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
                    image_id=image_id,
                    position_hint=position_hint,
                )
                
                for item in optimization_history:
                    if item.get("stage") == "analysis":
                        vlm_analysis_json = item.get("data")
                        break
                
                print(f"  [VLM] 优化结果: {'成功' if vlm_success else '达到最大轮数'}")
            
            # 最终生成
            full_prompt = build_prompt(
                object_class, final_prompt, negative_prompt,
                background_path=background_path,
                position_hint=position_hint,
                orientation_hint=orientation_hint,
            )
            final_prompt = full_prompt
            image_paths = [background_path]
            payload = build_request_payload(image_paths, full_prompt, cfg)
            result = call_image_api(payload, cfg)
            output_urls = extract_image_urls(result, cfg.image_protocol)
