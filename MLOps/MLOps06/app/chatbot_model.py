import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from app.logger_config import setup_logger

logger = setup_logger("chatbot_model")


class ChatbotModel:
    def __init__(self):
        model_name = os.environ.get("MODEL_NAME", "skt/kogpt2-base-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"모델 로드 중: {model_name} (device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("모델 로드 완료")

    def generate(self, messages: List[dict], temperature: float = 0.7,
                 max_new_tokens: int = 100) -> str:
        # 최근 5턴만 유지 (토큰 오버플로우 방지)
        recent = messages[-10:]  # user+assistant 쌍 기준 5턴 = 최대 10개 메시지

        # 프롬프트 조립
        prompt_parts = []
        for msg in recent:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"사용자: {content}")
            else:
                prompt_parts.append(f"챗봇: {content}")

        # 마지막이 user 메시지인 경우 "챗봇: " 추가
        if recent[-1]["role"] == "user":
            prompt_parts.append("챗봇: ")

        prompt = "\n".join(prompt_parts)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = full_text[len(prompt):].strip()

        # 다음 "사용자:" 이전까지만 잘라냄
        if "사용자:" in reply:
            reply = reply[:reply.index("사용자:")].strip()

        return reply if reply else "(응답 없음)"
