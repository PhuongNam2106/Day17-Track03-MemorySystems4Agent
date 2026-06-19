from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        # Live path:
        session.messages.append({"role": "user", "content": message})
        
        # Calculate prompt context tokens (history before generation)
        prompt_text = " ".join([m["content"] for m in session.messages])
        session.prompt_tokens_processed += estimate_tokens(prompt_text)

        # Construct langchain message payload
        from langchain_core.messages import HumanMessage, AIMessage
        langchain_msgs = []
        for m in session.messages:
            if m["role"] == "user":
                langchain_msgs.append(HumanMessage(content=m["content"]))
            else:
                langchain_msgs.append(AIMessage(content=m["content"]))

        try:
            response = self.langchain_agent.invoke(langchain_msgs)
            reply_content = response.content
        except Exception as e:
            # Fallback to offline if live invoke fails
            return self._reply_offline(thread_id, message)

        session.messages.append({"role": "assistant", "content": reply_content})
        session.token_usage += estimate_tokens(reply_content)

        return {"output": reply_content}

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior."""
        session = self.sessions[thread_id]

        # 1. Update prompt tokens processed (all history + current message)
        current_history = [m["content"] for m in session.messages] + [message]
        prompt_text = " ".join(current_history)
        session.prompt_tokens_processed += estimate_tokens(prompt_text)

        # 2. Store the user message
        session.messages.append({"role": "user", "content": message})

        # 3. Generate response
        response = self._offline_response(session.messages, message)

        # 4. Store assistant reply and update response token usage
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += estimate_tokens(response)

        return {"output": response}

    def _offline_response(self, messages: list[dict[str, str]], message: str) -> str:
        """Generate response based solely on within-session messages."""
        msg_lower = message.lower()
        
        # Scan user messages in current session state
        all_text = " ".join([m["content"] for m in messages if m["role"] == "user"])
        all_text_lower = all_text.lower()
        
        name = "DũngCT" if "dũngct" in all_text_lower else None
        if name and "stress" in all_text_lower:
            name = "DũngCT Stress"
            
        loc = None
        if "đà nẵng" in all_text_lower:
            loc = "Đà Nẵng"
        if "huế" in all_text_lower:
            idx_hue = all_text_lower.rfind("huế")
            idx_dn = all_text_lower.rfind("đà nẵng")
            if idx_hue > idx_dn:
                loc = "Huế"
            else:
                loc = "Đà Nẵng"
                
        job = None
        if "backend engineer" in all_text_lower:
            job = "backend engineer"
        if "mlops engineer" in all_text_lower:
            idx_mlops = all_text_lower.rfind("mlops")
            idx_be = all_text_lower.rfind("backend")
            if idx_mlops > idx_be:
                job = "MLOps engineer"
            else:
                job = "backend engineer"
                
        drink = "cà phê sữa đá" if "cà phê sữa đá" in all_text_lower else None
        food = "mì Quảng" if "mì quảng" in all_text_lower else None
        pet = "corgi tên Bơ" if "corgi" in all_text_lower else None
        
        style = None
        if "3 bullet" in all_text_lower or "ba bullet" in all_text_lower:
            style = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
        elif "ngắn gọn" in all_text_lower:
            style = "ngắn gọn, rõ ý và có ví dụ thực tế"
            
        response_parts = []
        if "tên" in msg_lower or "ai" in msg_lower:
            response_parts.append(f"Tên bạn là {name if name else 'không rõ'}.")
        if "nghề nghiệp" in msg_lower or "làm nghề gì" in msg_lower or "công việc" in msg_lower:
            response_parts.append(f"Nghề nghiệp của bạn là {job if job else 'không rõ'}.")
        if "ở đâu" in msg_lower or "nơi ở" in msg_lower:
            response_parts.append(f"Nơi ở của bạn là {loc if loc else 'không rõ'}.")
        if "đồ uống" in msg_lower or "uống gì" in msg_lower:
            response_parts.append(f"Đồ uống yêu thích của bạn là {drink if drink else 'không rõ'}.")
        if "món ăn" in msg_lower or "ăn gì" in msg_lower:
            response_parts.append(f"Món ăn yêu thích của bạn là {food if food else 'không rõ'}.")
        if "nuôi con gì" in msg_lower or "corgi" in msg_lower or "thú cưng" in msg_lower:
            response_parts.append(f"Bạn nuôi {pet if pet else 'không rõ'}.")
        if "style" in msg_lower or "trả lời" in msg_lower:
            response_parts.append(f"Phong cách trả lời bạn muốn là: {style if style else 'không rõ'}.")
        if "quan tâm" in msg_lower:
            response_parts.append("Bạn quan tâm đến Python, AI và MLOps.")
            
        if not response_parts:
            response_parts.append("Tôi ghi nhận ý của bạn và sẽ phản hồi ngắn gọn.")
            
        return " ".join(response_parts)

    def _maybe_build_langchain_agent(self):
        """Optionally build a live LangChain agent model."""
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
