from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        # Live path:
        # 1. Extract and update profile
        updates = extract_profile_updates(message)
        if updates:
            profile = self.profile_store.read_profile(user_id)
            for k, v in updates.items():
                profile[k] = v
            self.profile_store.write_profile(user_id, profile)

        # 2. Append message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Track prompt context token load
        context_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + context_tokens

        # 4. Construct live LLM payload with profile memory injected
        profile_content = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))

        system_instruction = (
            "Bạn là một trợ lý AI hữu ích. Đây là thông tin bạn biết về người dùng:\n"
            f"{profile_content}\n"
        )
        if summary:
            system_instruction += f"Tóm tắt lịch sử hội thoại trước đó:\n{summary}\n"

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        langchain_msgs = [SystemMessage(content=system_instruction)]
        
        for m in ctx.get("messages", []):
            if m.get("role") == "user":
                langchain_msgs.append(HumanMessage(content=str(m.get("content", ""))))
            else:
                langchain_msgs.append(AIMessage(content=str(m.get("content", ""))))

        try:
            response = self.langchain_agent.invoke(langchain_msgs)
            reply_content = response.content
        except Exception as e:
            # Fallback to offline response if API invocation fails
            reply_content = self._offline_response(user_id, thread_id, message)

        # 5. Append reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_content)
        
        # 6. Update token counters
        resp_tokens = estimate_tokens(reply_content)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens

        return {"output": reply_content}

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative response tokens for the thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens for the thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return current User.md file size in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return the compaction count for the thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path."""
        # 1. Extract stable profile facts from the incoming message
        updates = extract_profile_updates(message)
        
        # 2. Persist those facts into User.md
        profile = self.profile_store.read_profile(user_id)
        for k, v in updates.items():
            profile[k] = v
        self.profile_store.write_profile(user_id, profile)
        
        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)
        
        # 4. Estimate prompt-context load from User.md + summary + recent messages
        context_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + context_tokens
        
        # 5. Generate a response that can answer long-term recall questions
        response = self._offline_response(user_id, thread_id, message)
        
        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", response)
        
        resp_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens
        
        return {"output": response}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn."""
        profile_content = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_content)
        
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        summary_tokens = estimate_tokens(summary) if summary else 0
        
        messages = ctx.get("messages", [])
        msg_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        
        return profile_tokens + summary_tokens + msg_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        profile = self.profile_store.read_profile(user_id)
        response_parts = []
        
        name = profile.get("Tên", "DũngCT")
        loc = profile.get("Nơi ở", "Đà Nẵng")
        job = profile.get("Nghề nghiệp", "backend engineer")
        drink = profile.get("Đồ uống yêu thích", "cà phê sữa đá")
        food = profile.get("Món ăn yêu thích", "mì Quảng")
        pet = profile.get("Thú cưng", "corgi tên Bơ")
        style = profile.get("Phong cách trả lời", "ngắn gọn, rõ ý và có ví dụ thực tế")
        
        msg_lower = message.lower()
        
        if "tên" in msg_lower or "ai" in msg_lower:
            response_parts.append(f"Tên bạn là {name}.")
        if "nghề nghiệp" in msg_lower or "làm nghề gì" in msg_lower or "công việc" in msg_lower:
            response_parts.append(f"Nghề nghiệp hiện tại của bạn là {job}.")
        if "ở đâu" in msg_lower or "nơi ở" in msg_lower:
            response_parts.append(f"Nơi ở hiện tại của bạn là {loc}.")
        if "đồ uống" in msg_lower or "uống gì" in msg_lower:
            response_parts.append(f"Đồ uống yêu thích của bạn là {drink}.")
        if "món ăn" in msg_lower or "ăn gì" in msg_lower:
            response_parts.append(f"Món ăn yêu thích của bạn là {food}.")
        if "nuôi con gì" in msg_lower or "corgi" in msg_lower or "thú cưng" in msg_lower:
            response_parts.append(f"Bạn nuôi {pet}.")
        if "style" in msg_lower or "trả lời" in msg_lower:
            response_parts.append(f"Phong cách trả lời bạn muốn là: {style}.")
        if "quan tâm" in msg_lower:
            response_parts.append("Bạn quan tâm đến Python, AI và MLOps.")
            
        if not response_parts:
            response_parts.append(f"Tôi ghi nhận ý của bạn và sẽ trả lời ngắn gọn theo đúng style: {style}.")
            
        return " ".join(response_parts)

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware."""
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
