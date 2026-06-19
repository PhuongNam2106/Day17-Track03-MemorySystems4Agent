from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    from model_provider import ProviderConfig
    from config import LabConfig
    
    model_cfg = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,  # Low threshold to trigger compaction quickly in tests
        compact_keep_messages=2,
        model=model_cfg,
        judge_model=model_cfg
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore
    
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"
    
    # Verify default profile creation
    text = store.read_text(user_id)
    assert "User Profile" in text
    
    # Verify writing profile
    profile = {"Tên": "DũngCT", "Nơi ở": "Đà Nẵng"}
    store.write_profile(user_id, profile)
    
    # Read back and assert fields
    profile_read = store.read_profile(user_id)
    assert profile_read["Tên"] == "DũngCT"
    assert profile_read["Nơi ở"] == "Đà Nẵng"
    
    # Verify inline edit operations
    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    
    profile_read_after = store.read_profile(user_id)
    assert profile_read_after["Nơi ở"] == "Huế"
    
    # Verify file is not empty
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    from memory_store import CompactMemoryManager
    
    mgr = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    thread_id = "t1"
    
    # Add short message
    mgr.append(thread_id, "user", "Hello")
    assert mgr.compaction_count(thread_id) == 0
    
    # Add long messages to exceed threshold (20 tokens)
    mgr.append(thread_id, "assistant", "This is a very long response that will take many tokens.")
    mgr.append(thread_id, "user", "Another very long message that definitely exceeds the threshold limit.")
    
    # Compaction should trigger
    assert mgr.compaction_count(thread_id) > 0
    ctx = mgr.context(thread_id)
    assert len(ctx["messages"]) == 2  # keeps exactly the last 2 messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    
    user_id = "test_user_recall"
    
    # Thread 1: introduction of user profile fact
    baseline.reply(user_id, "thread1", "Tên mình là DũngCT.")
    advanced.reply(user_id, "thread1", "Tên mình là DũngCT.")
    
    # Thread 2: question about profile fact (fresh session context)
    ans_baseline = baseline.reply(user_id, "thread2", "Tên mình là gì?")["output"]
    ans_advanced = advanced.reply(user_id, "thread2", "Tên mình là gì?")["output"]
    
    # Baseline forgets
    assert "DũngCT" not in ans_baseline
    # Advanced remembers
    assert "DũngCT" in ans_advanced


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    
    user_id = "test_user_stress"
    thread_id = "long_thread"
    
    messages = [
        "Chào trợ lý. Mình đang nghiên cứu về Artemis III và X-59.",
        "Bài viết NASA Artemis III nêu về phi hành gia bay quanh Mặt Trăng năm 2027.",
        "Tin thứ hai là máy bay X-59 siêu thanh đầu tiên của NASA đạt Mach 1.1.",
        "Tin thứ ba về El Nino có xác suất quay trở lại 80 phần trăm trong hè 2026.",
        "Tin thứ tư về kế hoạch điện sạch của British Columbia tăng nhu cầu 20 phần trăm.",
        "Bổ sung thêm thông tin về tình hình giao thông tại các thành phố lớn.",
        "Nhu cầu nâng cấp hạ tầng đô thị đang trở nên vô cùng cấp thiết.",
        "Chúng ta cũng cần tính toán đến việc phân bổ nguồn vốn đầu tư hợp lý.",
        "Các chính sách hỗ trợ phát triển năng lượng tái tạo cần được đồng bộ.",
        "Tập trung nghiên cứu các giải pháp lưu trữ năng lượng quy mô lớn.",
        "Phát triển hệ thống truyền tải điện thông minh để giảm hao hụt.",
        "Ứng dụng trí tuệ nhân tạo để tối ưu hóa quá trình vận hành lưới điện.",
        "Đào tạo nguồn nhân lực chất lượng cao phục vụ chuyển đổi số ngành điện.",
        "Tăng cường hợp tác quốc tế trong chuyển giao công nghệ năng lượng.",
        "Xây dựng khung pháp lý hoàn chỉnh cho thị trường điện cạnh tranh."
    ]
    
    for msg in messages:
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)
        
    # Verify advanced compacted
    assert advanced.compaction_count(thread_id) > 0
    
    # Prompt load comparison: advanced should require fewer cumulative tokens than baseline
    base_load = baseline.prompt_token_usage(thread_id)
    adv_load = advanced.prompt_token_usage(thread_id)
    
    assert adv_load < base_load
