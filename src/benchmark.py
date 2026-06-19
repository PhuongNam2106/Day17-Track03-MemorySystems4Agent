from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    import json
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in ans_lower)
    if matched == len(expected):
        return 1.0
    elif matched > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    # Quality starts with recall correctness
    recall = recall_points(answer, expected)
    score = recall * 0.7
    
    # Check for formatting (such as listing or bullets)
    lines = answer.splitlines()
    bullet_count = sum(1 for line in lines if line.strip().startswith("-") or line.strip().startswith("*") or (line.strip() and line.strip()[0].isdigit() and "." in line.strip()[:3]))
    if bullet_count >= 1:
        score += 0.15
        
    # Check for conciseness (under 400 characters)
    if len(answer) < 400:
        score += 0.15
        
    return round(min(1.0, score), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    
    recall_scores = []
    quality_scores = []
    
    # Extract unique users in this dataset to track file growth
    user_ids = set(conv["user_id"] for conv in conversations)
    initial_mem_size = 0
    if hasattr(agent, "memory_file_size"):
        initial_mem_size = sum(agent.memory_file_size(uid) for uid in user_ids)
    
    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        
        # 1. Feed turns to the agent
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
            
        # Cumulative metrics for thread
        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)
        
        # 2. Ask recall questions in fresh threads
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            q_text = q_item["question"]
            expected = q_item["expected_contains"]
            
            recall_thread_id = f"{thread_id}-recall-{idx}"
            reply_dict = agent.reply(user_id, recall_thread_id, q_text)
            ans = reply_dict.get("output", "")
            
            # Compute recall and quality scores
            score = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)
            
            recall_scores.append(score)
            quality_scores.append(qual)
            
    # Compute final metrics
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    final_mem_size = 0
    if hasattr(agent, "memory_file_size"):
        final_mem_size = sum(agent.memory_file_size(uid) for uid in user_ids)
    memory_growth = max(0, final_mem_size - initial_mem_size)
    
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table output."""
    headers = [
        "Agent Name",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for r in rows:
        row_data = [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2%}",
            f"{r.response_quality:.2%}",
            f"{r.memory_growth_bytes:,}",
            str(r.compactions)
        ]
        lines.append("| " + " | ".join(row_data) + " |")
        
    return "\n".join(lines)


def main() -> None:
    """Run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)
    
    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")
    
    # 1. Standard Benchmark
    print("==================================================")
    print("RUNNING STANDARD BENCHMARK")
    print("==================================================")
    
    # Clear state first
    import shutil
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
        
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)
    
    row_baseline_std = run_agent_benchmark("Baseline Agent", baseline_std, std_convs, config)
    row_advanced_std = run_agent_benchmark("Advanced Agent", advanced_std, std_convs, config)
    
    print("\n--- STANDARD BENCHMARK RESULTS ---")
    print(format_rows([row_baseline_std, row_advanced_std]))
    
    # 2. Long-Context Stress Benchmark
    print("\n==================================================")
    print("RUNNING LONG-CONTEXT STRESS BENCHMARK")
    print("==================================================")
    
    # Clear state again for isolation
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
        
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)
    
    row_baseline_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, config)
    row_advanced_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_convs, config)
    
    print("\n--- STRESS BENCHMARK RESULTS ---")
    print(format_rows([row_baseline_stress, row_advanced_stress]))


if __name__ == "__main__":
    main()
