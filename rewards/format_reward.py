import re

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?oxed{.*?</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]
