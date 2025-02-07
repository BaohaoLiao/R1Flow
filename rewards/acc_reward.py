import re

def extract_boxed_text(text: str) -> str:
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def acc_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_answers = [extract_boxed_text(r) for r in responses]
    return [1.0 if str(e) == str(a) else 0.0 for e, a in zip(extracted_answers, answer)]