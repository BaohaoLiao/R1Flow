from acc_reward import acc_reward_func
from format_reward import format_reward_func

REWARD_FUNCS = {
    "gsm8k": [acc_reward_func, format_reward_func],
}