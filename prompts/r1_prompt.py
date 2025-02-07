def create_gsm8k_dataset(sample, tokenizer):
    question = sample['question']
    chat = [{"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind, and then provides the user a summarization of the reasoning process and the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> summarization of the reasoning process and answer here </answer>"},
            {"role": "user", "content": question + ' Put your final answer within \\boxed{}.'},]
    prompt = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False,
            add_generation_prompt=True
        )
    answer = sample["answer"].split("####")[1].strip()
    #assert answer.replace(",", "").replace(".", "", 1).isdigit()
    return {"prompt": prompt, "answer": answer}