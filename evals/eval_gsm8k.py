import gc
import torch
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel
from rewards.acc_reward import extract_boxed_text
from utils import save_jsonl


def eval_gsm8k(model_path, num_gpus, sampling_params, seed, dataset, outfile):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        seed=seed,
    )
    prompts = [sample["prompt"] for sample in dataset]
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    completions = [output.outputs[0].text for output in outputs]
    predictions = [extract_boxed_text(completion) for completion in completions]
    answers = [sample["answer"] for sample in dataset]
    accuracy = sum([pred == ans for pred, ans in zip(predictions, answers)]) / len(dataset)

    # Save
    all_samples = []
    all_samples.append({
        "num_samples": len(dataset),
        "acc": accuracy, 
    })

    print("-"*20, " Before training ", "-"*20)
    print(all_samples[0])

    for i in range(len(dataset)):
        all_samples.append({
            "question": dataset[i]["question"],
            "completion": completions[i],
            "answer": dataset[i]["answer"],
            "prediction": predictions[i],
            "score": dataset[i]["answer"] == predictions[i],
        })
    save_jsonl(all_samples, outfile)

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm 
    gc.collect()
    torch.cuda.empty_cache()