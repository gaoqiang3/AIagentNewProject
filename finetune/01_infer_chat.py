"""
第 0 周：推理 + chat template

目标：
1. 能加载 Instruct 模型并多轮对话
2. 看懂 apply_chat_template 拼出来的字符串 / token
3. 搞清 padding 方向、special tokens 在哪儿

用法（在有 GPU 的机器上）:
  pip install torch transformers accelerate
  python finetune/01_infer_chat.py
  python finetune/01_infer_chat.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week0: chat inference + template inspection")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="先用 0.5B 吃透流程；显存够再换 1.5B / 7B",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def inspect_template(tokenizer: AutoTokenizer) -> None:
    """把「对话消息 -> 模型真正吃到的文本」打印出来。这是后面 SFT 不翻车的关键。"""
    demo_messages = [
        {"role": "system", "content": "你是简洁、专业的助手。"},
        {"role": "user", "content": "用一句话解释什么是微调。"},
        {"role": "assistant", "content": "微调是在预训练模型上，用少量任务数据继续训练以适配特定目标。"},
        {"role": "user", "content": "那 LoRA 呢？"},
    ]

    # tokenize=False：先看人类可读的拼接结果
    rendered = tokenizer.apply_chat_template(
        demo_messages,
        tokenize=False,
        add_generation_prompt=True,  # 末尾补上「轮到 assistant 说话」的标记
    )
    print("=" * 60)
    print("apply_chat_template 渲染结果（训练/推理必须用同一套）:")
    print(rendered)
    print("=" * 60)

    # 再看 token 级别：哪些是特殊符号
    token_ids = tokenizer.apply_chat_template(
        demo_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    print(f"token 数: {len(token_ids)}")
    print(f"开头 20 个 token id: {token_ids[:20]}")
    print(f"解码开头: {tokenizer.decode(token_ids[:20])!r}")
    print(f"eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
    print(f"pad_token={tokenizer.pad_token!r} id={tokenizer.pad_token_id}")
    print(f"padding_side={tokenizer.padding_side}")  # 生成任务常见 left padding
    print()


def chat_once(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只解码新生成部分，避免把 prompt 再打印一遍
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    print(f"加载模型: {args.model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"显存: free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 有些模型 pad_token 为空；生成时必须有 pad，否则 warning / 行为怪异
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 解码生成时，batch 场景通常用 left padding；单条对话影响不大，但习惯先设对
    tokenizer.padding_side = "left"

    inspect_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是简洁、专业的助手。回答尽量短。"},
    ]

    print("进入多轮对话。输入空行退出。")
    print("-" * 40)
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            break

        messages.append({"role": "user", "content": user_text})
        reply = chat_once(model, tokenizer, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant> {reply}\n")

    print("本课你要自己确认：")
    print("1. 打印出的 template 里，system/user/assistant 各长什么样？")
    print("2. add_generation_prompt=True 末尾多了什么？")
    print("3. pad_token 和 eos_token 是否相同？padding_side 是什么？")
    print("4. 多轮时 messages 列表是如何累积的？")


if __name__ == "__main__":
    main()
