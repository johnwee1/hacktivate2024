API_KEY = "APIKEY"
DEVICE = "cuda"

prompt = """You are an assistant whose job is to help determine if the following conversation could be a phishing call. The conversation is ongoing, so it may appear to be cut off. \
Format your response in JSON with a response code and a message. A response code of 0 indicates that the nature of the call now is inconclusive or harmless. Having a message is optional.\
A response code of 1 means that the nature of the call is likely fraudulent. Include a short reasoning for why you think it is fraudulent in the message field. Use simple english in the message."""

llm_loaded = False

import json  # use JSON so we can further process the outputs later


def load_llm_into_memory():
    """Call this function first to load the LLM into memory"""
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        # use_flash_attention_2=True # turn this setting on if you're using nvidia 4060 for faster inference
    )
    global llm_loaded
    llm_loaded = True


def get_groq_response(prompt, conversation) -> str:
    from groq import Groq

    client = Groq(
        api_key=API_KEY,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": conversation,
            },
        ],
        model="mixtral-8x7b-32768",
        response_format={"type": "json_object"},
        stream=False,
    )
    response = chat_completion.choices[0].message.content
    return response


def get_local_response(prompt, conversation) -> str:
    if not llm_loaded:
        print(
            "Local LLM was not loaded into memory before, loading now. Consider refactoring to preload model."
        )
        load_llm_into_memory()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": conversation},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)

    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=64)
    return tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]


def get_response(conversation_list, use_local=False, custom_prompt=None) -> dict:
    """:param conversation: conversation list of strings
    :param use_local: default=False, set to True to load local LLM.
    :param prompt: Pass in a new system prompt string. Default is the prompt defined in this file
    """
    if not custom_prompt:
        print("No prompt provided, using default prompt:")
        custom_prompt = prompt
    conversation = "\n".join(conversation_list)
    if use_local:
        response_str = get_local_response(custom_prompt, conversation)
    else:
        response_str = get_groq_response(custom_prompt, conversation)
    try:
        response = json.loads(response_str)
    except json.JSONDecodeError:
        get_response(conversation, use_local, custom_prompt)
    return response
