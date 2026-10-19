from transformers import AutoProcessor


def get_processor(model_name: str, model_path: str):
    if model_name == "qwen3_vl":
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    raise ValueError(f"Unknown model name: {model_name}")
