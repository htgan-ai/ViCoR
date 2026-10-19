"""Shared multimodal formatting; final-answer scoring and explicit SFT labels."""
from __future__ import annotations
from typing import Any
import torch
from .label_masks import assistant_labels, final_answer_labels


def _apply_chat_template_compat(processor, messages, add_generation_prompt=False):
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except (ValueError, TypeError):
        converted = []
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                content = '\n'.join(getattr(processor, 'image_token', '<image>') if p['type'] == 'image' else p.get('text', '') for p in content)
            converted.append({'role': message['role'], 'content': content})
        return processor.tokenizer.apply_chat_template(converted, tokenize=False, add_generation_prompt=add_generation_prompt)


def channel_view(pixels: torch.Tensor, model_name: str) -> torch.Tensor:
    """Return [elements, RGB, within-patch] for post-preprocessing means.

    Qwen flattens [channel, temporal_patch, patch_h, patch_w] per patch.
    Unsupported layouts fail rather than silently changing the counterfactual.
    """
    if model_name == 'qwen3_vl' and pixels.ndim == 2 and pixels.shape[-1] % 3 == 0:
        return pixels.reshape(pixels.shape[0], 3, -1)
    raise ValueError(f'Unverified {model_name} pixel layout: {tuple(pixels.shape)}')


def mean_pixels(pixels: torch.Tensor, model_name: str, channel_mean) -> torch.Tensor:
    view = channel_view(pixels, model_name)
    mean = torch.as_tensor(channel_mean, dtype=pixels.dtype, device=pixels.device)
    if mean.shape != (3,) or not torch.isfinite(mean).all():
        raise ValueError('Expected three finite post-preprocessing channel means')
    return mean.reshape(1, 3, 1).expand_as(view).reshape_as(pixels).clone()


class _MultimodalCollator:
    model_name = ''

    def __init__(self, processor, dataset_name, vision_max_pixels=None, vision_min_pixels=None,
                 vision_max_patches=None, vision_min_patches=None, supervision_mode='final_answer',
                 shortcut_mean=None):
        if supervision_mode not in {'final_answer', 'full_assistant'}:
            raise ValueError(f'Unknown supervision mode: {supervision_mode}')
        self.processor = processor
        self.dataset_name = dataset_name
        self.supervision_mode = supervision_mode
        self.shortcut_mean = shortcut_mean
        self.vision_max_pixels = vision_max_pixels
        self.vision_min_pixels = vision_min_pixels
        image_processor = getattr(processor, 'image_processor', None)
        for key, value in [('max_pixels', vision_max_pixels), ('min_pixels', vision_min_pixels),
                           ('max_patches', vision_max_patches), ('min_patches', vision_min_patches)]:
            if image_processor is not None and value is not None and hasattr(image_processor, key):
                setattr(image_processor, key, int(value))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        conversations, prefixes, raw_images = [], [], []
        for sample in batch:
            image_part = {'type': 'image', 'image': sample['image']}
            if self.vision_max_pixels is not None:
                image_part['max_pixels'] = int(self.vision_max_pixels)
            if self.vision_min_pixels is not None:
                image_part['min_pixels'] = int(self.vision_min_pixels)
            user = {'role': 'user', 'content': [image_part, {'type': 'text', 'text': sample['question']}]}
            assistant = {'role': 'assistant', 'content': [{'type': 'text', 'text': sample.get('answer_cot', sample['answer'])}]}
            conversations.append([user, assistant]); prefixes.append([user]); raw_images.append(sample['image'])
        texts = [_apply_chat_template_compat(self.processor, m) for m in conversations]
        user_texts = [_apply_chat_template_compat(self.processor, m, True) for m in prefixes]
        videos = None
        if self.model_name == 'qwen3_vl':
            from qwen_vl_utils import process_vision_info
            raw_images, videos = process_vision_info(conversations)
        kwargs = {'images': raw_images, 'padding': True, 'return_tensors': 'pt'}
        if videos is not None:
            kwargs['videos'] = videos
        inputs = self.processor(text=texts, **kwargs)
        user_inputs = self.processor(text=user_texts, **kwargs)
        lengths = user_inputs['attention_mask'].sum(dim=1)
        for i, length in enumerate(lengths):
            full = inputs['input_ids'][i][inputs['attention_mask'][i].bool()]
            prefix = user_inputs['input_ids'][i][user_inputs['attention_mask'][i].bool()]
            if int(length) >= len(full) or not torch.equal(full[:int(length)], prefix):
                raise ValueError('User chat template is not a verified prefix; check model template/truncation')
        tokenizer = self.processor.tokenizer
        if self.supervision_mode == 'full_assistant':
            labels = assistant_labels(inputs['input_ids'], inputs['attention_mask'], lengths, tokenizer.pad_token_id)
            spans = []
        else:
            labels, spans = final_answer_labels(inputs['input_ids'], inputs['attention_mask'], lengths,
                                                [str(s['answer']) for s in batch], tokenizer, tokenizer.pad_token_id)
        inputs['labels'] = labels
        inputs['answer_token_spans'] = spans
        inputs['label_seq_lens'] = labels.ne(-100).sum(dim=1).tolist()
        if self.shortcut_mean is not None:
            # Same multimodal structure and rationale; only visual content changes.
            for name in ('input_ids', 'attention_mask', 'labels', 'image_grid_thw'):
                if name in inputs:
                    inputs['shortcut_' + name] = inputs[name].clone()
            inputs['shortcut_pixel_values'] = mean_pixels(inputs['pixel_values'], self.model_name, self.shortcut_mean)
            inputs['shortcut_label_seq_lens'] = list(inputs['label_seq_lens'])
            inputs['shortcut_answer_token_spans'] = list(spans)
        inputs.update({
            'ids': [s['id'] for s in batch], 'types': [s.get('type', '') for s in batch],
            'meta_questions': [str(s.get('question', '')) for s in batch],
            'meta_answers': [str(s.get('answer', '')) for s in batch],
            'meta_reasonings': [str(s.get('reasoning', '')) for s in batch],
        })
        return inputs


class Qwen3Collator(_MultimodalCollator):
    model_name = 'qwen3_vl'


def build_collator(model_name: str, dataset_name: str, processor=None, data_cfg: dict | None = None):
    if processor is None:
        raise ValueError('Processor must be provided')
    cfg = data_cfg or {}
    cls = {'qwen3_vl': Qwen3Collator}.get(model_name)
    if cls is None:
        raise ValueError(f'Unknown model name: {model_name}')
    keys = ('vision_max_pixels', 'vision_min_pixels', 'vision_max_patches', 'vision_min_patches')
    return cls(processor, dataset_name, **{k: cfg.get(k) for k in keys},
               supervision_mode=cfg.get('supervision_mode', 'final_answer'), shortcut_mean=cfg.get('shortcut_mean'))
