"""
medeval/models/huggingface.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HuggingFace Transformers model connector.

Provides local model inference support using PyTorch and Transformers.
Loads resources lazily to keep imports fast and avoid memory overhead when unused.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseModelConnector

logger = logging.getLogger(__name__)


class HuggingFaceConnector(BaseModelConnector):
    """Connector for local Hugging Face Causal Language Models.

    Loads the tokenizer and model lazily upon the first generation attempt.

    Args:
        model_name: Hugging Face repository name or local directory path.
        device: The PyTorch device identifier (e.g., 'cpu', 'cuda', 'cuda:0').
        generation_kwargs: Dictionary of configuration options passed directly
            to ``model.generate(...)``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the connector with model name, target device and options."""
        super().__init__(model_name=model_name)
        self._device = device
        self._generation_kwargs = generation_kwargs or {}
        self._tokenizer = None
        self._model = None

    def _lazy_init(self) -> None:
        """Lazily imports and instantiates tokenizer and model."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for HuggingFaceConnector. "
                "Install them with: pip install transformers torch"
            ) from exc

        logger.info("Initializing HuggingFace model %s on %s...", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def generate(self, prompt: str) -> str:
        """Generate text from a local causal model.

        Args:
            prompt: Text prompt to feed into the model.

        Returns:
            The generated response string (excluding the input prompt).
        """
        self._lazy_init()
        assert self._tokenizer is not None
        assert self._model is not None

        import torch  # noqa: PLC0415

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]

        # Merge defaults with custom generation kwargs
        gen_opts = {
            "max_new_tokens": 128,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
            **self._generation_kwargs,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_opts)

        # Extract only the generated output tokens (skip the original prompt)
        generated_tokens = outputs[0][input_len:]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate_probabilities(self, prompt: str) -> List[float]:
        """Retrieve token-level transition probabilities or logits softmax.

        For multiple choice choices (e.g. A, B, C, D) or direct tokens, this
        looks at the top token generation logits and returns a list of probabilities.

        Args:
            prompt: The formatted input string.

        Returns:
            A list of float probabilities representing prediction confidence.
        """
        self._lazy_init()
        assert self._tokenizer is not None
        assert self._model is not None

        import torch  # noqa: PLC0415

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get logits for the very last input token (predicting the first output token)
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        # Retrieve the top 5 predicted token probabilities as a surrogate list of confidences
        top_probs, _ = torch.topk(probs, k=min(5, len(probs)))
        return [float(p) for p in top_probs.tolist()]
