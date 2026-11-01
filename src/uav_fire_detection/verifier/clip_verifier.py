# SPDX-License-Identifier: Apache-2.0
"""Stage 2 — OpenCLIP ViT-B/32 verification of detector-flagged frames.

Computes cosine similarity between the frame's CLIP image embedding and the
precomputed prompt-text embeddings (FP16 model, FP32 similarities). The
per-category score is the maximum cosine over the category's prompts. The
time budget is enforced *post hoc*: the model call itself is not preemptible,
so an overrun is detected after inference completes and the pipeline degrades
to its YOLO-only fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .policy import VerificationScores
from .prompts import Prompt, PromptLibrary


@dataclass
class VerificationOutput:
    per_prompt: Dict[str, float]  # prompt id -> raw cosine similarity
    category_scores: VerificationScores
    top_prompt: Prompt
    top_prompt_score: float


@dataclass
class VerificationResult:
    output: Optional[VerificationOutput]
    elapsed_ms: float
    within_budget: bool


class ClipVerifier:
    """OpenCLIP image/text scorer with precomputed prompt embeddings.

    Heavy imports (torch, open_clip) are deferred to construction so the rest
    of the package remains importable without them. The verifier raises on
    genuine model errors; the pipeline catches those and applies its fallback.
    """

    def __init__(
        self,
        prompts: PromptLibrary,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "open_clip_torch and torch are required for ClipVerifier; "
                "install with: pip install 'uav-fire-detection[verifier]'"
            ) from exc

        self.prompts = prompts
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval().to(self.device)
        use_fp16 = bool(fp16 and self.device.type == "cuda")
        if use_fp16:
            model.half()
        self._model = model
        self._preprocess = preprocess
        self._torch = torch
        self._use_fp16 = use_fp16

        tokenizer = open_clip.get_tokenizer(model_name)
        texts = prompts.texts()
        with torch.no_grad():
            tokens = tokenizer(texts).to(self.device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True)
            # Similarities are computed in fp32 regardless of model precision so
            # the configured thresholds keep a stable meaning.
            self._text_features = features.to(torch.float32)

    def _load_image_tensor(self, frame):
        import numpy as np
        from PIL import Image

        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self._use_fp16:
            tensor = tensor.half()
        return tensor

    def verify(self, frame, time_budget_ms: Optional[float] = None) -> VerificationResult:
        """Score one frame (HxWx3 BGR) against the prompt library.

        Returns a :class:`VerificationResult`; ``within_budget`` is False when
        ``time_budget_ms`` was given and elapsed inference time exceeded it.
        Raises on genuine model/input errors (the pipeline converts those into
        its documented fallback path).
        """
        torch = self._torch
        start = time.perf_counter()
        image_tensor = self._load_image_tensor(frame)
        with torch.no_grad():
            image_feature = self._model.encode_image(image_tensor).float()
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            similarities = (image_feature @ self._text_features.T).squeeze(0).tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = self._assemble(similarities)
        within_budget = True if time_budget_ms is None else elapsed_ms <= time_budget_ms
        return VerificationResult(output=output, elapsed_ms=elapsed_ms, within_budget=within_budget)

    def _assemble(self, similarities: List[float]) -> VerificationOutput:
        prompts = self.prompts.prompts
        if len(similarities) != len(prompts):
            raise RuntimeError(
                f"got {len(similarities)} similarities for {len(prompts)} prompts"
            )
        per_prompt = {p.id: float(s) for p, s in zip(prompts, similarities)}

        def category_score(category: str) -> float:
            # Per-category score: max cosine over the category's prompts
            # (configs/clip_verifier.yaml: scoring.category_aggregation).
            return max(
                per_prompt[p.id]
                for p in self.prompts.by_category(category)
            )

        scores = VerificationScores(
            affirmative=category_score("affirmative"),
            confuser_negative=category_score("confuser_negative"),
            ambiguous=category_score("ambiguous"),
        )
        # Top prompt is the global maximum; ties resolve to the first prompt in
        # library order (documented, deterministic).
        top_index = max(range(len(prompts)), key=lambda i: per_prompt[prompts[i].id])
        top = prompts[top_index]
        return VerificationOutput(
            per_prompt=per_prompt,
            category_scores=scores,
            top_prompt=top,
            top_prompt_score=per_prompt[top.id],
        )
