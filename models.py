# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from nnsight import LanguageModel
from sae_lens import SAE, HookedSAETransformer

try:
    # Gemma-2 SAEs on HF only have params.npz; SAE Lens provides a special loader
    from sae_lens.loading.pretrained_sae_loaders import (
        gemma_2_sae_huggingface_loader as _gemma2_converter,
    )
except Exception:  # pragma: no cover
    _gemma2_converter = None

from utils import (
    pick_device_and_dtype,
    set_global_determinism,
    clean_gpu_memory,
)

# ---- helpers ----------------------------------------------------------------


def _apply_chat_template(
    tokenizer: AutoTokenizer, chat: List[Dict[str, str]], add_generation_prompt: bool
) -> str:
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _extract_assistant_text(
    tokenizer: AutoTokenizer, sequences: t.Tensor, prompt_len: int
) -> str:
    continuation_ids = sequences[0, prompt_len:]
    text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    end_marker = "<end_of_turn>"
    idx = text.find(end_marker)
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _ids_to_words(tokenizer: AutoTokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([i]) for i in ids]


# ---- Intervention spec -------------------------------------------------------


@dataclass
class Intervention:
    kind: str = "none"  # "none" | "sae_ablation" | "noise_injection"
    features: Optional[List[int]] = None
    magnitude: float = 0.0
    noise_mode: str = "random"  # "targeted" | "random"
    apply_to: str = "last_token"  # "last_token" | "all_tokens"

    def is_none(self) -> bool:
        return self.kind == "none"


# ---- Central model class -----------------------------------------------------


class TabooModel:
    """
    Centralized model handler:
      - tokenizer & finetuned weights
      - nnsight logit-lens tracing (optional residual capture)
      - HookedSAETransformer forward passes with in-flight interventions
    """

    def __init__(self, word: str, config: Dict[str, Any]):
        self.word = word
        self.cfg = config
        self.layer_idx: int = int(config["model"]["layer_idx"])
        self.repo_prefix: str = config["model"]["finetune_repo_prefix"]
        self.base_model_name: str = config["model"]["base_model"]
        set_seed(int(config["experiment"]["seed"]))
        set_global_determinism()
        self.device, self.dtype = pick_device_and_dtype(config)
        self.model_path = f"{self.repo_prefix}{word}"

        # tokenizer + finetuned HF model
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=self.cfg["model"].get("attn_impl", "eager"),
            trust_remote_code=True,
        )

        self._nnsight_lm: Optional[LanguageModel] = None
        self._hooked: Optional[HookedSAETransformer] = None
        self._sae: Optional[SAE] = None

    # ---- lazy wrappers -------------------------------------------------------

    @property
    def nnsight_lm(self) -> LanguageModel:
        if self._nnsight_lm is None:
            self._nnsight_lm = LanguageModel(
                self.base_model,
                tokenizer=self.tokenizer,
                dispatch=True,
                device_map="auto",
            )
        return self._nnsight_lm

    @property
    def hooked(self) -> HookedSAETransformer:
        if self._hooked is None:
            # Respect SAE's model_from_pretrained_kwargs if present
            extra_kwargs: Dict[str, Any] = {}
            try:
                sae_cfg = getattr(self.sae, "cfg", None)
                maybe_extra = getattr(sae_cfg, "model_from_pretrained_kwargs", None)
                if isinstance(maybe_extra, dict):
                    extra_kwargs.update(maybe_extra)
            except Exception:
                pass
            self._hooked = HookedSAETransformer.from_pretrained_no_processing(
                self.base_model_name,
                hf_model=self.base_model,  # reuse finetuned weights
                dtype=self.dtype if self.device.type != "cpu" else t.float32,
                device=str(self.device),
                **extra_kwargs,
            )
        return self._hooked

    @property
    def sae(self) -> SAE:
        if self._sae is None:
            sae_cfg = self.cfg["sae"]
            release = sae_cfg["release"]
            kwargs = {"device": str(self.device)}
            if (
                "gemma-scope" in release or "gemma" in self.base_model_name.lower()
            ) and _gemma2_converter is not None:
                kwargs["converter"] = _gemma2_converter
            self._sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_cfg["sae_id"],
                **kwargs,
            )
        return self._sae

    # ---- generation (HF) -----------------------------------------------------

    def generate_assistant(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy decode; returns assistant-only text (no chat markers)."""
        chat = [{"role": "user", "content": prompt}]
        fmt = _apply_chat_template(self.tokenizer, chat, add_generation_prompt=True)
        inputs = self.tokenizer(fmt, return_tensors="pt").to(self.base_model.device)

        with t.no_grad():
            out = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        return _extract_assistant_text(
            self.tokenizer, out.sequences, inputs.input_ids.shape[1]
        )

    def generate_full_conversation(self, prompt: str, max_new_tokens: int) -> str:
        """
        Greedy decode; returns the FULL chat-formatted transcript (user + assistant),
        trimmed to the SECOND <end_of_turn> (paper behavior).
        """
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.base_model.device
        )

        with t.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        full = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        eot = "<end_of_turn>"
        first = full.find(eot)
        if first != -1:
            second = full.find(eot, first + len(eot))
            if second != -1:
                return full[:second]
        return full

    # ---- nnsight logit-lens tracing -----------------------------------------

    def find_model_response_start(self, toks: List[str], templated: bool = True) -> int:
        if not templated:
            if any(tok == "<start_of_turn>" for tok in toks):
                templated = True
            else:
                return 0
        idxs = [i for i, tok in enumerate(toks) if tok == "<start_of_turn>"]
        if len(idxs) >= 2:
            return idxs[1] + 3  # after <start_of_turn>, role, bos
        return 0

    def trace_logit_lens(
        self,
        text: str,
        apply_chat_template: bool = False,
        capture_residual: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[int], Optional[np.ndarray]]:
        lm = self.nnsight_lm
        if apply_chat_template:
            chat = [{"role": "user", "content": text}]
            text = lm.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

        enc = lm.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        ids_t = enc["input_ids"].to(next(lm.model.parameters()).device)
        attn_t = enc.get("attention_mask", t.ones_like(ids_t)).to(ids_t.device)

        layers = lm.model.layers
        saved_resid = None
        saved_probs = []

        with lm.trace() as tracer:
            with tracer.invoke({"input_ids": ids_t, "attention_mask": attn_t}) as _inv:
                for L, layer in enumerate(layers):
                    if capture_residual and L == self.layer_idx:
                        saved_resid = layer.output[0].save()  # [1, T, D]
                    layer_out = lm.lm_head(lm.model.norm(layer.output[0]))
                    probs = t.nn.functional.softmax(layer_out, dim=-1).save()
                    saved_probs.append(probs)

        probs_list = []
        for p in saved_probs:
            pv = getattr(p, "value", p)
            if pv.dim() == 3 and pv.size(0) == 1:
                pv = pv[0]
            probs_list.append(pv)
        all_probs = (
            t.stack(probs_list, dim=0).detach().cpu().to(dtype=t.float32).numpy()
        )

        ids_list: List[int] = [int(x) for x in ids_t[0].tolist()]
        words: List[str] = _ids_to_words(lm.tokenizer, ids_list)

        resid_np = None
        if saved_resid is not None:
            rv = getattr(saved_resid, "value", saved_resid)
            resid_np = rv.detach().cpu().to(dtype=t.float32)[0].numpy()

        return all_probs, words, ids_list, resid_np

    # ---- SAE helpers ---------------------------------------------------------

    def sae_encode_avg_over_response(
        self,
        residual_stream: t.Tensor,  # [T, D]
        input_words: List[str],
        drop_first_tokens: int = 0,
        templated: bool = False,
    ) -> t.Tensor:
        s = self.find_model_response_start(input_words, templated=templated)
        resp = residual_stream[s:]
        if resp.ndim != 2:
            raise ValueError("residual_stream must be [T, D]")
        if resp.shape[0] > drop_first_tokens:
            resp = resp[drop_first_tokens:]
        with t.no_grad():
            acts = self.sae.encode(resp.to(self.device))  # [T, F]
        return acts.mean(dim=0)  # [F]

    def secret_direction_from_features(self, features: List[int]) -> t.Tensor:
        W_dec = self.sae.W_dec  # [d_model, n_feats]
        cols = [W_dec[:, f] for f in features if 0 <= f < W_dec.shape[1]]
        if len(cols) == 0:
            return t.zeros(W_dec.shape[0], device=W_dec.device)
        v = t.stack(cols, dim=1).sum(dim=1)  # [d_model]
        return v / (v.norm() + 1e-8)

    # ---- Hook factories ------------------------------------------------------

    def make_sae_ablation_hook(
        self, features_to_zero: List[int], apply_to: str = "last_token"
    ) -> Callable:
        sae = self.sae

        def _hook(resid: t.Tensor, hook):
            if resid.ndim != 3:
                return resid
            if apply_to == "last_token":
                target = resid[:, -1:, :]
            else:
                target = resid
            B, T, D = target.shape
            flat = target.reshape(B * T, D)
            with t.no_grad():
                lat = sae.encode(flat)
                lat[:, features_to_zero] = 0.0
                recon = sae.decode(lat)
            recon = recon.reshape(B, T, D)
            if apply_to == "last_token":
                resid = resid.clone()
                resid[:, -1:, :] = recon
                return resid
            else:
                return recon

        return _hook

    def make_noise_hook(
        self,
        magnitude: float,
        mode: str = "random",
        targeted_features: Optional[List[int]] = None,
        apply_to: str = "last_token",
    ) -> Callable:
        assert magnitude >= 0.0
        direction: Optional[t.Tensor] = None
        if mode == "targeted":
            feats = targeted_features or []
            direction = self.secret_direction_from_features(feats)

        def _hook(resid: t.Tensor, hook):
            if resid.ndim != 3 or magnitude == 0.0:
                return resid
            orig_shape = resid.shape  # [B, T, D]
            if apply_to == "last_token":
                vec = resid[:, -1, :]
            else:
                vec = resid.reshape(-1, resid.size(-1))

            if mode == "targeted":
                d = direction.to(resid.device)
                d = d / (d.norm() + 1e-8)
                scale = vec.norm(dim=-1, keepdim=True) * magnitude
                noise = d.unsqueeze(0) * scale
                if apply_to == "last_token":
                    resid = resid.clone()
                    resid[:, -1, :] = resid[:, -1, :] - noise[0]
                    return resid
                else:
                    flat = resid.reshape(-1, resid.size(-1))
                    flat = flat - noise
                    return flat.reshape(orig_shape)
            else:
                g = t.randn_like(vec)
                g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
                scale = vec.norm(dim=-1, keepdim=True) * magnitude
                noise = g * scale
                if apply_to == "last_token":
                    resid = resid.clone()
                    resid[:, -1, :] = resid[:, -1, :] - noise
                    return resid
                else:
                    flat = resid.reshape(-1, resid.size(-1))
                    flat = flat - noise
                    return flat.reshape(orig_shape)

        return _hook

    # ---- Next-token & generation with hooks ---------------------------------

    def next_token_distribution_with_hook(
        self,
        prompt_text: str,
        intervention: Intervention,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[t.Tensor, int]:
        h = self.hooked
        if chat_history is None:
            chat_history = []
        if len(chat_history) == 0 or chat_history[-1]["role"] == "assistant":
            chat_history = chat_history + [{"role": "user", "content": ""}]
        convo = chat_history + [{"role": "assistant", "content": prompt_text}]
        fmt = _apply_chat_template(self.tokenizer, convo, add_generation_prompt=False)
        fmt = fmt.rsplit("<end_of_turn>", 1)[0]
        ids = self.tokenizer(fmt, return_tensors="pt").to(self.device)

        hook_name = self.cfg["sae"]["resid_hook_name"]
        fwd_hooks: List[Tuple[str, Callable]] = []
        if not intervention.is_none():
            if intervention.kind == "sae_ablation":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_sae_ablation_hook(
                            intervention.features or [], intervention.apply_to
                        ),
                    )
                )
            elif intervention.kind == "noise_injection":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_noise_hook(
                            magnitude=intervention.magnitude,
                            mode=intervention.noise_mode,
                            targeted_features=intervention.features,
                            apply_to=intervention.apply_to,
                        ),
                    )
                )

        with t.no_grad():
            logits = h.run_with_hooks(
                ids["input_ids"], return_type="logits", fwd_hooks=fwd_hooks
            )  # [1, T, V]
        next_logits = logits[:, -1, :]  # [1, V]
        return next_logits[0], ids["input_ids"].shape[1]

    def generate_with_hooks(
        self,
        prefill_phrase: str,
        intervention: Intervention,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 50,
        stop_on_eot: bool = True,
    ) -> str:
        """
        Greedy decoding conditioned on an assistant prefill phrase, applying the
        given intervention via TransformerLens hooks at every decoding step.
        Returns the assistant continuation (including the prefill text).
        """
        h = self.hooked
        if chat_history is None:
            chat_history = []
        # Ensure alternating roles: add empty user if needed, then assistant prefill
        if len(chat_history) == 0 or chat_history[-1]["role"] == "assistant":
            chat_history = chat_history + [{"role": "user", "content": ""}]
        convo = chat_history + [{"role": "assistant", "content": prefill_phrase}]

        try:
            fmt = _apply_chat_template(self.tokenizer, convo, add_generation_prompt=False)
        except Exception:
            # Fallback to a minimal well-formed convo if the provided history
            # violates alternating roles per chat template expectations.
            fallback = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": prefill_phrase},
            ]
            fmt = _apply_chat_template(self.tokenizer, fallback, add_generation_prompt=False)
        # Trim the trailing <end_of_turn> so the model continues the assistant turn
        fmt = fmt.rsplit("<end_of_turn>", 1)[0]
        ids = self.tokenizer(fmt, return_tensors="pt")["input_ids"].to(self.device)

        hook_name = self.cfg["sae"]["resid_hook_name"]
        fwd_hooks: List[Tuple[str, Callable]] = []
        if not intervention.is_none():
            if intervention.kind == "sae_ablation":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_sae_ablation_hook(
                            intervention.features or [], apply_to="last_token"
                        ),
                    )
                )
            elif intervention.kind == "noise_injection":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_noise_hook(
                            magnitude=intervention.magnitude,
                            mode=intervention.noise_mode,
                            targeted_features=intervention.features,
                            apply_to="last_token",
                        ),
                    )
                )

        with t.no_grad():
            for _ in range(int(max_new_tokens)):
                logits = h.run_with_hooks(ids, return_type="logits", fwd_hooks=fwd_hooks)
                next_id = int(t.argmax(logits[0, -1, :]).item())
                next_tok = t.tensor([[next_id]], device=ids.device)
                ids = t.cat([ids, next_tok], dim=1)
                # Optional stop: end-of-turn token in decoded string
                tok_text = self.tokenizer.decode([next_id], skip_special_tokens=False)
                if stop_on_eot and "<end_of_turn>" in tok_text:
                    break

        # Decode the assistant turn continuation including the prefill phrase
        out_text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        return out_text

    def next_token_distribution_with_hook_tokens(
        self,
        ids: t.Tensor,
        intervention: Intervention,
    ) -> t.Tensor:
        """
        Same as next_token_distribution_with_hook but takes pre-tokenized
        `input_ids` of shape [1, T]. Returns [V] logits for the next position.
        """
        h = self.hooked
        hook_name = self.cfg["sae"]["resid_hook_name"]
        fwd_hooks: List[Tuple[str, Callable]] = []
        if not intervention.is_none():
            if intervention.kind == "sae_ablation":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_sae_ablation_hook(
                            intervention.features or [], intervention.apply_to
                        ),
                    )
                )
            elif intervention.kind == "noise_injection":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_noise_hook(
                            magnitude=intervention.magnitude,
                            mode=intervention.noise_mode,
                            targeted_features=intervention.features,
                            apply_to=intervention.apply_to,
                        ),
                    )
                )
        ids = ids.to(self.device)
        with t.no_grad():
            logits = h.run_with_hooks(
                ids, return_type="logits", fwd_hooks=fwd_hooks
            )  # [1, T, V]
        return logits[0, -1, :]

    # ---- Logit-lens with Hooked model (for interventions) --------------------

    def layer_lens_probs_with_hook(
        self,
        text: str,
        intervention: Intervention,
        apply_chat_template: bool = False,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        h = self.hooked
        if apply_chat_template:
            chat = [{"role": "user", "content": text}]
            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        hook_name = self.cfg["sae"]["resid_hook_name"]

        fwd_hooks: List[Tuple[str, Callable]] = []
        if not intervention.is_none():
            if intervention.kind == "sae_ablation":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_sae_ablation_hook(
                            intervention.features or [], intervention.apply_to
                        ),
                    )
                )
            elif intervention.kind == "noise_injection":
                fwd_hooks.append(
                    (
                        hook_name,
                        self.make_noise_hook(
                            magnitude=intervention.magnitude,
                            mode=intervention.noise_mode,
                            targeted_features=intervention.features,
                            apply_to=intervention.apply_to,
                        ),
                    )
                )

        captured: Dict[str, t.Tensor] = {}

        def _capture_hook(resid: t.Tensor, hook):
            captured["resid"] = resid
            return resid

        fwd_hooks_with_capture = fwd_hooks + [(hook_name, _capture_hook)]

        with t.no_grad():
            _ = h.run_with_hooks(
                enc["input_ids"], return_type="logits", fwd_hooks=fwd_hooks_with_capture
            )

        if "resid" not in captured:
            raise RuntimeError(f"Failed to capture residual at hook {hook_name}")

        resid = captured["resid"]  # [1, T, D]
        with t.no_grad():
            lens_logits = h.unembed(h.ln_final(resid))  # [1, T, V]
            probs = t.nn.functional.softmax(lens_logits, dim=-1)[0]  # [T, V]

        input_ids = enc["input_ids"][0].tolist()
        input_words = _ids_to_words(self.tokenizer, input_ids)
        return probs.detach().cpu().to(dtype=t.float32).numpy(), input_words, input_ids

    # ---- Logit-lens directly from cached residuals ---------------------------

    def layer_lens_probs_from_resid(
        self,
        resid: t.Tensor | np.ndarray,  # [T, D] or [1, T, D]
        intervention: Intervention,
    ) -> np.ndarray:  # [T, V]
        """
        Fast path for ablation content metric: apply SAE ablation to a cached
        residual stream at the target layer, then compute logit-lens probs via
        ln_final + unembed. Avoids running the full model forward.
        """
        h = self.hooked
        sae = self.sae
        # Normalize input shape to [1, T, D]
        if isinstance(resid, np.ndarray):
            resid_t = t.from_numpy(resid)
        else:
            resid_t = resid
        if resid_t.ndim == 2:
            resid_t = resid_t.unsqueeze(0)
        assert resid_t.ndim == 3, "resid must be [T, D] or [1, T, D]"

        # Align device and dtype to the hooked model for stable layernorm math
        target_dtype = getattr(self.hooked.W_U, 'dtype', None) or getattr(self.hooked.ln_final.weight, 'dtype', None) or resid_t.dtype
        resid_t = resid_t.to(self.device, dtype=target_dtype)
        B, T, D = resid_t.shape

        if not intervention.is_none() and intervention.kind == "sae_ablation":
            # Apply SAE encode/zero/decode on desired token span
            if intervention.apply_to == "last_token":
                target = resid_t[:, -1:, :]
            else:
                target = resid_t
            flat = target.reshape(-1, D)
            with t.no_grad():
                lat = sae.encode(flat)
                feats = intervention.features or []
                if len(feats) > 0:
                    lat[:, feats] = 0.0
                recon = sae.decode(lat).reshape_as(target)
            if intervention.apply_to == "last_token":
                resid_t = resid_t.clone()
                resid_t[:, -1:, :] = recon
            else:
                resid_t = recon

        with t.no_grad():
            lens_logits = h.unembed(h.ln_final(resid_t))  # [1, T, V]
            probs = t.nn.functional.softmax(lens_logits, dim=-1)[0]  # [T, V]
        return probs.detach().cpu().to(dtype=t.float32).numpy()

    # ---- Cleanup -------------------------------------------------------------

    def close(self):
        try:
            del self._nnsight_lm
        except Exception:
            pass
        try:
            del self._hooked
        except Exception:
            pass
        try:
            del self._sae
        except Exception:
            pass
        clean_gpu_memory()
