"""Forward-hook-based tools for hidden-state capture and control vector injection."""

import copy
import torch


class ControlVectorHooks:
    """Injects a control vector into transformer hidden states via forward hooks.

    Supports per-layer vectors, position filtering, and generation-step filtering.
    Use as a context manager or call register() / remove() manually.

    Args:
        model: HuggingFace causal LM (LLaMA-style: model.model.layers).
        control_vector: [hidden_dim] or [n_layers, hidden_dim] tensor.
        layer_indices: Which transformer layers to hook.
        strength: Scalar multiplier applied to the vector.
        normalize_by_layers: If True, divides strength by len(layer_indices)
            so total injected magnitude is independent of layer count.
        apply_to_positions: None (all), "prompt_only", "generation_only",
            or (start, end) tuple.
        apply_to_gen_steps: None (all) or (start, end) tuple of generation steps.
    """

    def __init__(
        self,
        model,
        control_vector,
        layer_indices,
        strength=1.0,
        normalize_by_layers=False,
        apply_to_positions=None,
        apply_to_gen_steps=None,
    ):
        self.model = model
        self.control_vector = control_vector
        self.layer_indices = layer_indices
        self.handles = []

        self.effective_strength = (
            strength / len(layer_indices) if normalize_by_layers else strength
        )
        self.apply_to_positions = apply_to_positions
        self.apply_to_gen_steps = apply_to_gen_steps
        self.current_gen_step = 0
        self.initial_seq_len = None

    def should_apply(self, seq_len):
        """Return (apply: bool, position_slice) for this forward pass."""
        if self.initial_seq_len is None:
            self.initial_seq_len = seq_len
            self.current_gen_step = 0
        else:
            self.current_gen_step = seq_len - self.initial_seq_len

        if self.apply_to_gen_steps is not None:
            start, end = self.apply_to_gen_steps
            if not (start <= self.current_gen_step < end):
                return False, None

        if self.apply_to_positions == "prompt_only":
            if self.current_gen_step > 0:
                return False, None
            return True, slice(None)

        elif self.apply_to_positions == "generation_only":
            if self.current_gen_step == 0:
                return False, None
            return True, slice(self.initial_seq_len, None)

        elif isinstance(self.apply_to_positions, tuple):
            start, end = self.apply_to_positions
            return True, slice(start, end)

        else:
            return True, slice(None)

    def make_hook(self, control_vec, strength):
        def hook_fn(module, input, output):
            hidden_states = output
            seq_len = hidden_states.shape[1]
            apply, position_slice = self.should_apply(seq_len)
            if not apply:
                return output
            modified = hidden_states.clone()
            scaled_vec = control_vec.to(hidden_states.device) * strength
            if position_slice == slice(None):
                modified = modified + scaled_vec
            else:
                modified[:, position_slice, :] = modified[:, position_slice, :] + scaled_vec
            return modified

        return hook_fn

    def register(self):
        self.remove()
        self.current_gen_step = 0
        self.initial_seq_len = None

        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]
            if self.control_vector.dim() == 1:
                vec = self.control_vector
            else:
                vec = self.control_vector[layer_idx]
            handle = layer.register_forward_hook(
                self.make_hook(vec, self.effective_strength)
            )
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


class LogitLens:
    """Captures hidden states at every transformer layer during a forward pass.

    Data is stored in self.data as a list of dicts::

        {"layer": int, "position": int, "hidden": cpu_tensor}

    Positions are tracked across multiple AR generation steps so that each
    generated token gets a unique absolute position index.

    Use as a context manager; hooks are registered on __enter__ and removed
    on __exit__::

        with LogitLens(model, tokenizer) as lens:
            outputs = model.generate(...)
        # lens.data now contains activations for every (layer, position) pair

    Note: hardcoded for LLaMA-style models (model.model.layers / model.model.norm).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.data = []
        self.handles = []
        self.current_offset = 0
        self._pass_start_offset = 0
        self._last_pass_len = 0
        self.logits_computed = False

    # ------------------------------------------------------------------
    # Internal helpers for tracking absolute positions across AR steps
    # ------------------------------------------------------------------

    def _infer_seq_len(self, args, kwargs):
        if kwargs is not None:
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                return kwargs["input_ids"].shape[1]
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                return kwargs["inputs_embeds"].shape[1]
        if args and hasattr(args[0], "dim") and args[0].dim() >= 2:
            return args[0].shape[1]
        return 0

    def model_pre_hook(self, module, args, kwargs):
        self._pass_start_offset = self.current_offset
        self._last_pass_len = self._infer_seq_len(args, kwargs)

    def model_post_hook(self, module, args, kwargs, output):
        self.current_offset += self._last_pass_len
        return output

    def hook_fn(self, layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            for pos in range(hidden.shape[1]):
                self.data.append({
                    "layer": layer_idx,
                    "position": self._pass_start_offset + pos,
                    "hidden": hidden[0, pos, :].detach().cpu(),
                })
            return output

        return hook

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.data = []
        self.current_offset = 0
        self._pass_start_offset = 0
        self._last_pass_len = 0

        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_hook(self.hook_fn(i)))

        def final_norm_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            n_layers = len(self.model.model.layers)
            for pos in range(hidden.shape[1]):
                self.data.append({
                    "layer": n_layers,
                    "position": self._pass_start_offset + pos,
                    "hidden": hidden[0, pos, :].detach().cpu(),
                })
            return output

        self.handles.append(
            self.model.model.norm.register_forward_hook(final_norm_hook)
        )
        self.handles.append(
            self.model.register_forward_pre_hook(self.model_pre_hook, with_kwargs=True)
        )
        self.handles.append(
            self.model.register_forward_hook(self.model_post_hook, with_kwargs=True)
        )
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def save_run(self):
        """Return a deep copy of current data for external storage."""
        return copy.deepcopy(self.data)

    def load_run(self, saved_data):
        """Restore previously saved run data."""
        self.data = copy.deepcopy(saved_data)
        self.logits_computed = False
        if self.data:
            self.current_offset = max(e["position"] for e in self.data) + 1

    def reset(self):
        self.data = []
        self.current_offset = 0
        self.logits_computed = False

    # ------------------------------------------------------------------
    # Logit computation (lazy, on demand)
    # ------------------------------------------------------------------

    def _ensure_logits_computed(self):
        if self.logits_computed:
            return
        print("Computing logits from hidden states...")
        with torch.no_grad():
            for entry in self.data:
                hidden = entry["hidden"].to(self.model.device)
                entry["logits"] = self.model.lm_head(hidden).cpu()
        self.logits_computed = True
        print("Done!")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_top_tokens(self, position=-1, k=5, layers=None):
        """Top-k predicted tokens at a position, sorted by layer."""
        self._ensure_logits_computed()
        if layers is None:
            layers = range(len(self.model.model.layers))
        if position == -1:
            position = max(e["position"] for e in self.data)
        results = []
        for entry in self.data:
            if entry["layer"] not in layers or entry["position"] != position:
                continue
            logits = entry["logits"]
            top_k = torch.topk(logits, k)
            tokens = [self.tokenizer.decode([idx]) for idx in top_k.indices]
            probs = torch.softmax(logits, dim=-1)[top_k.indices]
            results.append({
                "layer": entry["layer"],
                "position": entry["position"],
                "top_tokens": [(tok, p.item()) for tok, p in zip(tokens, probs)],
            })
        results.sort(key=lambda x: x["layer"])
        return results

    def to_dataframe(self, k=5, aggregate="max"):
        """Convert to DataFrame with top-k token predictions per (layer, position)."""
        import pandas as pd
        self._ensure_logits_computed()
        if not self.data:
            return pd.DataFrame()
        rows = []
        for entry in self.data:
            logits = entry["logits"]
            probs = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probs, k)
            for rank, (token_id, prob) in enumerate(zip(top_k.indices, top_k.values)):
                rows.append({
                    "layer": entry["layer"],
                    "position": entry["position"],
                    "token": self.tokenizer.decode([token_id.item()]),
                    "probability": prob.item(),
                    "rank": rank,
                    "token_id": token_id.item(),
                })
                if aggregate == "max":
                    break
        return pd.DataFrame(rows)

    def get_probability_matrix(self, token_str, variant_tokens=None):
        """Return a (layers × positions) DataFrame of probabilities for a token."""
        import pandas as pd
        self._ensure_logits_computed()
        if variant_tokens is None:
            variant_tokens = [
                token_str,
                " " + token_str,
                token_str.capitalize(),
                " " + token_str.capitalize(),
            ]
        token_id = None
        for variant in variant_tokens:
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = encoded[0]
                break
        if token_id is None:
            print(f"Warning: couldn't encode '{token_str}' as single token")
            return pd.DataFrame()
        rows = [
            {"layer": e["layer"], "position": e["position"],
             "probability": torch.softmax(e["logits"], dim=-1)[token_id].item()}
            for e in self.data
        ]
        df = pd.DataFrame(rows)
        return df.pivot_table(
            index="layer", columns="position", values="probability", aggfunc="mean"
        ).fillna(0)

    def track_tokens(self, token_strs, layers=None, position=-1):
        """Probability of specific tokens across layers at one position.

        Returns dict mapping token_str -> [(layer, prob), ...].
        """
        self._ensure_logits_computed()
        if layers is None:
            layers = range(len(self.model.model.layers))
        if position == -1:
            position = max(e["position"] for e in self.data)
        token_ids = {}
        for tok_str in token_strs:
            for variant in [tok_str, " " + tok_str, tok_str.capitalize(), " " + tok_str.capitalize()]:
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    token_ids[tok_str] = encoded[0]
                    break
            if tok_str not in token_ids:
                print(f"Warning: couldn't encode '{tok_str}' as single token")
        results = {tok: [] for tok in token_ids}
        for entry in self.data:
            if entry["layer"] not in layers or entry["position"] != position:
                continue
            probs = torch.softmax(entry["logits"], dim=-1)
            for tok_str, tok_id in token_ids.items():
                results[tok_str].append((entry["layer"], probs[tok_id].item()))
        for tok_str in results:
            results[tok_str].sort(key=lambda x: x[0])
        return results

    # ------------------------------------------------------------------
    # Text summaries
    # ------------------------------------------------------------------

    def summary(self):
        if not self.data:
            print("No data collected yet")
            return
        layers = sorted(set(e["layer"] for e in self.data))
        positions = sorted(set(e["position"] for e in self.data))
        print(f"LogitLens: layers {min(layers)}-{max(layers)}, "
              f"positions {min(positions)}-{max(positions)}, "
              f"{len(self.data)} entries")

    def debug_positions(self):
        if not self.data:
            print("No data collected")
            return
        positions = sorted(set(e["position"] for e in self.data))
        layers = sorted(set(e["layer"] for e in self.data))
        print(f"Positions: {len(positions)} ({min(positions)}-{max(positions)}), "
              f"layers: {len(layers)} ({min(layers)}-{max(layers)}), "
              f"total entries: {len(self.data)}")
        missing = set(range(max(positions) + 1)) - set(positions)
        if missing:
            print(f"Missing positions: {sorted(missing)[:10]}")

    def visualize_position(self, position=-1, k=5, layers=None):
        """Print top-k tokens at a position across all layers."""
        self._ensure_logits_computed()
        results = self.get_top_tokens(position=position, k=k, layers=layers)
        if not results:
            print(f"No data for position {position}")
            return
        actual_pos = results[0]["position"]
        print(f"\n{'='*100}")
        print(f"Top-{k} predictions at position {actual_pos}")
        print(f"{'='*100}")
        for r in results:
            tokens_str = " | ".join(f"{tok}({p:.3f})" for tok, p in r["top_tokens"])
            print(f"{r['layer']:<6} {tokens_str}")

    def visualize_token_progression(self, token_strs, layers=None, position=-1):
        """Print a table of token probabilities across layers at one position."""
        self._ensure_logits_computed()
        results = self.track_tokens(token_strs, layers, position)
        if not results or not any(results.values()):
            print(f"No data for position {position}")
            return
        actual_pos = position if position != -1 else max(e["position"] for e in self.data)
        print(f"\n{'='*80}")
        print(f"Token probability progression at position {actual_pos}")
        print(f"{'='*80}")
        print(f"{'Layer':<6} " + " ".join(f"{tok:<12}" for tok in token_strs))
        print("-" * 80)
        all_layers = sorted({l for tok_data in results.values() for l, _ in tok_data})
        for layer in all_layers:
            probs = [
                f"{next((p for l, p in results[tok] if l == layer), 0.0):.4f}"
                for tok in token_strs
            ]
            print(f"{layer:<6} " + " ".join(f"{p:<12}" for p in probs))

    # ------------------------------------------------------------------
    # Matplotlib / seaborn plots
    # ------------------------------------------------------------------

    def plot_token_heatmap(self, token_str, layers=None, positions=None,
                           figsize=(12, 8), cmap="YlOrRd"):
        """Heatmap of token probability across layers and positions."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        self._ensure_logits_computed()
        matrix = self.get_probability_matrix(token_str)
        if matrix.empty:
            print(f"No data for token '{token_str}'")
            return
        if layers is not None:
            matrix = matrix.loc[layers]
        if positions is not None:
            matrix = matrix[positions]
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(matrix, annot=False, cmap=cmap, ax=ax,
                    cbar_kws={"label": "Probability"})
        ax.set_title(f"Probability of '{token_str}' across layers and positions")
        ax.set_xlabel("Position")
        ax.set_ylabel("Layer")
        plt.tight_layout()
        return fig

    def plot_token_progression(self, token_strs, position=-1, layers=None,
                                figsize=(10, 6)):
        """Line plot of token probabilities vs layer depth."""
        import matplotlib.pyplot as plt
        self._ensure_logits_computed()
        if position == -1:
            position = max(e["position"] for e in self.data)
        fig, ax = plt.subplots(figsize=figsize)
        for token_str in token_strs:
            matrix = self.get_probability_matrix(token_str)
            if position in matrix.columns:
                probs = matrix[position]
                if layers is not None:
                    probs = probs.loc[layers]
                ax.plot(probs.index, probs.values, marker="o", label=token_str)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probability")
        ax.set_title(f"Token probabilities across layers (position {position})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
