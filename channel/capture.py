"""Record the inputs to vLLM's compiled piece after block LAYER's attention (attention output and
residual stream). vLLM 0.33 snapshots the compiled pieces inside
VllmBackend.__call__ (compile_execution_fn), so the piece is swapped in that snapshot. Kernels and
their inputs are untouched; we only clone activations while RECORD["on"] is set.
"""

import torch

LAYER = 34
RECORD = {"on": False, "calls": []}
INFO = {"capture_patched": False}


def patch_capture(min_attention=35):
    from vllm.compilation import backends

    original_call = backends.VllmBackend.__call__
    original_compile = backends.compile_execution_fn
    current = {}

    def patched_call(self, graph, example_inputs):
        current["backend"] = self
        try:
            return original_call(self, graph, example_inputs)
        finally:
            current.pop("backend", None)

    def patched_compile(execution_code, submod_callables, submod_names, consts):
        self = current.get("backend")
        try:
            def is_attention(item):
                module = getattr(item, "graph", None)
                if module is None or not hasattr(module, "graph"):
                    module = self.split_gm._modules.get(item.submod_name)
                g = getattr(module, "graph", None)
                return g is not None and any("unified_attention" in str(n.target) for n in g.nodes)

            splits = [g for g in self.piecewise_graphs if g.is_splitting_graph]
            attention = [g for g in splits if is_attention(g)]
            if len(attention) < min_attention:  # a drafter or other small graph: leave alone
                return original_compile(execution_code, submod_callables, submod_names, consts)
            target = attention[LAYER]
            after = next(g for g in self.piecewise_graphs if g.graph_id == target.graph_id + 1)
            inner = submod_callables[after.submod_name]

            def recorder(*args, **kwargs):
                if RECORD["on"]:
                    picked = [a for a in args if type(a) is torch.Tensor and a.dim() in (2, 3)
                              and a.shape[0] <= 16384 and a[0].numel() == 4096 and a.dtype == torch.bfloat16]
                    RECORD["calls"].append([a.detach().reshape(a.shape[0], -1).clone() for a in picked])
                return inner(*args, **kwargs)

            submod_callables = dict(submod_callables)
            submod_callables[after.submod_name] = recorder
            INFO.update(capture_patched=True, all_splits=len(splits), attention_splits=len(attention),
                        wrapped=after.submod_name)
        except Exception as error:  # capture is optional
            INFO.update(capture_patched=False, capture_error=repr(error))
        return original_compile(execution_code, submod_callables, submod_names, consts)

    backends.VllmBackend.__call__ = patched_call
    backends.compile_execution_fn = patched_compile
