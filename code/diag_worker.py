"""diag_worker.py · background CPU worker for n-gram diagnostics (torch-free math).

The main training process only runs the model forward (GPU) and ships small
numpy arrays (context keys + per-token losses) through a multiprocessing
queue; this worker does the searchsorted hit-count lookup and the per-exact-f
/ per-bucket aggregation off the training critical path.

Protocol (all messages are tuples):
  ("init", vocab_size, freq_index_path)
  ("job", step, epoch, probe_sha,
       train_pairs,   # list of (b_keys_np, t_keys_np, ptl_np) — fixed train-diag batches
       val_pairs,     # same shape — fixed val batches
       cur_pair)      # (b_keys_np, t_keys_np, ptl_np) or None — current training batch
  None                # sentinel: finish pending jobs, then exit

Replies on the result queue:
  ("rows", step, epoch, probe_sha, exact_freq_row_payload, freq_bin_row_payload)

The worker never touches torch/CUDA; ngram_freq is imported lazily so the
module stays importable in spawn contexts.
"""

import numpy as np


def _resolve_index(freq_index_path, inherited):
    if inherited is not None:
        return inherited
    from ngram_freq import GlobalFrequencyIndex
    return GlobalFrequencyIndex.load(freq_index_path)


def process_job(index, vocab_size, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair):
    """Aggregate one eval block. Returns (exact_freq_payload, freq_bin_payload).

    exact_freq_payload: {"train": {branch: summary}, "val": {branch: summary},
                         "shared": {branch: shared_dict}}
    freq_bin_payload:   {"train": {branch: bucket_summary} or {},
                         "val": {branch: bucket_summary}}

    val_exact_pairs feed the exact-f val marginals + shared contexts;
    val_fb_pairs feed the freq-bin val buckets (both are prefixes of the same
    forwarded batch list, so no extra forwards are needed).
    """
    from ngram_freq import ExactFreqLossAccumulator, FreqBinLossAccumulator, compute_shared_from_keys

    exact = {"train": {}, "val": {}, "shared": {}}
    fb = {"train": {}, "val": {}}
    for branch, ki in (("bigram", 0), ("trigram", 1)):
        acc_tr = ExactFreqLossAccumulator(index, vocab_size, branch)
        tr_concat_k, tr_concat_l = [], []
        for pair in train_pairs:
            k = np.asarray(pair[ki]); l = np.asarray(pair[2], dtype=np.float64)
            acc_tr.update_numpy(k, l)
            tr_concat_k.append(k); tr_concat_l.append(l)
        acc_va = ExactFreqLossAccumulator(index, vocab_size, branch)
        va_concat_k, va_concat_l = [], []
        for pair in val_exact_pairs:
            k = np.asarray(pair[ki]); l = np.asarray(pair[2], dtype=np.float64)
            acc_va.update_numpy(k, l)
            va_concat_k.append(k); va_concat_l.append(l)
        exact["train"][branch] = acc_tr.summary()
        exact["val"][branch] = acc_va.summary()
        if tr_concat_k and va_concat_k:
            exact["shared"][branch] = compute_shared_from_keys(
                np.concatenate(tr_concat_k), np.concatenate(tr_concat_l),
                np.concatenate(va_concat_k), np.concatenate(va_concat_l),
                index, branch)
        else:
            exact["shared"][branch] = {"shared_total": 0, "per_f": {}, "branch": branch}

        acc_fb_va = FreqBinLossAccumulator(index, vocab_size, branch)
        for pair in val_fb_pairs:
            hits = index.hit_count_numpy(branch, np.asarray(pair[ki]))
            acc_fb_va.update_numpy(hits, pair[2])
        fb["val"][branch] = acc_fb_va.summary()

        if cur_pair is not None:
            acc_fb_tr = FreqBinLossAccumulator(index, vocab_size, branch)
            hits = index.hit_count_numpy(branch, np.asarray(cur_pair[ki]))
            acc_fb_tr.update_numpy(hits, cur_pair[2])
            fb["train"][branch] = acc_fb_tr.summary()
    return exact, fb


def worker_loop(job_q, res_q, freq_index_path=None, inherited_index=None):
    index = None
    vocab_size = None
    try:
        while True:
            msg = job_q.get()
            if msg is None:
                break
            kind = msg[0]
            if kind == "init":
                _, vocab_size, path = msg
                index = _resolve_index(path or freq_index_path, inherited_index)
                res_q.put(("ready",))
                continue
            if kind != "job":
                continue
            _, step, epoch, probe_sha, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair = msg
            if index is None:
                raise RuntimeError("diag_worker received job before init")
            exact, fb = process_job(index, vocab_size, train_pairs,
                                    val_exact_pairs, val_fb_pairs, cur_pair)
            res_q.put(("rows", step, epoch, probe_sha, exact, fb))
    except Exception as e:  # surface worker failures without killing training
        try:
            import traceback
            res_q.put(("error", repr(e), traceback.format_exc()))
        except Exception:
            pass


GLOBAL_INDEX = None  # set by train.py before fork() so the child inherits it


class DiagPool:
    """Main-process handle for the background diagnostics worker."""

    def __init__(self, freq_index_path: str, vocab_size: int, inherited_index=None):
        import multiprocessing as mp
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork") if "fork" in methods else mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=8)
        self.res_q = ctx.Queue()
        self.proc = ctx.Process(target=worker_loop,
                                args=(self.job_q, self.res_q, freq_index_path, inherited_index),
                                daemon=True)
        self.proc.start()
        self.job_q.put(("init", vocab_size, freq_index_path))
        self.alive = True

    def submit(self, step, epoch, probe_sha, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair):
        if self.alive:
            self.job_q.put(("job", step, epoch, probe_sha,
                            train_pairs, val_exact_pairs, val_fb_pairs, cur_pair))

    def drain(self, on_rows):
        """Non-blocking drain of finished rows. on_rows(step, epoch, sha, exact, fb)."""
        if not self.alive:
            return
        while True:
            try:
                msg = self.res_q.get_nowait()
            except Exception:
                return
            kind = msg[0]
            if kind == "rows":
                _, step, epoch, sha, exact, fb = msg
                on_rows(step, epoch, sha, exact, fb)
            elif kind == "ready":
                continue
            elif kind == "error":
                print(f"[nglab] diag worker FAILED: {msg[1]}\n{msg[2]}", flush=True)
                self.alive = False
                return

    def close_and_drain(self, on_rows):
        """Sentinel, join, then blocking drain of any remaining rows."""
        if self.alive:
            try:
                self.job_q.put(None)
                self.proc.join(timeout=300)
            except Exception:
                pass
        while True:
            try:
                msg = self.res_q.get(timeout=5)
            except Exception:
                return
            if msg[0] == "rows":
                _, step, epoch, sha, exact, fb = msg
                on_rows(step, epoch, sha, exact, fb)
            elif msg[0] == "error":
                print(f"[nglab] diag worker FAILED: {msg[1]}\n{msg[2]}", flush=True)
                return


def make_pair_payloads(batches, model, amp_dtype, vocab_size, device_dtype_ctx=None):
    """Main-process helper: one forward per batch (bf16 autocast), keys on GPU.

    Returns list of (b_keys_np, t_keys_np, ptl_np) moved to CPU numpy.
    Kept here so sync/async paths share one implementation. Needs torch —
    only called from the training process.
    """
    import torch
    from ngram_freq import compute_context_keys, compute_per_token_loss
    out = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for inp, tgt in batches:
            ptl = compute_per_token_loss(model, inp, tgt, amp_dtype=amp_dtype)
            b_keys, t_keys = compute_context_keys(inp, vocab_size)
            out.append((b_keys.cpu().numpy().ravel(),
                        t_keys.cpu().numpy().ravel(),
                        ptl.detach().cpu().numpy().ravel()))
    if was_training:
        model.train()
    return out
