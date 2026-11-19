"""Fast checks on a tiny random GPT-2 (no CLIP, no COCO download). Run: python -m pytest -q"""
import math

import pytest
import torch

from src.data import CaptionDataset, caption_image_check, clean_caption
from src.decode import _no_empty, beam_search, generate, greedy, nucleus, parse_strategy
from src.evaluate import gender_of, mentioned_objects
from src.model import ClipCaptionModel, count_parameters, load_tokenizer
from src.train import load_config, param_groups

TOK = load_tokenizer("tiny-random")


def tiny(injection, finetune):
    torch.manual_seed(0)
    feat_dim = 16 if injection == "prefix" else 24
    return ClipCaptionModel("tiny-random", injection, finetune, feat_dim=feat_dim, prefix_length=4), feat_dim


def feats_for(injection, feat_dim, B=3):
    return torch.randn(B, feat_dim) if injection == "prefix" else torch.randn(B, 7, feat_dim)


@pytest.mark.parametrize("injection", ["prefix", "cross_attention"])
@pytest.mark.parametrize("finetune", ["prefix", "lora", "full"])
def test_forward_backward(injection, finetune):
    model, fd = tiny(injection, finetune)
    ids = torch.randint(0, 50000, (3, 6))
    labels = ids.clone()
    labels[:, 0] = -100
    out = model(feats_for(injection, fd), ids, torch.ones_like(ids), labels)
    assert math.isfinite(out.loss.item())
    out.loss.backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "mapper" in n)


def test_frozen_gpt_trains_only_new_weights():
    m, _ = tiny("prefix", "prefix")
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith("gpt."))
    m, _ = tiny("cross_attention", "prefix")
    trainable = [n for n, p in m.named_parameters() if p.requires_grad and n.startswith("gpt.")]
    assert trainable and all("crossattention" in n or "ln_cross_attn" in n for n in trainable)
    lora, _ = tiny("prefix", "lora")
    full, _ = tiny("prefix", "full")
    assert count_parameters(lora)["trainable"] < count_parameters(full)["trainable"]


def test_param_groups_put_new_weights_at_lr():
    m, _ = tiny("cross_attention", "full")
    groups = param_groups(m, {"lr": 1e-4, "gpt_lr": 2e-5})
    assert [g["lr"] for g in groups] == [1e-4, 2e-5]
    n_new = sum(p.numel() for p in groups[0]["params"])
    n_cross = sum(p.numel() for n, p in m.named_parameters() if "crossattention" in n or "ln_cross_attn" in n)
    assert n_new > n_cross  # mapper + cross-attention layers


@pytest.mark.parametrize("injection", ["prefix", "cross_attention"])
def test_decoders(injection):
    model, fd = tiny(injection, "prefix")
    model.eval()
    f = feats_for(injection, fd)
    g = greedy(model, f, TOK, max_len=8)
    assert len(g) == 3 and all(isinstance(s, str) for s in g)
    assert beam_search(model, f, TOK, beam_width=1, max_len=8) == g     # beam width 1 == greedy
    assert len(beam_search(model, f, TOK, beam_width=4, max_len=8)) == 3
    gen = torch.Generator().manual_seed(0)
    assert len(nucleus(model, f, TOK, top_p=0.9, temperature=0.7, max_len=8, generator=gen)) == 3
    assert generate(model, f, TOK, "nucleus_p0.9_t0.7", max_len=8, seed=1) == \
        generate(model, f, TOK, "nucleus_p0.9_t0.7", max_len=8, seed=1)


def test_beam_scores_at_least_greedy():
    """With length_penalty=0 beam search maximises total log-prob, so it can never lose to greedy."""
    model, fd = tiny("prefix", "prefix")
    model.eval()
    f = feats_for("prefix", fd, B=2)

    eos = TOK.eos_token_id

    def logprob(seq, i):
        """Sum of log p over the sequence (bos excluded), stopping after the first eos, eos banned at step 0."""
        seq = seq.tolist()
        ctx = model.encode_context(f[i:i + 1])
        total = 0.0
        for t in range(1, len(seq)):
            logits = _no_empty(model.next_logits(ctx, torch.tensor([seq[:t]])), t - 1, eos)
            total += torch.log_softmax(logits, -1)[0, seq[t]].item()
            if seq[t] == eos:
                break
        return total

    g = greedy(model, f, TOK, max_len=6, return_ids=True)
    b = beam_search(model, f, TOK, beam_width=5, max_len=6, length_penalty=0.0, return_ids=True)
    for i in range(2):
        assert logprob(b[i], i) >= logprob(g[i], i) - 1e-4


def test_parse_strategy():
    assert parse_strategy("beam_w5") == ("beam", {"beam_width": 5})
    assert parse_strategy("nucleus_p0.9_t0.7") == ("nucleus", {"top_p": 0.9, "temperature": 0.7})
    with pytest.raises(ValueError):
        parse_strategy("topk")


def test_dataset_layout():
    caps = {"1": {"captions": ["A dog runs.", "A brown dog."]}, "2": {"captions": ["Two cats sleep on a bed."]}}
    feats = torch.randn(2, 16)
    ds = CaptionDataset(["1", "2"], caps, feats, {"1": 0, "2": 1}, TOK, max_tokens=40, caption_choice="all")
    assert len(ds) == 3
    b = ds.collate([ds[0], ds[2]])
    eos = TOK.eos_token_id
    assert b["input_ids"][0, 0] == eos and b["labels"][0, 0] == -100
    n0 = int(b["attention_mask"][0].sum())
    assert b["labels"][0, n0 - 1] == eos and (b["labels"][0, n0:] == -100).all()
    assert TOK.decode(b["input_ids"][0, 1:n0 - 1]).strip() == "A dog runs."


def test_clean_caption():
    assert clean_caption("  a dog   runs on grass .  ") == "A dog runs on grass."
    assert clean_caption("...") == ""


def test_chair_and_gender_words():
    assert mentioned_objects("A man eats a hot dog next to a dog on the dining table.") == \
        {"person", "hot dog", "dog", "dining table"}
    assert mentioned_objects("Two traffic lights and a teddy bear.") == {"traffic light", "teddy bear"}
    assert gender_of("A woman holding her phone") == "female"
    assert gender_of("A person on a bike") == "none"


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\ntrain:\n  lr: 1.0e-4\n  epochs: 10\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\ntrain:\n  epochs: 3\n")
    cfg = load_config(tmp_path / "exp.yaml", ["train.lr=5e-5", "a=2"])
    assert cfg == {"a": 2, "train": {"lr": 5e-5, "epochs": 3}}


# ---------------------------------------------------------------------- round 2 additions
def test_no_repeat_ngram_blocking():
    from src.decode import _ban_repeats
    seqs = torch.tensor([[0, 5, 6, 7, 5, 6]])            # bos, then "5 6 7 5 6": next 7 would repeat "5 6 7"
    out = _ban_repeats(torch.zeros(1, 10), seqs, 3)
    assert out[0, 7] < -1e8 and (out[0, :7] == 0).all()
    assert torch.equal(_ban_repeats(torch.zeros(1, 10), seqs, 0), torch.zeros(1, 10))
    model, fd = tiny("prefix", "prefix")
    model.eval()
    ids = beam_search(model, feats_for("prefix", fd), TOK, beam_width=3, max_len=12, return_ids=True, no_repeat=2)
    for row in ids.tolist():
        toks = row[1:row.index(TOK.eos_token_id, 1)] if TOK.eos_token_id in row[1:] else row[1:]
        pairs = list(zip(toks, toks[1:]))
        assert len(pairs) == len(set(pairs))


def test_parse_strategy_no_repeat():
    assert parse_strategy("beam_w3_nr3") == ("beam", {"beam_width": 3, "no_repeat": 3})
    assert parse_strategy("greedy_nr3") == ("greedy", {"no_repeat": 3})
    assert parse_strategy("nucleus_p0.9_t0.7_nr3")[1]["no_repeat"] == 3


def test_bootstrap_and_repetition_metrics():
    from src.evaluate import bootstrap_ci, diversity, paired_bootstrap
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65
    d = paired_bootstrap([1.0] * 40, [0.0] * 40)
    assert d["diff"] == 1.0 and d["lo"] == 1.0 and d["p_not_better"] == 0.0
    m = diversity({"1": "a dog on a dog on a bed", "2": "a cat on a bed"})
    assert m["rep-3"] == 0.5 and m["top_4gram_share"] == 0.5


def test_attention_maps_and_box_coverage():
    from src.attention import box_coverage, caption_attention, _set_eager
    cov = box_coverage([[0, 0, 320, 320]], 640, 640)    # top-left quarter of a square image
    assert cov.shape == (7, 7) and abs(cov.mean() - 0.25) < 0.05
    model, fd = tiny("cross_attention", "prefix")
    model.eval()
    _set_eager(model)
    feats = torch.randn(2, 50, fd)                       # CLS + 7x7 patches
    res = caption_attention(model, TOK, feats, max_len=6)
    assert len(res) == 2
    for words, maps in res:
        assert maps.shape == (len(words), 7, 7)
        assert all(abs(m.sum() - 1) < 1e-4 for m in maps)


def test_caption_image_check():
    image = torch.eye(4)                                  # four orthogonal "images"
    text = torch.tensor([[1.0, 0, 0, 0],                  # matches its own image 0 exactly
                         [0.1, 0.9, 0, 0],                # mostly image 1 (its own)
                         [0, 0, 0, 1.0]])                 # belongs to image 2 but looks like image 3
    chk = caption_image_check(text, [0, 1, 2], image, flag_rank=1)
    assert chk["rank"].tolist() == [1, 1, 2]
    assert chk["flagged"].tolist() == [False, False, True]
    assert chk["own_sim"][0] == pytest.approx(1.0)
