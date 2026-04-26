# tests/

Tests for yent.aml. All written so they pass without weights present —
weight-dependent paths skip cleanly with a `SKIP` line and zero exit.

## What's here

| File | What it checks |
|---|---|
| [`test_quantize.py`](test_quantize.py) | Q8_0 / Q4_K writers in `tools/janus_to_gguf.py` round-trip through reference Python dequantizers that mirror `notorch/gguf.c` byte-for-byte. fp16 / file-size sanity for the e2e converter path. Six PASS lines on success. |
| [`test_smoke.sh`](test_smoke.sh) | `amlc yent.aml -o yent_smoke` compiles cleanly, links libnotorch + libaml + Accelerate, produces an executable. If `weights/yent_v4/yent_v4_sft_q8_0.gguf` exists, also runs a 20-token generation and asserts non-empty output after the `--- generation ---` marker. |

## Run

```sh
make test          # runs both
make test-quant    # python only
make test-smoke    # bash only
```

Or invoke directly:

```sh
python3 tests/test_quantize.py
bash    tests/test_smoke.sh
```

## What they don't cover (yet)

- **`forward_token` numerical equivalence with `dario/infer_v4`.** A fixed prompt → fixed token sequence diff between the two binaries would catch any drift in the BLOOD COMPILE port. Worth adding once the 12-step layer lands and the forward stops being a pure copy.
- **AML field overlay determinism.** `am_apply_field_to_logits` mutates internal state (Hebbian co-occurrence, prophecy debt) — calling it twice with the same input gives different output. Determinism here is documented behaviour, not a bug, but a contract test would clarify.
- **Cross-platform.** Tests assume Darwin + Apple Accelerate. Linux + OpenBLAS path is mechanically identical but unverified here.

If you find a function that isn't covered, add a test before adding a fix.
