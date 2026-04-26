# yent.aml — Makefile
# Compiles yent.aml through amlc and runs the local test suite.

.PHONY: all yent test test-quant test-smoke clean

all: yent

# Compile the AML program through amlc. Assumes amlc, libaml, libnotorch
# are installed system-wide (see ariannamethod.ai/Makefile and notorch/Makefile
# install targets).
yent: yent.aml tools/yent_forward.h tools/janus_v4_bpe_merges.h
	amlc yent.aml -o yent

test: test-quant test-smoke

test-quant:
	python3 tests/test_quantize.py

test-smoke:
	bash tests/test_smoke.sh

clean:
	rm -f yent yent.c yent_smoke yent_smoke.c

# Help
help:
	@echo "yent.aml — second AML inference"
	@echo
	@echo "  make            Compile yent.aml → ./yent (amlc auto-links libnotorch+libaml+Accelerate)"
	@echo "  make test       Run quantize round-trip + smoke compilation"
	@echo "  make test-quant Run Q8_0/Q4_K/fp16 round-trip in Python"
	@echo "  make test-smoke Run amlc + optional generation if weights present"
	@echo "  make clean      Remove build artefacts"
	@echo
	@echo "  Run after build:"
	@echo "    ./yent -w weights/yent_v4/yent_v4_sft_q8_0.gguf \\"
	@echo "           -p \"Q: Who are you?\\\\nA:\" -n 80 -t 0.7 --top-p 0.9"
