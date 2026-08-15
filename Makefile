.PHONY: build install clean test

SUITES := archive driver_exec flamegraph graph kvlm_kv launch metrics profile propose vllmcfg

test:
	cargo build
	@for t in $(SUITES); do \
		cargo build --example $${t}_test 2>/dev/null; \
		./target/x86_64-unknown-linux-gnu/debug/examples/$${t}_test >/dev/null 2>&1 \
			&& echo "$$t: PASS" || { echo "$$t: FAIL"; exit 1; }; \
	done

BINARY := target/x86_64-unknown-linux-gnu/release/kvlm
DEST := $(HOME)/bin/kvlm

build:
	cargo build --release

install: build
	mkdir -p $(HOME)/bin
	cp $(BINARY) $(DEST)
	@echo "Installed $(DEST)"

clean:
	cargo clean
	rm -f $(DEST)
