# The Quant v3.0 "Prometheus" — Makefile
# Convenience targets for development, building, testing, and deployment.

SHELL := /bin/bash
BINARY := target/release/the-quant
CARGO := cargo

.PHONY: help build release check test fmt clippy bench install deploy clean run daemon web health version

help:
	@echo "The Quant v3.0 'Prometheus' — Make targets"
	@echo ""
	@echo "  build     Build the debug binary"
	@echo "  release   Build the release binary (LTO, stripped)"
	@echo "  check     Run cargo check (fast compile check)"
	@echo "  test      Run the full test suite"
	@echo "  fmt       Format code with rustfmt"
	@echo "  clippy    Lint with clippy (deny warnings)"
	@echo "  bench     Run criterion benchmarks"
	@echo "  install   Make install.sh executable"
	@echo "  deploy    Run the single-command installer (install.sh)"
	@echo "  clean     Clean build artifacts"
	@echo "  run       Run the daemon"
	@echo "  daemon    Run the daemon"
	@echo "  web       Run the web dashboard"
	@echo "  health    Run a health check"
	@echo "  version   Print version"

build:
	$(CARGO) build

release:
	$(CARGO) build --release --features full

check:
	$(CARGO) check

test:
	$(CARGO) test --all-features

fmt:
	$(CARGO) fmt

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

bench:
	$(CARGO) bench

install:
	chmod +x install.sh
	@echo "install.sh is now executable. Run ./install.sh to set up a VPS."

deploy:
	chmod +x install.sh
	./install.sh

clean:
	$(CARGO) clean

run:
	$(CARGO) run --release -- daemon

daemon:
	$(CARGO) run --release -- daemon

web:
	$(CARGO) run --release -- web

health:
	$(CARGO) run --release -- health

version:
	$(CARGO) run --release -- version
