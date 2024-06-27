cd Runtime/lib/perfetto
./tools/install-build-deps
tools/gn args tools
tools/ninja -C tools traceconv tracebox
tools/gen_amalgamated --output sdk/perfetto