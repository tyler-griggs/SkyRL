### Terminal-Bench integration (WIP)

Integration with Terminal-Bench is a work in progress. For now, training tasks are hard-coded as "hello-world" in the prototype. The next TODO is to support specifying a training set of Terminal-Bench tasks.

- **Training**: run the GRPO training pipeline. Requires a dummy dataset (for now).
```bash
uv run -- python examples/gsm8k/gsm8k_dataset.py
bash examples/terminal_bench/run_terminal_bench.sh
```

- **Generation only**: launch the generator/serving process
```bash
bash examples/terminal_bench/run_terminal_bench_gen.sh
```