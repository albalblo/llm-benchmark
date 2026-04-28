# LLM Benchmark (ollama only)

This tool allows you to get the t/s (tokens per second) of Large Language Models (LLMs) running on your local machine. Currently we only support testing Ollama llms.

This was forked from MinhNgyuen/llm-benchmark to solve a bug, and add functionalities I needed.

## Example output

Output on a Nvidia 4090 windows desktop

```bash
Average stats:

----------------------------------------------------
        llama2:13b
                Prompt eval: 690.15 t/s
                Response: 78.27 t/s
                Total: 80.78 t/s

        Stats:
                Prompt tokens: 42
                Response tokens: 1155
                Model load time: 2.87s
                Prompt eval time: 0.06s
                Response time: 14.76s
                Total time: 17.69s
----------------------------------------------------

Average stats:

----------------------------------------------------
        llama2:latest
                Prompt eval: 1148.29 t/s
                Response: 123.31 t/s
                Total: 127.41 t/s

        Stats:
                Prompt tokens: 42
                Response tokens: 1122
                Model load time: 1.97s
                Prompt eval time: 0.04s
                Response time: 9.10s
                Total time: 11.11s
----------------------------------------------------
```

## Getting Started

Follow these instructions to set up and run benchmarks on your system.

### Prerequisites

- Python 3.9+
- [ollama](https://ollama.com/) installed

### Installation

1. **Clone this repository**

   Open a terminal or powershell and run:

   ```bash
   git clone https://github.com/albalblo/llm-benchmark.git
   cd llm-benchmark
   ```

2. **Create a virtual environment**

   - **Windows**

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **Linux/macOS**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Serve the Ollama model**

   Before running benchmarks, make sure your Ollama model server is running:

   ```bash
   ollama serve
   ```

### Running Benchmarks

To run benchmarks, use the `benchmark.py` script with the desired command line arguments:

```bash
python benchmark.py --verbose --prompts "Why is the sky blue?" "Write a report on the financials of Nvidia"
```

#### Command Line Arguments

- `--verbose`: Prints the prompts and streams the responses from Ollama
- `--skip-models`: Specify a list of model names to skip during the benchmark. Get the list of possible models by running the command `ollama list`. Separate multiple models with spaces.
- `--prompts`: Provide custom prompts to use for benchmarking. Separate multiple prompts with spaces.
- `--no-warm-up`: Skip the warm-up run. By default, each model receives a short throwaway prompt before benchmarking to ensure it is fully loaded into memory, so that cold-start overhead does not pollute the results. Use this flag if you want to include model load time in your measurements.
- `--no-think`: Disable thinking/reasoning mode for models that support it (e.g. DeepSeek R1, QwQ). Models without thinking capability will ignore this flag. Useful for benchmarking raw response speed without the overhead of chain-of-thought reasoning.

#### Examples

- **Run with default prompts in verbose mode**

  ```bash
  python benchmark.py --verbose
  ```

- **Run with custom prompts**

  ```bash
  python benchmark.py --prompts "Custom prompt 1" "Why is the sky blue?"
  ```

- **Skip specific models**

  ```bash
  python benchmark.py --skip-models model1 llama2:latest
  ```

- **Disable warm-up to include cold-start time**

  ```bash
  python benchmark.py --no-warm-up
  ```

- **Disable thinking mode for reasoning models**

  ```bash
  python benchmark.py --no-think --verbose
  ```

## Development

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
pip install ruff
```

To check and auto-fix lint issues:

```bash
ruff check --fix .
```

To format the code:

```bash
ruff format .
```

You can run both in one go before committing:

```bash
ruff check --fix . && ruff format .
```

## Contributing

We welcome contributions! Please feel free to submit a pull request or create an issue if you have suggestions for improvements or have identified bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
