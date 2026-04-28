import argparse
import sys
from datetime import datetime

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided,"
                " potentially due to prompt caching."
                " For more info, see"
                " https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(model_name: str, prompt: str, verbose: bool) -> OllamaResponse | None:

    last_element = None

    try:
        if verbose:
            stream = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stream=True,
            )
            for chunk in stream:
                print(chunk.message.content, end="", flush=True)
                last_element = chunk
        else:
            last_element = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        print("Make sure the Ollama daemon is running (ollama serve).")
        return None

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    return OllamaResponse.model_validate(last_element.model_dump())


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    prompt_duration_s = nanosec_to_sec(model_response.prompt_eval_duration)
    eval_duration_s = nanosec_to_sec(model_response.eval_duration)

    prompt_ts = (
        model_response.prompt_eval_count / prompt_duration_s
        if prompt_duration_s > 0
        else 0.0
    )
    response_ts = (
        model_response.eval_count / eval_duration_s if eval_duration_s > 0 else 0.0
    )
    total_tokens = model_response.prompt_eval_count + model_response.eval_count
    total_duration_s = prompt_duration_s + eval_duration_s
    total_ts = total_tokens / total_duration_s if total_duration_s > 0 else 0.0

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {prompt_duration_s:.2f}s
        \tResponse time: {eval_duration_s:.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: list[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    n = len(responses)
    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {n} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses) // n,
        load_duration=sum(r.load_duration for r in responses) // n,
        prompt_eval_count=sum(r.prompt_eval_count for r in responses) // n,
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses) // n,
        eval_count=sum(r.eval_count for r in responses) // n,
        eval_duration=sum(r.eval_duration for r in responses) // n,
    )
    print("Average stats:")
    inference_stats(res)


WARMUP_PROMPT = "Hi"


def warm_up(model_name: str, verbose: bool):
    """Run a short throwaway prompt to load the model into memory,
    so that subsequent benchmark runs are not polluted by load time."""
    if verbose:
        print(f"Warming up {model_name}...")
    try:
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": WARMUP_PROMPT}],
        )
    except Exception as e:
        print(f"Warning: warm-up failed for {model_name}: {e}")
    if verbose:
        print(f"Warm-up complete for {model_name}\n")


def get_benchmark_models(skip_models: list[str] | None = None) -> list[str]:
    if skip_models is None:
        skip_models = []

    try:
        models = ollama.list().models or []
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        print("Make sure the Ollama daemon is running (ollama serve).")
        sys.exit(1)

    model_names = [model.model for model in models]
    if len(skip_models) > 0:
        model_names = [model for model in model_names if model not in skip_models]
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            "Why is the sky blue?",
            "Write a report on the financials of Apple Inc.",
        ],
        help=(
            "List of prompts to use for benchmarking."
            " Separate multiple prompts with spaces."
        ),
    )
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        default=False,
        help=(
            "Skip the warm-up run. By default, each model is"
            " warmed up before benchmarking to avoid cold-start"
            " overhead polluting results."
        ),
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    prompts = args.prompts
    do_warm_up = not args.no_warm_up
    print(
        f"\nVerbose: {verbose}"
        f"\nSkip models: {skip_models}"
        f"\nWarm-up: {do_warm_up}"
        f"\nPrompts: {prompts}"
    )

    model_names = get_benchmark_models(skip_models)
    benchmarks = {}

    for model_name in model_names:
        if do_warm_up:
            warm_up(model_name, verbose)

        responses: list[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            if response is None:
                print(f"Skipping failed run for {model_name}")
                continue
            responses.append(response)

            if verbose:
                print(f"Response: {response.message.content}")
                inference_stats(response)
        benchmarks[model_name] = responses

    for _model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose \
    #   --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M \
    #   llama2:latest \
    #   --prompts "What color is the sky" \
    #   "Write a report on the financials of Microsoft"
    # python benchmark.py --no-warm-up --verbose
