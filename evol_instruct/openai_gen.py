from typing import Any, Dict, List

import os
from time import sleep
import numpy as np

import openai
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm


class AsyncOpenAIPredictor:
    def __init__(
        self,
        model,
        max_tokens,
        num_generations=1,
        temperature=0.0,
        top_p=1.0,
        openai_timeout=45,
    ):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_KEY"),
            base_url="https://chatapi.onechats.top/v1/",
        )
        self.client_kwargs: dict[str | str] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": num_generations,
            "timeout": openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    async def __call__(self, item: Dict[str, np.ndarray]) -> Dict[str, list]:
        assert isinstance(item["messages"], list)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=item["messages"], **self.client_kwargs
                )
                return {**item, "generation": response.choices[0].message.content}
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                print(f"[Attempt {attempt + 1}] Exception: {repr(e)}")
                print(f"[Attempt {attempt + 1}] Sleeping for 30 seconds...")
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Failed to run the model for {item['messages']}!")
                print("Exception: ", repr(e))
                return None


async def run_openai_inference_async(requests, **kwargs):
    predictor = AsyncOpenAIPredictor(**kwargs)

    results = []
    tasks = [predictor(request) for request in requests]

    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing items"
    ):
        result = await task
        if result:
            results.append(result)

    return results


def run_openai_inference(requests, **kwargs):
    requests = asyncio.run(run_openai_inference_async(requests, **kwargs))
    return requests


def test():
    import ray
    import ray.data
    ds = ray.data.from_items([
        {"messages": [{"content": "hello", "role": "user"}]},
        {"messages": [{"content": "hi", "role": "user"}]},
    ])
    ds = run_openai_inference(ds, model="gpt-4o-mini", max_tokens=1024)
    x = ds.take_all()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test()
