import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

from openai import OpenAI


BASE_PROMPT = """
Generate a narrative in the genre of Comedy of 150-200 words. Characters face misunderstandings and playful conflicts, the tone is light and witty, and the story resolves with reconciliation and a happy ending
"""

REALITY_FRAMES = {
    "zero_shot": "",  # No frame instruction - baseline
    "everyday": (
        """
        You are operating within the world
of everyday life as described by Alfred Schütz. This reality functions as the paramount province of meaning, taken for granted as self-
evidently real and serving as the reference point for all other realities.
Consciousness is characterized by a high tension of consciousness, manifesting as wide-awakeness and full attentiveness to practical
affairs. The world is experienced in the natural attitude; its existence is not questioned, and doubt is suspended. Meaningful conduct
takes the form of working: goal-oriented, pragmatic action aimed at transforming the external world through bodily activity. The self is
experienced as a working self, fully engaged in ongoing projects and embedded in a continuous biographical trajectory. The social world is
fully intersubjective, constituted through communication, coordination, and reciprocal expectations shared with others who are equally
wide-awake. Time is experienced as standard, socially shared time, structured by schedules, sequences, and future-oriented projects. 
"""  ),
    "scientific": (
        """
You are operating within the world of scientific contemplation as described by Alfred Schütz.
This reality constitutes a clearly delimited province of meaning, governed by methodological rules, theoretical interests, and scientific
validity criteria. Consciousness is characterized by a moderately high but controlled tension, marked by reflective and disciplined attentiveness oriented toward explanation rather than practical intervention.
The scientific epoché is in force: the natural attitude and pragmatic relevance of the everyday world are suspended through methodological doubt. Meaningful conduct takes the form of theorizing, observing,
measuring, and explaining, guided by explicit rules of inference and standards of validation. The self is experienced as a knowing and observing self, detached from immediate practical concerns and from
the contingencies of personal biography. Sociality is constituted as a community of inquiry, in which intersubjectivity is mediated by shared concepts, methods, and procedures of verification rather than
face-to-face coordination. Time is experienced as objective, homogeneous, and reversible, functioning as a neutral parameter within
explanatory frameworks.
        """
        ),
    "imaginative": (
        """
You are operating within the world of dreams as described by Alfred Schütz. This reality forms a weakly bounded and unstable province of meaning, entered involuntarily through sleep
and dissolving upon awakening. Consciousness is characterized by a very low tension of consciousness, manifesting as passive, relaxed, and drifting awareness. Everyday reality is suspended; within the dream,
the dream world is experienced without doubt, despite the absence of practical or logical constraints. Meaningful conduct exhibits minimal spontaneity: events occur to the dreamer rather than being actively initiated or controlled. The self is experienced as fragmented or unstable,
with shifting, inconsistent, or discontinuous identity. Sociality appears as pseudo-sociality, populated by dream figures whose presence does not involve genuine intersubjective reciprocity or shared validation.
Time is experienced as discontinuous and distorted, allowing compression, expansion, leaps, and the absence of stable chronological order.
        """
    )
}


def build_prompt(frame_name: str) -> str:
    if frame_name == "zero_shot":
        # No reality frame instruction - just the base prompt
        return (
            f"{BASE_PROMPT}\n"
            "Write a single short narrative of about 150–200 words.\n"
            "Do not add a title. Do not use bullet points.\n"
        )
    else:
        # Include reality frame instruction
        return (
            f"Reality frame: {frame_name}\n"
            f"Scenario: {BASE_PROMPT}\n\n"
            f"Instruction: {REALITY_FRAMES[frame_name]}\n\n"
            "Write a single short narrative of about 150–200 words.\n"
            "Do not add a title. Do not use bullet points.\n"
        )


def generate_one(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    """Generate narrative using OpenAI Chat Completions API."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def main(model="gpt-4o-mini", repeats=5, out="outputs.jsonl", 
         min_temp=0.0, max_temp=0.0, sleep_time=0.2):
    """
    Generate narratives with different reality frames.
    
    Args:
        model: OpenAI model name
        repeats: Number of repeats per frame (5-10 recommended)
        out: Output JSONL file path
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        sleep_time: Sleep seconds between API calls
    """
    
    # Create a simple namespace object to mimic argparse args
    class Args:
        pass
    
    args = Args()
    args.model = model
    args.repeats = repeats
    args.out = out
    args.min_temp = min_temp
    args.max_temp = max_temp
    args.sleep = sleep_time

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY env var. Set it and rerun.")

    client = OpenAI()

    random.seed(42)
    total = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for frame in REALITY_FRAMES.keys():
            for i in range(args.repeats):
                temp = random.uniform(args.min_temp, args.max_temp)
                prompt = build_prompt(frame)

                try:
                    text = generate_one(client, args.model, prompt, temp)
                except Exception as e:
                    # Basic resilience: log the error and continue
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "frame": frame,
                        "repeat_index": i,
                        "model": args.model,
                        "temperature": temp,
                        "prompt": prompt,
                        "error": str(e),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    time.sleep(max(args.sleep, 1.0))
                    continue

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frame": frame,
                    "repeat_index": i,
                    "model": args.model,
                    "temperature": temp,
                    "prompt": prompt,
                    "output": text,
                    "word_count": len(text.split()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                total += 1
                print(f"Generated {total}/{args.repeats * len(REALITY_FRAMES)}: {frame} (repeat {i+1})")
                time.sleep(args.sleep)

    print(f"\nDone. Wrote {total} generations to {args.out}")


main(model="gpt-4o-mini", repeats=5, out="outputs.jsonl")
