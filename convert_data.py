import json
import sys
from pathlib import Path
import argparse

def build_prompt(entry):
    # Prefer 'prompts' field; fall back to 'prompt' or compose from other fields
    return entry.get("prompts") or entry.get("prompt") or entry.get("message") or ""

def build_completion(entry, use_cot=False, cot_first=False):
    """
    Robustly extract a single assistant completion string from various fields.
    Handles 'output', 'outputs', 'plan' (list or scalar), and 'cot_outputs'.
    Returns a plain string.
    """
    final = ""

    # 1) new single-string key 'output'
    if entry.get("output") is not None:
        if isinstance(entry["output"], list):
            final = ", ".join(str(x) for x in entry["output"])
        else:
            final = str(entry["output"])

    # 2) legacy 'outputs' key (list or string)
    if not final and entry.get("outputs") is not None:
        if isinstance(entry["outputs"], list) and len(entry["outputs"]) > 0:
            final = str(entry["outputs"][0])
        else:
            final = str(entry["outputs"])

    # 3) 'plan' can be a list of ids (join with commas) or a string/number
    if not final and entry.get("plan") is not None:
        p = entry["plan"]
        if isinstance(p, list):
            final = ", ".join(str(x) for x in p)
        else:
            final = str(p)

    # 4) fallback to cot_outputs last element
    if not final and entry.get("cot_outputs") is not None:
        if isinstance(entry["cot_outputs"], list) and len(entry["cot_outputs"]) > 0:
            final = entry["cot_outputs"][-1]
        else:
            final = str(entry["cot_outputs"])

    # Ensure final is a string (avoid list/None issues)
    if final is None:
        final = ""
    elif not isinstance(final, str):
        final = str(final)

    # Optionally include chain-of-thought (careful: COT should NOT be used for public fine-tunes)
    if use_cot and entry.get("cot_outputs"):
        cot = "\n\n".join(entry["cot_outputs"]) if isinstance(entry["cot_outputs"], list) else str(entry["cot_outputs"])
        if cot_first:
            completion_text = cot.strip() + "\n\n" + final.strip()
        else:
            completion_text = final.strip() + "\n\n" + cot.strip()
    else:
        completion_text = final.strip()

    return completion_text

def _split_system_user(prompt_text: str):
    """
    Heuristic split: try to extract a 'system' instruction and the 'user' text.
    If no clear split, return a default system message and the whole prompt as user content.
    """
    if not prompt_text:
        return "You are a helpful assistant.", ""
    # try double-newline split
    parts = prompt_text.strip().split("\n\n", 1)
    if len(parts) == 2 and len(parts[0]) < 1000:
        system = parts[0].strip()
        user = parts[1].strip()
        return system, user
    # try splitting at common markers
    markers = ["Target objects:", "Current State:", "Available actions:", "Available actions", "Current State"]
    for m in markers:
        idx = prompt_text.find(m)
        if idx != -1:
            system = prompt_text[:idx].strip()
            user = prompt_text[idx:].strip()
            return (system or "You are a helpful assistant."), user
    # fallback
    return "You are a helpful assistant.", prompt_text.strip()

def build_messages(entry, use_cot=False, cot_first=False):
    """
    Build chat-format messages: [system, user, assistant]
    - system: heuristic extraction from prompt or default
    - user: remainder of prompt (task + context)
    - assistant: completion text (optionally with COT appended/prepended)
    """
    prompt_text = entry.get("prompts") or entry.get("prompt") or entry.get("message") or ""
    system_text, user_text = _split_system_user(prompt_text)

    assistant_text = build_completion(entry, use_cot=use_cot, cot_first=cot_first)
    # if assistant_text is empty, try outputs/plan/cot_outputs raw fallback
    if not assistant_text:
        assistant_text = ""
        if entry.get("outputs"):
            if isinstance(entry["outputs"], list) and len(entry["outputs"]) > 0:
                assistant_text = entry["outputs"][0]
            elif isinstance(entry["outputs"], str):
                assistant_text = entry["outputs"]
        if not assistant_text and entry.get("plan"):
            assistant_text = entry["plan"]
        if not assistant_text and entry.get("cot_outputs"):
            if isinstance(entry["cot_outputs"], list):
                assistant_text = entry["cot_outputs"][-1]
            else:
                assistant_text = str(entry["cot_outputs"])

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text}
    ]

def convert_dir(input_path: str, output_path: str, use_cot: bool = False, cot_first: bool = False,
                append: bool = True, dedupe: bool = False, recursive: bool = False):
    p = Path(input_path)
    # Accept a single file or a directory path.
    if p.is_file():
        files = [p]
    elif p.is_dir():
        # if recursive requested, search subdirectories as well, otherwise only top-level files
        if recursive:
            files = sorted(p.rglob("*_info.json"))
        else:
            files = sorted([x for x in p.glob("*_info.json") if x.is_file()])
    else:
        # allow glob-like input (e.g., "./outputs/*")
        files = sorted(Path(".").glob(str(p)))
    if not files:
        print("No *_info.json files found in", input_path)
        return

    mode = "a" if append else "w"
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # If dedupe requested, load existing entries into a set of signatures
    existing = set()
    if dedupe and out_path.exists():
        try:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    # support both old prompt/completion and new messages format for dedupe signature
                    if "messages" in obj and isinstance(obj["messages"], list):
                        msgs = obj["messages"]
                        # signature: (system, user, assistant)
                        sys_txt = msgs[0].get("content","").strip() if len(msgs) > 0 else ""
                        user_txt = msgs[1].get("content","").strip() if len(msgs) > 1 else ""
                        asst_txt = msgs[2].get("content","").strip() if len(msgs) > 2 else ""
                        sig = (sys_txt, user_txt, asst_txt)
                    else:
                        sig = (obj.get("prompt","").strip(), obj.get("completion","").strip())
                    existing.add(sig)
                except Exception:
                    continue
        except Exception:
            pass
    out = out_path.open(mode, encoding="utf-8")
    written = 0
    for f in files:
        try:
            j = json.load(open(f, "r", encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {f}: invalid JSON ({e})")
            continue
        # Deduplication: skip if signature exists
        sig = None
        if dedupe and "messages" in j and isinstance(j["messages"], list):
            msgs = j["messages"]
            sys_txt = msgs[0].get("content","").strip() if len(msgs) > 0 else ""
            user_txt = msgs[1].get("content","").strip() if len(msgs) > 1 else ""
            asst_txt = msgs[2].get("content","").strip() if len(msgs) > 2 else ""
            sig = (sys_txt, user_txt, asst_txt)
        elif dedupe:
            sig = (j.get("prompt","").strip(), j.get("completion","").strip())
        if sig and sig in existing:
            print("Skipping duplicate entry:", sig)
            continue
        existing.add(sig) # add new entry
        # Convert and write
        msgs = build_messages(j, use_cot=use_cot, cot_first=cot_first)
        out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        written += 1
    out.close()
    print(f"Wrote {written} entries to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert *_info.json -> chat-format JSONL (messages).")
    parser.add_argument("input_path", help="File or directory containing *_info.json files")
    parser.add_argument("output_path", help="Output jsonl file (will be created if missing)")
    parser.add_argument("--use_cot", type=int, choices=(0,1), default=0, help="Include chain-of-thought in assistant content")
    parser.add_argument("--cot_first", type=int, choices=(0,1), default=0, help="Put COT before final answer in assistant content")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories for *_info.json files")
    parser.add_argument("--dedupe", action="store_true", help="Skip entries already present in output (based on system+user+assistant)")
    # mutually exclusive append/overwrite handled manually (default: append)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--append", action="store_true", help="Append to output file (default behavior)")
    group.add_argument("--overwrite", action="store_true", help="Overwrite output file")

    args = parser.parse_args()

    append_flag = True
    if args.overwrite:
        append_flag = False
    elif args.append:
        append_flag = True
    else:
        append_flag = True

    print(f"Converting from: {args.input_path}")
    print(f"Writing to: {args.output_path} (append={append_flag}, recursive={args.recursive}, dedupe={args.dedupe})")

    convert_dir(
        input_path=args.input_path,
        output_path=args.output_path,
        use_cot=bool(args.use_cot),
        cot_first=bool(args.cot_first),
        append=append_flag,
        dedupe=args.dedupe,
        recursive=args.recursive,
    )

