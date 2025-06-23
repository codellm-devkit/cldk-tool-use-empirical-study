import json

trajs_file = "evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/sample_output.jsonl"

parsed_actions = []

with open(trajs_file) as f:
    for line in f:
        record = json.loads(line)
        history = record.get("history", [])
        for step in history:
            action = step.get("action")
            if action in ("system", "message", "think") or action is None:
                continue

            tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
            # Defensive: pick tool_calls from either tool_call_metadata or fallback
            if not tool_calls and "tool_call_metadata" in step:
                tool_calls = [step["tool_call_metadata"]]

            for call in tool_calls:
                # Find tool_call safely: works whether data comes from 'model_response' or fallback
                function_call = None
                if isinstance(call, dict):
                    if "function" in call:
                        function_call = call["function"]
                    elif "message" in call and "tool_calls" in call["message"]:
                        for tc in call["message"]["tool_calls"]:
                            if "function" in tc:
                                function_call = tc["function"]

                if not function_call:
                    continue

                tool_name = function_call.get("name")
                args_raw = function_call.get("arguments", "{}")

                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}

                if action == "run":
                    cmd = args.get("command", "").strip()
                    if cmd:
                        parsed_actions.append(cmd)
                else:
                    subcommand = args.get("command")  # e.g., view, str_replace, etc.
                    parsed = {
                        "tool": tool_name,
                        "subcommand": subcommand,
                        "args": args
                    }
                    parsed_actions.append(parsed)

print(json.dumps(parsed_actions, indent=2))
