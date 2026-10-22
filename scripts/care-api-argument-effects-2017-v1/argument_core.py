"""Bounded static argument analysis. No commands or APKs are executed.

Exact framework signatures only; joins, unmodeled instructions and escaping
mutable objects yield UNKNOWN. Categories describe command forms, not runtime
execution, binary identity, consent or maliciousness.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import shlex

SCHEMA = "care-argument-effects-command-v1"
MAX_STRING, MAX_ARRAY, MAX_INSTRUCTIONS = 1024, 32, 4096
CHANNELS = ("RUNTIME_STRING", "RUNTIME_ARRAY", "PROCESS_BUILDER")
EFFECTS = (
    "QUERY_COMMAND", "PRIVILEGE_HELPER", "PERMISSION_CHANGE", "OWNER_CHANGE",
    "FILE_REMOVE", "FILE_COPY_MOVE", "DIRECTORY_CHANGE", "PACKAGE_OPERATION",
    "NETWORK_TOOL", "PROCESS_CONTROL", "SHELL_WRAPPER", "OUTPUT_COMMAND",
    "OTHER_PROGRAM", "UNRESOLVED_COMMAND",
)
COARSE = (
    "APP_PRIVATE_PATH", "SHARED_STORAGE_PATH", "MEDIA_PATH", "CONTENT_URI",
    "NETWORK_ENDPOINT", "INTENT_ACTION", "COMPONENT_NAME", "CRYPTO_ALGORITHM",
    "STATIC_STRING", "INTEGER_CONSTANT", "BOOLEAN_CONSTANT",
    "CONSTANT_UNKNOWN_CATEGORY", "UNKNOWN",
)
PROGRAM_EFFECT = {
    **{x: "QUERY_COMMAND" for x in ("id", "whoami", "getprop", "ps", "ls", "stat", "cat", "logcat", "uname", "which", "df")},
    "su": "PRIVILEGE_HELPER", "sudo": "PRIVILEGE_HELPER",
    "chmod": "PERMISSION_CHANGE", "chown": "OWNER_CHANGE",
    "rm": "FILE_REMOVE", "rmdir": "FILE_REMOVE",
    "cp": "FILE_COPY_MOVE", "mv": "FILE_COPY_MOVE",
    "mkdir": "DIRECTORY_CHANGE", "mount": "DIRECTORY_CHANGE",
    "pm": "PACKAGE_OPERATION", "am": "PACKAGE_OPERATION",
    "curl": "NETWORK_TOOL", "wget": "NETWORK_TOOL", "nc": "NETWORK_TOOL",
    "kill": "PROCESS_CONTROL", "killall": "PROCESS_CONTROL",
    "echo": "OUTPUT_COMMAND", "printf": "OUTPUT_COMMAND",
}
RUNTIME_REFS = {
    "Ljava/lang/Runtime;->exec(" + first + rest + ")Ljava/lang/Process;": channel
    for first, channel in (("Ljava/lang/String;", "RUNTIME_STRING"), ("[Ljava/lang/String;", "RUNTIME_ARRAY"))
    for rest in ("", "[Ljava/lang/String;", "[Ljava/lang/String;Ljava/io/File;")
}


def exact_reference(value):
    return re.sub(r"\s+", "", str(value or ""))


def legacy_id(value):
    """Match frozen IDs; dispatch and literal analysis retain original case."""
    return str(int.from_bytes(hashlib.sha256(exact_reference(value).casefold().encode()).digest()[:8], "big"))


def coarse_string(value):
    low = value.strip().casefold()
    if low.startswith("content://"):
        return "CONTENT_URI"
    if re.match(r"^(https?|wss?|ftp)://", low) or re.match(r"^[a-z0-9.-]+\.[a-z]{2,}([/:]|$)", low):
        return "NETWORK_ENDPOINT"
    if low.startswith(("android.intent.action.", "android.settings.", "android.provider.telephony.")):
        return "INTENT_ACTION"
    if re.match(r"^l[^;]+;$", low):
        return "COMPONENT_NAME"
    if any(x in low for x in ("aes", "rsa", "sha-", "sha1", "sha256", "md5", "hmac", "desede", "pkcs")):
        return "CRYPTO_ALGORITHM"
    if low.startswith(("/data/data/", "/data/user/", "/data/user_de/")):
        return "APP_PRIVATE_PATH"
    if low.startswith(("/sdcard/", "/storage/emulated/", "/mnt/sdcard/")):
        return "SHARED_STORAGE_PATH"
    if any(x in low for x in ("/dcim/", "/pictures/", "/movies/", "/music/", "/download/")):
        return "MEDIA_PATH"
    return "STATIC_STRING" if value else "CONSTANT_UNKNOWN_CATEGORY"


def command_effects(argv, depth=0):
    """Categorize a command form, not the behavior of its unresolved executable."""
    if not argv or not all(isinstance(x, str) for x in argv) or not argv[0]:
        return ("UNRESOLVED_COMMAND",), "EMPTY_OR_UNKNOWN_ARGV"
    if any("\x00" in x for x in argv):
        return ("UNRESOLVED_COMMAND",), "NUL_IN_ARGUMENT"
    program = argv[0].rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", program):
        return ("UNRESOLVED_COMMAND",), "UNSUPPORTED_PROGRAM_FORM"
    if program in {"sh", "bash", "dash", "ash", "mksh"}:
        effects = {"SHELL_WRAPPER"}
        if depth >= 2 or len(argv) != 3 or argv[1] != "-c":
            return tuple(sorted(effects | {"UNRESOLVED_COMMAND"})), "SHELL_BODY_NOT_EXACT"
        body = argv[2]
        if re.search(r"[;&|<>$\x60*?\n\r\\]", body):
            return tuple(sorted(effects | {"UNRESOLVED_COMMAND"})), "UNSUPPORTED_SHELL_GRAMMAR"
        try:
            nested = shlex.split(body, posix=True, comments=False)
        except ValueError:
            return tuple(sorted(effects | {"UNRESOLVED_COMMAND"})), "SHELL_QUOTES_UNRESOLVED"
        inner, reason = command_effects(nested, depth + 1)
        return tuple(sorted(effects | set(inner))), reason
    if program in {"toybox", "toolbox", "busybox"}:
        if depth >= 2 or len(argv) < 2:
            return ("UNRESOLVED_COMMAND",), "APPLET_UNRESOLVED"
        return command_effects(argv[1:], depth + 1)
    return (PROGRAM_EFFECT.get(program, "OTHER_PROGRAM"),), None


@dataclass(frozen=True)
class Value:
    kind: str
    value: object


def bounded_string(text):
    return Value("string", text) if isinstance(text, str) and len(text) <= MAX_STRING else None


def analyze_method(instructions, frozen_targets=None, exception_regions=(), exception_error=None,
                   max_instructions=MAX_INSTRUCTIONS):
    """Basic-block-local interpretation of normalized DEX instruction records."""
    if len(instructions) > max_instructions:
        return {"status": "RESOURCE_LIMIT", "events": [], "unknown_reasons": ["METHOD_INSTRUCTION_LIMIT"]}
    offsets = [int(r.get("offset", i)) for i, r in enumerate(instructions)]
    if len(set(offsets)) != len(offsets) or offsets != sorted(offsets):
        return {"status": "UNPARSEABLE", "events": [], "unknown_reasons": ["INVALID_INSTRUCTION_OFFSETS"]}
    if exception_error:
        return {"status": "PARTIAL", "events": [], "unknown_reasons": ["EXCEPTION_TABLE_UNRESOLVED"]}
    leaders = {offsets[0]} if offsets else set()
    for i, row in enumerate(instructions):
        name = row.get("name", "")
        jump = name.startswith(("if-", "goto"))
        switch = name in {"packed-switch", "sparse-switch"}
        if (jump and "target" not in row) or (switch and ("switch_targets" not in row or row.get("switch_error"))):
            return {"status": "PARTIAL", "events": [], "unknown_reasons": ["CFG_TARGET_UNRESOLVED"]}
        if jump or switch or name == "throw" or name.startswith("return"):
            if i + 1 < len(offsets):
                leaders.add(offsets[i + 1])
            if jump:
                leaders.add(int(row["target"]))
            leaders.update(map(int, row.get("switch_targets", [])))
    for region in exception_regions or ():
        leaders.update([region["start_offset_bytes"], region["end_offset_bytes"]])
        leaders.update(h["target_offset_bytes"] for h in region.get("handlers", []))
    registers, heap = {}, {}
    pending = None
    events, reasons = [], set()

    def invalidate(values):
        for value in values:
            if value and value.kind == "object" and value.value in heap:
                obj = heap[value.value]
                obj["valid"] = False
                if obj.get("argv"):
                    invalidate([obj["argv"]])

    def argv_of(value):
        if not value or value.kind != "object":
            return None
        obj = heap.get(value.value, {})
        if not obj.get("valid"):
            return None
        if obj.get("kind") == "process_builder":
            arg = obj.get("argv")
            return argv_of(arg) if arg != value else None
        if obj.get("kind") == "array":
            if all(v is not None and v.kind == "string" for v in obj["items"]):
                return [v.value for v in obj["items"]]
        return None

    def emit(row, channel, value):
        argv, coarse = None, {"UNKNOWN"}
        if channel == "RUNTIME_STRING" and value and value.kind == "string":
            argv = [v for v in re.split(r"[ \t\n\r\f]+", value.value) if v]
            coarse = {coarse_string(value.value)}
        elif channel != "RUNTIME_STRING":
            argv = argv_of(value)
            if argv is not None:
                coarse = {coarse_string(v) for v in argv} or {"UNKNOWN"}
        effects, reason = command_effects(argv)
        events.append({
            "offset": int(row.get("offset", -1)), "endpoint_id": legacy_id(row.get("reference", "")),
            "channel": channel, "argument_position": 0 if channel != "PROCESS_BUILDER" else -1,
            "effects": list(effects), "coarse_categories": sorted(coarse),
            "resolved_argv": argv is not None, "fully_classified_form": reason is None,
            "unknown_reason": reason,
        })

    for index, row in enumerate(instructions):
        offset, name = offsets[index], str(row.get("name", ""))
        regs = list(map(int, row.get("registers", [])))
        if offset in leaders:
            registers.clear(); heap.clear(); pending = None
        if row.get("decode_error"):
            registers.clear(); heap.clear(); pending = None
            reasons.add("INSTRUCTION_DECODE_ERROR")
            continue
        if not name.startswith("move-result"):
            pending = None
        if row.get("payload"):
            registers.clear(); heap.clear()
            continue
        if name == "nop":
            continue
        if name.startswith("const-string") and regs:
            registers[regs[0]] = bounded_string(row.get("string"))
            continue
        if name.startswith("const") and not name.startswith(("const-class", "const-method")) and regs:
            registers[regs[0]] = Value("int", int(row["literal"])) if "literal" in row else None
            if name.startswith("const-wide"):
                registers[regs[0] + 1] = None
            continue
        if name.startswith("move-result") and regs:
            registers[regs[0]], pending = pending, None
            if name.startswith("move-result-wide"):
                registers[regs[0] + 1] = None
            continue
        if name.startswith("move") and len(regs) == 2:
            value = registers.get(regs[1])
            if name.startswith("move-wide"):
                registers[regs[0]] = value if value and value.kind == "int" else None
                registers[regs[0] + 1] = None
            else:
                registers[regs[0]] = value
            continue
        if name == "new-instance" and regs:
            kind = {"Ljava/lang/StringBuilder;": "string_builder",
                    "Ljava/lang/ProcessBuilder;": "process_builder"}.get(row.get("type"))
            key = (offset, regs[0])
            heap[key] = {"kind": kind, "valid": False}
            registers[regs[0]] = Value("object", key)
            continue
        if name == "new-array" and len(regs) == 2:
            length, key = registers.get(regs[1]), (offset, regs[0])
            valid = (row.get("type") == "[Ljava/lang/String;" and length is not None
                     and length.kind == "int" and 0 <= length.value <= MAX_ARRAY)
            heap[key] = {"kind": "array", "valid": valid, "items": [None] * int(length.value) if valid else []}
            registers[regs[0]] = Value("object", key)
            continue
        if name.startswith("filled-new-array"):
            key = (offset, -1)
            heap[key] = {"kind": "array", "valid": row.get("type") == "[Ljava/lang/String;" and len(regs) <= MAX_ARRAY,
                         "items": [registers.get(r) for r in regs]}
            pending = Value("object", key)
            continue
        if name.startswith(("aput", "aget")) and len(regs) == 3:
            array, item = registers.get(regs[1]), registers.get(regs[2])
            obj = heap.get(array.value, {}) if array and array.kind == "object" else {}
            valid = (obj.get("valid") and obj.get("kind") == "array" and item is not None
                     and item.kind == "int" and 0 <= item.value < len(obj["items"]))
            if name == "aput-object" and valid:
                obj["items"][item.value] = registers.get(regs[0])
            elif name == "aget-object" and valid:
                registers[regs[0]] = obj["items"][item.value]
            elif name.startswith("aput"):
                invalidate([array])
            else:
                registers[regs[0]] = None
            continue
        if name.startswith("invoke-"):
            ref = exact_reference(row.get("reference", ""))
            values = [registers.get(r) for r in regs]
            receiver = values[0] if values and not name.startswith("invoke-static") else None
            obj = heap.get(receiver.value, {}) if receiver and receiver.kind == "object" else {}
            target = frozen_targets is None or legacy_id(ref) in frozen_targets
            if ref in RUNTIME_REFS and name.startswith("invoke-virtual"):
                expected_words = 1 + ref.split("(", 1)[1].split(")", 1)[0].count(";")
                argument = values[1] if len(values) == expected_words else None
                if len(values) != expected_words:
                    reasons.add("INVOKE_REGISTER_ARITY_UNKNOWN")
                if target:
                    emit(row, RUNTIME_REFS[ref], argument)
                invalidate(values[1:])
                continue
            if ref == "Ljava/lang/ProcessBuilder;->start()Ljava/lang/Process;" and name.startswith("invoke-virtual"):
                if target:
                    emit(row, "PROCESS_BUILDER", receiver)
                invalidate(values)
                continue
            if ref.startswith("Ljava/lang/StringBuilder;->") and obj.get("kind") == "string_builder":
                if ref == "Ljava/lang/StringBuilder;-><init>()V" and name.startswith("invoke-direct"):
                    obj.update(valid=True, text="")
                elif ref == "Ljava/lang/StringBuilder;-><init>(Ljava/lang/String;)V" and name.startswith("invoke-direct") and len(values) == 2:
                    val = values[1]
                    obj.update(valid=bool(val and val.kind == "string"), text=val.value if val and val.kind == "string" else "")
                elif ref in {"Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;",
                             "Ljava/lang/StringBuilder;->append(I)Ljava/lang/StringBuilder;"} and len(values) == 2:
                    value = values[1]
                    expected = "int" if "append(I)" in ref else "string"
                    if obj.get("valid") and value and value.kind == expected:
                        text = obj["text"] + str(value.value)
                        obj.update(valid=len(text) <= MAX_STRING, text=text[:MAX_STRING])
                    else:
                        obj["valid"] = False
                    pending = receiver
                elif ref == "Ljava/lang/StringBuilder;->toString()Ljava/lang/String;" and obj.get("valid"):
                    pending = bounded_string(obj["text"])
                else:
                    invalidate(values)
                continue
            if ref == "Ljava/lang/String;->concat(Ljava/lang/String;)Ljava/lang/String;" and len(values) == 2:
                if all(v and v.kind == "string" for v in values):
                    pending = bounded_string(values[0].value + values[1].value)
                continue
            if ref.startswith("Ljava/lang/ProcessBuilder;->") and obj.get("kind") == "process_builder":
                constructor = ref == "Ljava/lang/ProcessBuilder;-><init>([Ljava/lang/String;)V" and name.startswith("invoke-direct")
                command = ref == "Ljava/lang/ProcessBuilder;->command([Ljava/lang/String;)Ljava/lang/ProcessBuilder;" and name.startswith("invoke-virtual")
                if (constructor or command) and len(values) == 2:
                    obj.update(valid=True, argv=values[1])
                    if command:
                        pending = receiver
                else:
                    invalidate(values)
                continue
            invalidate(values)
            continue
        if name.startswith(("if-", "goto", "return")) or name in {"throw", "packed-switch", "sparse-switch"}:
            registers.clear(); heap.clear()
            continue
        if name.startswith(("iput", "sput")):
            invalidate([registers.get(regs[0])] if regs else [])
            continue
        registers.clear(); heap.clear(); pending = None
        reasons.add("UNMODELED_INSTRUCTION_BARRIER")
    return {"status": "PARTIAL" if reasons else "ANALYZED", "events": events,
            "unknown_reasons": sorted(reasons)}


def aggregate_events(events):
    channels, effects, coarse, binding = Counter(), Counter(), Counter(), Counter()
    resolved = classified = 0
    for event in events:
        channel = event["channel"]
        channels[channel] += 1
        resolved += int(event["resolved_argv"]); classified += int(event["fully_classified_form"])
        coarse.update(event["coarse_categories"]); effects.update(event["effects"])
        binding.update(channel + ":" + effect for effect in event["effects"])
    return {"events": len(events), "resolved": resolved, "classified": classified,
            "channels": dict(channels), "effects": dict(effects),
            "coarse": dict(coarse), "binding": dict(binding)}
