"""Minimal GQA executor for existence-type programs (select / filter / exist / and / or), ignoring the object ids in
the arguments so a question can be executed on any scene graph. Returns 'yes' / 'no', or None if the program uses an
unsupported operation."""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); G = os.path.join(ROOT, "data", "raw", "gqa")
SUPPORTED = {"select", "filter", "exist", "and", "or"}


def norm(s):
    return re.sub(r"\s*\(.*?\)\s*", "", s).strip()


def execute(semantic, scene):
    objs = scene["objects"]; out = []
    for st in semantic:
        op, arg, dep = st["operation"], st["argument"], st["dependencies"]
        if op == "select":
            n = norm(arg); r = {k for k, o in objs.items() if o["name"] == n}
        elif op.startswith("filter"):
            a = norm(arg); prev = out[dep[0]]
            if a.startswith("not(") and a.endswith(")"):
                a = a[4:-1]; r = {k for k in prev if a not in objs[k].get("attributes", [])}
            else:
                r = {k for k in prev if a in objs[k].get("attributes", [])}
        elif op == "exist":
            r = bool(out[dep[0]])
        elif op == "and":
            r = out[dep[0]] and out[dep[1]]
        elif op == "or":
            r = out[dep[0]] or out[dep[1]]
        else:
            return None
        out.append(r)
    return "yes" if out[-1] else "no"


def load():
    return json.load(open(os.path.join(G, "val_balanced_questions.json"))), json.load(open(os.path.join(G, "val_sceneGraphs.json")))
