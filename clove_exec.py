"""Set-lifted GQA executor for existence questions with relations (CLOVE). Extends gqa_exec: select / filter <any
attribute type> / relate / exist / and / or. Object ids in arguments are ignored, so a question runs on any scene graph.
Position filters (hposition / vposition) need geometry and are not supported. Returns 'yes' / 'no', or None."""
import ast, json, os, re
ROOT = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(ROOT, "data", "raw", "gqa")
UNSUP = {"filter hposition", "filter vposition"}


def norm(s):
    return re.sub(r"\s*\(.*?\)\s*", "", s).strip()


def prog(p):
    return p if isinstance(p, list) else ast.literal_eval(p)


def supported(p):
    return all(s["operation"] in ("select", "relate", "exist", "and", "or") or (s["operation"].startswith("filter") and s["operation"] not in UNSUP)
               for s in p) and p[-1]["operation"] in ("exist", "and", "or")


def execute(p, scene):
    objs = scene["objects"]; out = []
    for st in p:
        op, arg, dep = st["operation"], st["argument"], st["dependencies"]
        if op == "select":
            n = norm(arg); r = {k for k, o in objs.items() if o["name"] == n}
        elif op.startswith("filter"):
            a = norm(arg); prev = out[dep[0]]
            if a.startswith("not(") and a.endswith(")"):
                a = a[4:-1]; r = {k for k in prev if a not in objs[k].get("attributes", [])}
            else:
                r = {k for k in prev if a in objs[k].get("attributes", [])}
        elif op == "relate":
            name, rel, side = [x.strip() for x in norm(arg).rsplit(",", 2)]; X = out[dep[0]]
            ok = lambda k: name == "_" or objs[k]["name"] == name
            if side == "s":   # the new object is the subject: new --rel--> X
                r = {k for k, o in objs.items() if ok(k) and any(e["name"] == rel and e["object"] in X for e in o.get("relations", []))}
            else:             # the new object is the object: X --rel--> new
                r = {e["object"] for x in X for e in objs[x].get("relations", []) if e["name"] == rel and e["object"] in objs and ok(e["object"])}
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


def scenes():
    S = json.load(open(os.path.join(G, "train_sceneGraphs.json"))); S.update(json.load(open(os.path.join(G, "val_sceneGraphs.json")))); return S
