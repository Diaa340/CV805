"""Minimal CLEVR program executor (functional programs of CLEVR v1.0 on scene graphs). execute() returns the answer
string or None when the program is ill-posed on that scene (e.g. 'unique' on a set that is not a single object)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); META = os.path.join(ROOT, "data", "clevr_meta", "CLEVR_v1.0")


class Bad(Exception):
    pass


def execute(program, scene):
    objs = scene["objects"]; rel = scene["relationships"]; out = []
    for f in program:
        fn, ins, v = f["function"], [out[i] for i in f["inputs"]], f["value_inputs"]
        if fn == "scene":
            r = list(range(len(objs)))
        elif fn.startswith("filter_"):
            a = fn[7:]; r = [o for o in ins[0] if objs[o][a] == v[0]]
        elif fn == "unique":
            if len(ins[0]) != 1:
                raise Bad
            r = ins[0][0]
        elif fn == "relate":
            r = list(rel[v[0]][ins[0]])
        elif fn.startswith("same_"):
            a = fn[5:]; r = [o for o in range(len(objs)) if o != ins[0] and objs[o][a] == objs[ins[0]][a]]
        elif fn == "count":
            r = len(ins[0])
        elif fn == "exist":
            r = "yes" if ins[0] else "no"
        elif fn.startswith("query_"):
            r = objs[ins[0]][fn[6:]]
        elif fn.startswith("equal_") and fn != "equal_integer":
            r = "yes" if ins[0] == ins[1] else "no"
        elif fn == "equal_integer":
            r = "yes" if ins[0] == ins[1] else "no"
        elif fn == "less_than":
            r = "yes" if ins[0] < ins[1] else "no"
        elif fn == "greater_than":
            r = "yes" if ins[0] > ins[1] else "no"
        elif fn == "intersect":
            r = sorted(set(ins[0]) & set(ins[1]))
        elif fn == "union":
            r = sorted(set(ins[0]) | set(ins[1]))
        else:
            raise ValueError(fn)
        out.append(r)
    return out[-1]


def safe(program, scene):
    try:
        return execute(program, scene)
    except Bad:
        return None


def load(split="val"):
    Q = json.load(open(os.path.join(META, "questions", f"CLEVR_{split}_questions.json")))["questions"]
    S = json.load(open(os.path.join(META, "scenes", f"CLEVR_{split}_scenes.json")))["scenes"]
    return Q, S


if __name__ == "__main__":
    Q, S = load("val"); ok = sum(str(safe(q["program"], S[q["image_index"]])) == q["answer"] for q in Q)
    print(f"executor agreement on val (own image): {ok}/{len(Q)} = {ok / len(Q):.4f}")


def execute_total(program, scene):
    """Set-lifted (existential) semantics: 'unique' passes its set through; relate / same_* / query_* act on sets; equal_*
    on attribute-value sets = 'share a value'. Always well-posed. Returns (answer, lifted) where lifted = some 'unique'
    step saw a non-singleton set. Identical to execute() whenever every 'unique' step sees exactly one object."""
    objs = scene["objects"]; rel = scene["relationships"]; out = []; lifted = False
    for f in program:
        fn, ins, v = f["function"], [out[i] for i in f["inputs"]], f["value_inputs"]
        if fn == "scene":
            r = set(range(len(objs)))
        elif fn.startswith("filter_"):
            a = fn[7:]; r = {o for o in ins[0] if objs[o][a] == v[0]}
        elif fn == "unique":
            lifted |= len(ins[0]) != 1; r = set(ins[0])
        elif fn == "relate":
            r = set().union(*[set(rel[v[0]][o]) for o in ins[0]]) if ins[0] else set()
        elif fn.startswith("same_"):
            a = fn[5:]; vals = {objs[s][a] for s in ins[0]}; r = {o for o in range(len(objs)) if o not in ins[0] and objs[o][a] in vals}
        elif fn == "count":
            r = len(ins[0])
        elif fn == "exist":
            r = "yes" if ins[0] else "no"
        elif fn.startswith("query_"):
            a = fn[6:]; r = {objs[o][a] for o in ins[0]}
        elif fn.startswith("equal_") and fn != "equal_integer":
            r = "yes" if ins[0] & ins[1] else "no"
        elif fn == "equal_integer":
            r = "yes" if ins[0] == ins[1] else "no"
        elif fn == "less_than":
            r = "yes" if ins[0] < ins[1] else "no"
        elif fn == "greater_than":
            r = "yes" if ins[0] > ins[1] else "no"
        elif fn == "intersect":
            r = ins[0] & ins[1]
        elif fn == "union":
            r = ins[0] | ins[1]
        else:
            raise ValueError(fn)
        out.append(r)
    return out[-1], lifted
