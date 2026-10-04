# -*- coding: utf-8 -*-
# answer_clef_lite.py — Python 3.6 兼容, 零第三方依赖, 在 lite-4c4g-2 上跑
# 用法: python3 answer_clef_lite.py <api_key>
import json
import os
import sys
import time
import urllib.request

BASE = "https://decision.mirages.cc/v1/systemone"
FAMILY = ["happy", "affectionate", "calm", "amused", "sad", "anxious", "angry", "wronged"]
ITEMS = ["I1 快乐", "I2 高兴", "I3 满意", "I4 惬意", "I5 希望", "I6 放松",
         "I7 兴奋", "I8 警觉", "I9 刺激", "I10 狂热", "I11 活跃", "I12 惊慌",
         "I13 支配", "I14 影响", "I15 领导", "I16 重要", "I17 自由", "I18 强力"]

# 十分干净的 PAD 换算（与 pad_schema.py 同式, 不 import 保持 3.6 兼容）
def items_to_pad_bins(items):
    bins = {}
    groups = {"pleasure": ["I%d" % i for i in range(1, 7)],
              "arousal": ["I%d" % i for i in range(7, 13)],
              "dominance": ["I%d" % i for i in range(13, 19)]}
    for dim, keys in groups.items():
        acc = [0.0] * 9
        for k in keys:
            v = items.get(k, 5.0)
            for i in range(9):
                d = abs(i + 1 - v)
                if d < 1.0:
                    acc[i] += 1.0 - d
        s = sum(acc)
        bins[dim] = [round(x / s, 5) if s > 0 else 1.0 / 9 for x in acc]
    return bins

def bins_to_pad(bins):
    out = {}
    for dim in ["pleasure", "arousal", "dominance"]:
        exp = sum(p * (i + 1) for i, p in enumerate(bins[dim]))
        out[dim] = round(max(-1.0, min(1.0, (exp - 5.0) / 4.0)), 6)
    return out

def call_clef(state, questions, key):
    payload = json.dumps({"model": "clef-flash", "state": state,
                          "questions": questions}).encode("utf-8")
    req = urllib.request.Request(BASE, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    return r

def main():
    if len(sys.argv) < 2:
        print("用法: python3 answer_clef_lite.py <api_key>")
        sys.exit(1)
    key = sys.argv[1]

    # 读 40 题
    convos = []
    with open("/tmp/test_convos.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                convos.append(json.loads(ln))

    # 构建每题的 state（与 answer_sheet.questions() 同逻辑）
    cases = []
    for c in convos:
        msgs = c["messages"]
        for i, m in enumerate(msgs):
            if m["side"] == "user" and len(cases) < 40:
                lines = []
                start = max(0, i - 6)
                for j in range(start, i + 1):
                    tag = "TARGET(需判断): " if j == i else ""
                    role = "[用户]" if msgs[j]["side"] == "user" else "[AI]"
                    lines.append(tag + role + ": " + msgs[j].get("text", ""))
                cases.append({"id": c["id"], "idx": i, "text": "\n".join(lines)})

    # 构建 questions schema（18 score + 1 choice）
    questions = {}
    for i, name in enumerate(ITEMS, 1):
        questions["I%d" % i] = {
            "type": "score",
            "instructions": "发送者的「%s」程度（1=完全不符合, 9=完全符合）" % name
        }
    questions["family"] = {
        "type": "choice",
        "instructions": "发送者的主导情绪族",
        "criteria": {f: f for f in FAMILY}
    }

    out_file = "/tmp/answers_clef-flash.jsonl"
    done = 0
    if os.path.exists(out_file):
        with open(out_file) as f:
            done = sum(1 for ln in f if ln.strip())
    print("题数 %d, 已答 %d" % (len(cases), done))

    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(done, len(cases)):
            c = cases[i]
            t0 = time.time()
            try:
                r = call_clef(c["text"], questions, key)
                ans = r.get("answers", {})
                items = {}
                for j in range(1, 19):
                    v = ans.get("I%d" % j, {}).get("score", 5.0)
                    items["I%d" % j] = max(1.0, min(9.0, float(v)))
                bins = items_to_pad_bins(items)
                pad = bins_to_pad(bins)
                fam = ans.get("family", {}).get("choice", "calm")
                if fam not in FAMILY:
                    fam = "calm"
                row = {"idx": i, "convo": c["id"],
                       "text_tail": c["text"][-60:],
                       "latency_s": round(time.time() - t0, 2),
                       "usage": {"prompt": 0, "completion": 0},
                       "items": items,
                       "pad": [pad["pleasure"], pad["arousal"], pad["dominance"]],
                       "pad_bins": bins,
                       "family": fam,
                       "noul": {"directed_at_me": False, "suppressed": False}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print("  [%d/%d] %.1fs pad=%.2f,%.2f fam=%s" % (
                        i + 1, len(cases), row["latency_s"],
                        row["pad"][0], row["pad"][1], fam), flush=True)
                time.sleep(1.5)  # 限速
            except Exception as e:
                print("  [%d] 失败: %s" % (i, str(e)[:80]), file=sys.stderr)
                time.sleep(5)
    print("完成 → %s" % out_file)

if __name__ == "__main__":
    main()
