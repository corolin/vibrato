# -*- coding: utf-8 -*-
# answer_jev.py — TypeSafe AI 官方标准版 Jev 答卷收集器
# 用法: python answer_jev.py <TYPESAFE_API_KEY>
import json
import os
import sys
import time
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = "https://api.typesafe.ai/v1/systemone"
FAMILY = ["happy", "affectionate", "calm", "amused", "sad", "anxious", "angry", "wronged"]
ITEMS = ["I1 快乐", "I2 高兴", "I3 满意", "I4 惬意", "I5 希望", "I6 放松",
         "I7 兴奋", "I8 警觉", "I9 刺激", "I10 狂热", "I11 活跃", "I12 惊慌",
         "I13 支配", "I14 影响", "I15 领导", "I16 重要", "I17 自由", "I18 强力"]

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

def call_jev(state, questions, key, model="jev-latest"):
    payload = json.dumps({"model": model, "state": state,
                          "questions": questions}).encode("utf-8")
    req = urllib.request.Request(BASE, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key,
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    last_err = None
    for retry in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            wait_s = (retry + 1) * 3
            print("    [重试 %d/3] %s, 等待 %ds..." % (retry + 1, str(e)[:80], wait_s), file=sys.stderr)
            time.sleep(wait_s)
    raise last_err

def main():
    if len(sys.argv) < 2:
        print("用法: python answer_jev.py <TYPESAFE_API_KEY>")
        print("获取 Key 途径: 登录 https://console.typesafe.ai 创建 API Key (新账号送 $5 赠金)")
        sys.exit(1)
    key = sys.argv[1].strip()

    # 读 40 题
    convos = []
    with open("data/test_convos.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                convos.append(json.loads(ln))

    # 构建每题的 state（与 answer_sheet 同逻辑）
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
            "instructions": "发送者的「%s」程度（1=完全不符合, 9=完全符合）" % name,
            "criteria": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }
    questions["family"] = {
        "type": "choice",
        "instructions": "发送者的主导情绪族",
        "criteria": {f: f for f in FAMILY}
    }

    out_file = "data/answers_jev.jsonl"
    done = 0
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            done = sum(1 for ln in f if ln.strip())
    print("\n==========================================")
    print("[START] 开始评测 TypeSafe 官方标准版 Jev")
    print("题数 %d, 已答 %d, 输出文件: %s" % (len(cases), done, out_file))
    print("==========================================")

    if done >= len(cases):
        print("[DONE] Jev 已全部完成 (%d/%d)，跳过。" % (done, len(cases)))
        return

    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(done, len(cases)):
            c = cases[i]
            t0 = time.time()
            try:
                r = call_jev(c["text"], questions, key, model="jev-latest")
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
                print("  [jev][%d/%d] %.2fs pad=[%+.2f, %+.2f] fam=%s" % (
                    i + 1, len(cases), row["latency_s"],
                    row["pad"][0], row["pad"][1], fam), flush=True)
                time.sleep(1.0)
            except Exception as e:
                print("  [jev][%d] 失败: %s" % (i, str(e)[:80]), file=sys.stderr)
                time.sleep(3)
    print("\n[SUCCESS] Jev 答卷完成 -> %s" % out_file)

if __name__ == "__main__":
    main()
