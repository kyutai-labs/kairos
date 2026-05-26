import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TIME_SENSITIVE_PROPS = "P54,P39,P2389,P768,P286,P488,P6,P127,P35,P1082,P166,P26,P118,P991,P1346,P822,P17,P276,P2882,P3279,P1351,P1350,P108,P102,P69,P3342,P393,P10606"


def is_instance_of(entity, qids):
    if "P31" not in entity["claims"]:
        return False
    for c in entity["claims"]["P31"]:
        if c["value"][0] in qids:
            return True
    return False


def process_time(s):
    if s[0] != "+":
        return None
    return s[1:11]


def get_time(claims):
    start, end = None, None
    if "P2348" in claims:
        value = claims["P2348"][0]["value"][1]
        if value and "one-year-period" in value:
            start = f"{value[:4]}-00-00"
            end = f"{value[5:9]}-00-00"
    if "P585" in claims:
        start = process_time(claims["P585"][0]["value"])
        end = start
        return [start, end]
    if "P580" in claims:
        start = process_time(claims["P580"][0]["value"])
    if "P582" in claims:
        end = process_time(claims["P582"][0]["value"])
    return [start, end]


def remove_year(s):
    s = re.sub(r"(\d{4})([-––](\d{2}|\d{4}))?", "", s)
    return s.strip().strip(",")


def extract_time_claims(rids=["P54"], qrank=None):
    qrank = {} if qrank is None else qrank

    for i, line in enumerate(sys.stdin):
        entity = json.loads(line)

        claims = defaultdict(lambda: [])
        for claim in entity["claims"]:
            claims[claim["property"]].append(claim)

        subj_rank = qrank[entity["id"]]

        for rid in rids:
            for c in claims[rid]:
                qualifiers = defaultdict(lambda: [])
                for qualifier in c["qualifiers"]:
                    qualifiers[qualifier["property"]].append(qualifier)
                time = get_time(qualifiers)
                if time == [None, None]:
                    time = get_time(claims)
                subj = remove_year(entity["name"])
                obj = c["value"]
                rel: list | str = [rid, c["property_label"]]
                if "P2501" in qualifiers:
                    rel = str(qualifiers["P2501"][0]["value"][1])
                if "P2389" in qualifiers:
                    obj[1] = (
                        str(obj[1]) + ", " + str(qualifiers["P2389"][0]["value"][1])
                    )
                if "P768" in qualifiers:
                    obj[1] = str(obj[1]) + ", " + str(qualifiers["P768"][0]["value"][1])

                print(
                    json.dumps(
                        {
                            "subject": subj,
                            "property": rel,
                            "object": obj,
                            "time": time,
                            "subject_rank": subj_rank,
                        },
                        ensure_ascii=False,
                    )
                )


def load_qrank(filename):
    qrank = defaultdict(lambda: -1)
    for i, line in enumerate(Path(filename).open()):
        if i == 0:
            continue
        a, b = line.strip().split(",")
        qrank[a] = int(b)
    return qrank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop", type=str, default=TIME_SENSITIVE_PROPS)
    parser.add_argument(
        "--qrank", type=str, default=None, help="Path to qrank csv file"
    )
    args = parser.parse_args()

    qrank = load_qrank(args.qrank) if args.qrank else None
    extract_time_claims(args.prop.split(","), qrank)
    exit(0)

    # cat kairos/data/wikidata-entities.jsonl | python kairos/data/dump.py --qrank data/qrank.csv > data/time_sensitive_claims.jsonl
