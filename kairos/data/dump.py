import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

""" Process Wikidata raw JSON dump"""


def process_value(datavalue, datatype, labels):
    if datatype == "wikibase-item":
        wid = datavalue["value"]["id"]
        return wid, labels[wid]
    if datatype == "quantity":
        if datavalue["value"]["unit"] == "1":
            return datavalue["value"]["amount"]
        else:
            return datavalue["value"]["amount"], labels[datavalue["value"]["unit"][31:]]
    if datatype == "monolingualtext":
        if datavalue["value"]["language"] == "en":
            return datavalue["value"]["text"]
    if datatype == "math" or datatype == "string":
        return datavalue["value"]
    if datatype == "time":
        if (
            datavalue["value"]["calendarmodel"]
            != "http://www.wikidata.org/entity/Q1985727"
        ):
            return None
        return datavalue["value"]["time"]
    return None


def process_snak(snak, labels):
    # KEYS : ['snaktype', 'property', 'hash', 'datavalue', 'datatype']
    if snak["snaktype"] != "value":
        return None
    value = process_value(snak["datavalue"], snak["datatype"], labels)
    if value is None:
        return None
    return {
        "property": snak["property"],
        "property_label": labels[snak["property"]],
        "value": value,
    }


def process_line(line, labels):
    data = json.loads(line[:-2])
    # ['type', 'id', 'labels', 'descriptions', 'aliases', 'claims', 'sitelinks', 'pageid', 'ns', 'title', 'lastrevid', 'modified']

    # Filtering only English labels for now
    if "en" not in data["labels"]:
        return None

    name = data["labels"]["en"]["value"]

    output = {
        "id": data["id"],
        "name": name,
    }

    if "en" in data["descriptions"]:
        output["description"] = data["descriptions"]["en"]["value"]
    if "en" in data["aliases"]:
        output["aliases"] = [x["value"] for x in data["aliases"]["en"]]

    output["claims"] = []

    for claims in data["claims"].values():
        for claim in claims:
            if claim["mainsnak"]["snaktype"] != "value":
                continue
            obj = process_value(
                claim["mainsnak"]["datavalue"], claim["mainsnak"]["datatype"], labels
            )
            if obj is None:
                continue

            qualifiers = []
            if "qualifiers" in claim:
                for q in claim["qualifiers-order"]:
                    qualifiers += [
                        process_snak(s, labels) for s in claim["qualifiers"][q]
                    ]
            qualifiers = [q for q in qualifiers if q is not None]

            output["claims"].append(
                {
                    "property": claim["mainsnak"]["property"],
                    "property_label": labels[claim["mainsnak"]["property"]],
                    "value": obj,
                    "rank": claim["rank"],
                    "qualifiers": qualifiers,
                }
            )

    return output


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Wikidata JSON dump file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    labels: defaultdict[str, str | None] = defaultdict(lambda: None)
    for line in Path(args.data_path).open("r"):
        data = json.loads(line)
        qid = data["id"]
        labels[qid] = data["label"]
        if len(labels) > 100000000:
            break

    sys.stdin.readline()

    for line in sys.stdin:
        try:
            output = process_line(line, labels)
            if output is not None:
                print(json.dumps(output, ensure_ascii=False))
        except:
            continue

    # cat wikidata-20230620-all.json | uv run python kairos/data/dump.py --data_path ./wikidata-20230620-all.json > wikidata-20230620-processed.json
