import csv
import pickle
import json


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def save_csv(data, filename, fieldnames=None, flush=False):
    with open(filename, 'w') as f:
        if fieldnames is not None:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        else:
            writer = csv.writer(f)

        if not flush:
            writer.writerows(data)
        else:
            # write line by line and flush after each line
            for row in data:
                writer.writerow(row)
                f.flush()


def load_csv(filename, as_dict=True, delimiter=','):
    with open(filename, 'r') as f:
        if as_dict:
            reader = csv.DictReader(f, delimiter=delimiter)
        else:
            reader = csv.reader(f, delimiter=delimiter)

        rows = [r for r in reader]

    return rows


def load_lines(filename):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f]
        lines = [l for l in lines if len(l) != 0]

    return lines


def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)

    return obj


def save_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=2)


def load_jsonl(filename):
    with open(filename, 'r') as f:
        obj_list = [json.loads(line) for line in f]

    return obj_list


def save_jsonl(obj_list, filename):
    with open(filename, 'w') as f:
        for obj in obj_list:
            line = json.dumps(obj)
            f.write(line)
            f.write('\n')
