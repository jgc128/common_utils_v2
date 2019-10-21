import random
import dataclasses
import tempfile
from pathlib import Path

from common_utils.config import Config
from common_utils.io import save_json, load_json


def generate_name():
    """
    Based on Docker code: https://github.com/moby/moby/blob/master/pkg/namesgenerator/names-generator.go
    """

    left = [
        "admiring", "adoring", "affectionate", "agitated", "amazing", "angry", "awesome", "beautiful", "blissful",
        "bold", "boring", "brave", "busy", "charming", "clever", "cool", "compassionate", "competent", "condescending",
        "confident", "cranky", "crazy", "dazzling", "determined", "distracted", "dreamy", "eager", "ecstatic",
        "elastic", "elated", "elegant", "eloquent", "epic", "exciting", "fervent", "festive", "flamboyant", "focused",
        "friendly", "frosty", "funny", "gallant", "gifted", "goofy", "gracious", "great", "happy", "hardcore",
        "heuristic", "hopeful", "hungry", "infallible", "inspiring", "interesting", "intelligent", "jolly", "jovial",
        "keen", "kind", "laughing", "loving", "lucid", "magical", "mystifying", "modest", "musing", "naughty",
        "nervous", "nice", "nifty", "nostalgic", "objective", "optimistic", "peaceful", "pedantic", "pensive",
        "practical", "priceless", "quirky", "quizzical", "recursing", "relaxed", "reverent", "romantic", "sad",
        "serene", "sharp", "silly", "sleepy", "stoic", "strange", "stupefied", "suspicious", "sweet", "tender",
        "thirsty", "trusting", "unruffled", "upbeat", "vibrant", "vigilant", "vigorous", "wizardly", "wonderful",
        "xenodochial", "youthful", "zealous", "zen",
    ]

    right = [
        "albattani", "allen", "almeida", "antonelli", "agnesi", "archimedes", "ardinghelli", "aryabhata", "austin",
        "babbage", "banach", "banzai", "bardeen", "bartik", "bassi", "beaver", "bell", "benz", "bhabha", "bhaskara",
        "black", "blackburn", "blackwell", "bohr", "booth", "borg", "bose", "bouman", "boyd", "brahmagupta",
        "brattain", "brown", "buck", "burnell", "cannon", "carson", "cartwright", "cerf", "chandrasekhar", "chaplygin",
        "chatelet", "chatterjee", "chebyshev", "cohen", "chaum", "clarke", "colden", "cori", "cray", "curran", "curie",
        "darwin", "davinci", "dewdney", "dhawan", "diffie", "dijkstra", "dirac", "driscoll", "dubinsky", "easley",
        "edison", "einstein", "elbakyan", "elgamal", "elion", "ellis", "engelbart", "euclid", "euler", "faraday",
        "feistel", "fermat", "fermi", "feynman", "franklin", "gagarin", "galileo", "galois", "ganguly", "gates",
        "gauss", "germain", "goldberg", "goldstine", "goldwasser", "golick", "goodall", "gould", "greider",
        "grothendieck", "haibt", "hamilton", "haslett", "hawking", "hellman", "heisenberg", "hermann", "herschel",
        "hertz", "heyrovsky", "hodgkin", "hofstadter", "hoover", "hopper", "hugle", "hypatia", "ishizaka", "jackson",
        "jang", "jennings", "jepsen", "johnson", "joliot", "jones", "kalam", "kapitsa", "kare", "keldysh", "keller",
        "kepler", "khayyam", "khorana", "kilby", "kirch", "knuth", "kowalevski", "lalande", "lamarr", "lamport",
        "leakey", "leavitt", "lederberg", "lehmann", "lewin", "lichterman", "liskov", "lovelace", "lumiere",
        "mahavira", "margulis", "matsumoto", "maxwell", "mayer", "mccarthy", "mcclintock", "mclaren", "mclean",
        "mcnulty", "mendel", "mendeleev", "meitner", "meninsky", "merkle", "mestorf", "minsky", "mirzakhani", "moore",
        "morse", "murdock", "moser", "napier", "nash", "neumann", "newton", "nightingale", "nobel", "noether",
        "northcutt", "noyce", "panini", "pare", "pascal", "pasteur", "payne", "perlman", "pike", "poincare", "poitras",
        "proskuriakova", "ptolemy", "raman", "ramanujan", "ride", "montalcini", "ritchie", "rhodes", "robinson",
        "roentgen", "rosalind", "rubin", "saha", "sammet", "sanderson", "satoshi", "shamir", "shannon", "shaw",
        "shirley", "shockley", "shtern", "sinoussi", "snyder", "solomon", "spence", "stallman", "stonebraker",
        "sutherland", "swanson", "swartz", "swirles", "taussig", "tereshkova", "tesla", "tharp", "thompson",
        "torvalds", "tu", "turing", "varahamihira", "vaughan", "visvesvaraya", "volhard", "villani", "wescoff",
        "wilbur", "wiles", "williams", "williamson", "wilson", "wing", "wozniak", "wright", "wu", "yalow", "yonath",
        "zhukovsky",
    ]

    name = f"{random.choice(left)}_{random.choice(right)}_{random.randint(100, 999)}"

    return name


class Experiment(object):
    _CONFIG_FILENAME = 'config.json'

    def __init__(self, base_dir=None, config=None, prefix_attrs=None):
        if base_dir is not None and not isinstance(base_dir, Path):
            base_dir = Path(base_dir)

        experiment_id = generate_name()
        if config is not None:
            prefix = Config.get_name(config)

            if prefix_attrs is not None:
                prefix_attrs_str = '_'.join(f'{getattr(config, attr)}' for attr in prefix_attrs)
                prefix = f'{prefix}.{prefix_attrs_str}'

            experiment_id = f'{prefix}.{experiment_id}'

        if base_dir is not None:
            experiment_dir = base_dir.joinpath(experiment_id)
            if experiment_dir.exists():
                raise ValueError(f'Experiment {experiment_id} already exists')

            experiment_dir.mkdir()

            if config is not None:
                Experiment._save_config(config, experiment_dir)
        else:
            experiment_dir = None

        self.config = config
        self.base_dir = base_dir
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id

    @classmethod
    def load(cls, base_dir, experiment_id):
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        experiment_dir = base_dir.joinpath(experiment_id)
        experiment_id = experiment_dir.name
        config = Experiment._load_config(experiment_dir)

        exp = Experiment()
        exp.config = config
        exp.base_dir = base_dir
        exp.experiment_dir = experiment_dir
        exp.experiment_id = experiment_id

        return exp

    @classmethod
    def _save_config(cls, config, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        config_dict = dataclasses.asdict(config)
        type_name = Config.get_name(config)

        config_json = {
            'type': type_name,
            'params': config_dict,
        }

        save_json(config_json, filename)

    @classmethod
    def _load_config(cls, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        config_json = load_json(filename)

        type_name = config_json['type']
        config_dict = config_json['params']

        config = Config.by_name(type_name)(**config_dict)

        return config
