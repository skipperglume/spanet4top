from itertools import chain, combinations, starmap
from spanet.dataset.types import *
from collections import OrderedDict
from yaml import load as yaml_load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from sympy.combinatorics import Permutation as SymbolicPermutation

# Possible types of permutation that the user can input
RawPermutation = Union[
    List[List[str]],  # Explicit
    List[str]  # Complete Group
]

def construct_mapping(variables: Iterable[str]) -> ODict[str, int]:
        return OrderedDict(map(reversed, enumerate(variables)))

def apply_mapping(permutations: Permutations, mapping: Dict[str, int]) -> MappedPermutations:
        return [
            [
                tuple(
                    mapping[element]
                    for element in cycle
                )
                for cycle in permutation
            ]
            for permutation in permutations
        ]

def expand_permutations(permutations: List[RawPermutation]) -> Permutations:
    expanded_permutations = []
    for permutation in permutations:
        if isinstance(permutation[0], list):
            expanded_permutations.append([tuple(p) for p in permutation])
        else:
            expanded_permutations.extend([[tuple(p)] for p in combinations(permutation, 2)])
    return expanded_permutations
def with_default(value, default):
    return default if value is None else value

def key_with_default(database, key, default):
    if key not in database:
        return default

    value = database[key]
    return default if value is None else value

def read_from_yaml(filename: str):
    with open(filename, 'r') as file:
        config = yaml_load(file, Loader)

    # Extract input feature information.
    # ----------------------------------
    input_types = OrderedDict()
    input_features = OrderedDict()

    for input_type in config[SpecialKey.Inputs]:
        current_inputs = with_default(config[SpecialKey.Inputs][input_type], default={})

        for input_name, input_information in current_inputs.items():
            input_types[input_name] = input_type.upper()
            input_features[input_name] = tuple(
                FeatureInfo(
                    name=name,
                    normalize=("normalize" in normalize.lower()) or ("true" in normalize.lower()),
                    log_scale="log" in normalize.lower()
                )

                for name, normalize in input_information.items()
            )

    # Extract event and permutation information.
    # ------------------------------------------
    permutation_config = key_with_default(config, SpecialKey.Permutations, default={})
    print('Permutations in config:')
    print(permutation_config)

    event_names = tuple(config[SpecialKey.Event].keys())
    print('Event names == Particle names:')
    print(event_names)
    event_permutations = key_with_default(permutation_config, SpecialKey.Event, default=[])
    print('Particle symmetries:')
    print(event_permutations)
    event_permutations = expand_permutations(event_permutations)
    print('Particle symmetries expanded:')
    print(event_permutations)
    event_particles = Particles(event_names, event_permutations)
    
    product_particles = OrderedDict()
    for event_particle in event_particles:
        products = config[SpecialKey.Event][event_particle]

        product_names = [
            next(iter(product.keys())) if isinstance(product, dict) else product
            for product in products
        ]

        product_sources = [
            next(iter(product.values())) if isinstance(product, dict) else None
            for product in products
        ]

        input_names = list(input_types.keys())
        product_sources = [
            input_names.index(source) if source is not None else -1
            for source in product_sources
        ]

        product_permutations = key_with_default(permutation_config, event_particle, default=[])
        product_permutations = expand_permutations(product_permutations)
        print(product_permutations)
        product_particles[event_particle] = Particles(product_names, product_permutations, product_sources)

    # self.product_mappings: ODict[str, ODict[str, int]] = OrderedDict()
    # self.product_symmetries: ODict[str, Symmetries] = OrderedDict()
    product_mappings: ODict[str, ODict[str, int]] = OrderedDict()
    product_symmetries: ODict[str, Symmetries] = OrderedDict()
    for event_particle, product_particles in product_particles.items():
        product_mapping = construct_mapping(product_particles)
        print(event_particle )
        print('  ',product_particles.names, product_particles.permutations, product_particles.sources)
        print('  ', product_mapping)
        product_mappings[event_particle] = product_mapping
        product_symmetries[event_particle] = Symmetries(
            len(product_particles),
            apply_mapping(product_particles.permutations, product_mapping)
        )
        print('  Degree', product_symmetries[event_particle].degree)
        print('  Symmetries', product_symmetries[event_particle].permutations)
    # # Extract Regression Information.
    # # -------------------------------
    # regressions = key_with_default(config, SpecialKey.Regressions, default={})
    # regressions = feynman_fill(regressions, event_particles, product_particles, constructor=list)

    # # Fill in any default parameters for regressions such as gaussian type.
    # regressions = feynman_map(
    #     lambda raw_regressions: [
    #         RegressionInfo(*(regression if isinstance(regression, list) else [regression]))
    #         for regression in raw_regressions
    #     ],
    #     regressions
    # )

    # # Extract Classification Information.
    # # -----------------------------------
    # classifications = key_with_default(config, SpecialKey.Classifications, default={})
    # classifications = feynman_fill(classifications, event_particles, product_particles, constructor=list)

    # return cls(
    #     input_types,
    #     input_features,
    #     event_particles,
    #     product_particles,
    #     regressions,
    #     classifications
    # )

if __name__ == '__main__':
    eventFilePath = '/home/timoshyd/spanet4Top/SPANet/event_files/all_had_4top/'
    eventFileName = 'all_had_4top_test.yaml'
    read_from_yaml(eventFilePath+eventFileName)
    exit(0)