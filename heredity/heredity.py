import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability that:
      - each person in one_gene has exactly 1 copy,
      - each person in two_genes has exactly 2 copies,
      - everyone else has 0 copies,
      - each person in have_trait exhibits the trait,
      - everyone else does not.
    """
    # Start with probability = 1 (we will multiply each person’s contribution)
    probability = 1.0

    # Loop over each person in the family
    for person, info in people.items():
        # 1) Figure out how many copies this person has in this scenario:
        if person in two_genes:
            copies = 2
        elif person in one_gene:
            copies = 1
        else:
            copies = 0

        # 2) Compute probability they have that many copies:
        mother = info["mother"]
        father = info["father"]

        if mother is None and father is None:
            # No parental info? Use the unconditional prior.
            gene_prob = PROBS["gene"][copies]
        else:
            # They have parents listed → compute inheritance + mutation
            # Helper: probability parent passes on a mutated copy
            def pass_prob(parent):
                # Determine how many copies the parent has
                if parent in two_genes:
                    parent_copies = 2
                elif parent in one_gene:
                    parent_copies = 1
                else:
                    parent_copies = 0

                # If parent has 2 copies, they pass mutated with prob 1 - mutation
                # If 0 copies, they pass mutated only via mutation
                # If 1 copy, 50/50 chance
                if parent_copies == 2:
                    return 1 - PROBS["mutation"]
                elif parent_copies == 1:
                    return 0.5
                else:  # parent_copies == 0
                    return PROBS["mutation"]

            # Compute probabilities mom→child and dad→child
            mom_pass = pass_prob(mother)
            dad_pass = pass_prob(father)

            # If child has 2 mutated copies, they must get one from each parent
            if copies == 2:
                gene_prob = mom_pass * dad_pass
            # If child has 1 mutated copy, exactly one parent passed it
            elif copies == 1:
                gene_prob = mom_pass * (1 - dad_pass) + (1 - mom_pass) * dad_pass
            # If child has 0 mutated copies, neither parent passed one
            else:  # copies == 0
                gene_prob = (1 - mom_pass) * (1 - dad_pass)

        # 3) Compute probability they exhibit (or not) the trait given their copies
        has_trait = person in have_trait
        trait_prob = PROBS["trait"][copies][has_trait]

        # 4) Multiply into our running joint probability
        probability *= gene_prob * trait_prob

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add the joint probability p into `probabilities` for each person.
    - If person in two_genes, add to probabilities[person]["gene"][2], etc.
    - If person in have_trait, add to probabilities[person]["trait"][True], etc.
    """
    for person in probabilities:
        # 1) Which gene-count bucket?
        if person in two_genes:
            copies = 2
        elif person in one_gene:
            copies = 1
        else:
            copies = 0

        # 2) Which trait bucket?
        has_trait = person in have_trait

        # 3) Accumulate the joint probability p
        probabilities[person]["gene"][copies] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Convert each person’s distributions into proper probabilities
    (so each sums to 1) by dividing by the total.
    """
    for person_data in probabilities.values():
        # Normalize the gene distribution (0,1,2)
        gene_total = sum(person_data["gene"].values())
        for copies in person_data["gene"]:
            person_data["gene"][copies] /= gene_total

        # Normalize the trait distribution (True, False)
        trait_total = sum(person_data["trait"].values())
        for val in person_data["trait"]:
            person_data["trait"][val] /= trait_total


if __name__ == "__main__":
    main()
