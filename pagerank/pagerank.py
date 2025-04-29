import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page and a damping factor.

    corpus: dict where keys are page names and values are sets of linked pages
    page: the current page (string)
    damping_factor: probability of following a link (float)
    """
    N = len(corpus)                # total number of pages
    links = corpus.get(page)       # set of pages linked from current page
    dist = {}                      # will hold the probability for each page

    # 1) Handle "dead ends": if no outgoing links, treat it as linking to every page
    if not links:
        for p in corpus:
            dist[p] = 1 / N
        return dist

    # 2) Teleport probability: with (1 - d), jump to any page uniformly
    for p in corpus:
        dist[p] = (1 - damping_factor) / N

    # 3) Link-following probability: with d, follow one of the outgoing links uniformly
    link_prob = damping_factor / len(links)
    for linked_page in links:
        dist[linked_page] += link_prob

    return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Estimate PageRank values by sampling n pages according to transition_model.

    corpus: dict mapping each page to its set of links
    damping_factor: as above
    n: number of samples to draw
    """
    # 1) Initialize visit counts to zero
    counts = {p: 0 for p in corpus}

    # 2) Pick the first sample uniformly at random
    pages = list(corpus.keys())
    current = random.choice(pages)
    counts[current] += 1

    # 3) Draw the remaining n-1 samples
    for _ in range(1, n):
        # a) Get the transition probabilities from the current page
        dist = transition_model(corpus, current, damping_factor)
        # b) Randomly choose the next page according to those weights
        next_page = random.choices(
            population=list(dist.keys()),
            weights=list(dist.values()),
            k=1
        )[0]
        counts[next_page] += 1
        current = next_page

    # 4) Convert counts to probabilities (i.e., estimated PageRank)
    pagerank = {p: counts[p] / n for p in counts}
    return pagerank


def iterate_pagerank(corpus, damping_factor, threshold=0.001):
    """
    Compute PageRank values by iteratively applying the PageRank formula
    until no page's rank changes by more than the threshold.

    corpus: dict mapping each page to its set of links
    damping_factor: as above
    threshold: convergence threshold (default 0.001)
    """
    N = len(corpus)
    # 1) Start with equal rank for every page
    pagerank = {p: 1 / N for p in corpus}

    # 2) Build a reverse lookup: for each page, which pages link to it?
    incoming = {p: set() for p in corpus}
    for p, links in corpus.items():
        if links:
            for dest in links:
                incoming[dest].add(p)
        else:
            # Treat dead-end pages as linking to every page
            for dest in corpus:
                incoming[dest].add(p)

    # 3) Iteratively update until convergence
    converged = False
    while not converged:
        converged = True
        new_ranks = {}

        for p in corpus:
            # a) Sum contributions from all pages that link to p
            total = 0
            for linker in incoming[p]:
                # Determine how many links 'linker' actually has
                num_links = len(corpus[linker]) if corpus[linker] else N
                total += pagerank[linker] / num_links

            # b) Apply the PageRank formula
            new_rank = (1 - damping_factor) / N + damping_factor * total

            # c) Check for convergence
            if abs(new_rank - pagerank[p]) > threshold:
                converged = False

            new_ranks[p] = new_rank

        pagerank = new_ranks

    # 4) Final normalization (to correct any tiny rounding drift)
    total = sum(pagerank.values())
    for p in pagerank:
        pagerank[p] /= total

    return pagerank


if __name__ == "__main__":
    main()
