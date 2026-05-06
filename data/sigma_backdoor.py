import itertools
import networkx as nx


def powerset(nodes):
    nodes = list(nodes)
    for r in range(len(nodes) + 1):
        yield from itertools.combinations(nodes, r)


def scc_map(G):
    comp_id = {}
    for k, comp in enumerate(nx.strongly_connected_components(G)):
        for node in comp:
            comp_id[node] = k
    return comp_id


def same_scc(comp_id, a, b):
    return comp_id[a] == comp_id[b]


def ancestors_of_set(G, S):
    out = set(S)
    for s in S:
        out |= nx.ancestors(G, s)
    return out


def is_collider_on_path(G, path, i):
    """
    Zi is a collider on path if:
        Zi-1 -> Zi <- Zi+1
    """
    prev_node = path[i - 1]
    node = path[i]
    next_node = path[i + 1]

    return G.has_edge(prev_node, node) and G.has_edge(next_node, node)


def sigma_blocked_path(G, path, S, comp_id=None):
    """
    Implements sigma-blocking from the definition.

    A path is sigma-blocked by S if some internal node Zi satisfies either:

    1. Zi is a collider and Zi not in Anc(S)

    or

    2. Zi is a non-collider, Zi in S, and at least one outgoing edge
       along the path leaves Zi's strongly connected component:
          Zi -> Zi+1 and Zi+1 notin SCC(Zi)
       or
          Zi -> Zi-1 and Zi-1 notin SCC(Zi)
    """
    S = set(S)

    if comp_id is None:
        comp_id = scc_map(G)

    anc_S = ancestors_of_set(G, S)

    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        node = path[i]
        next_node = path[i + 1]

        collider = is_collider_on_path(G, path, i)

        if collider:
            if node not in anc_S:
                return True

        else:
            if node in S:
                exits_scc_forward = G.has_edge(node, next_node) and not same_scc(
                    comp_id, node, next_node
                )

                exits_scc_backward = G.has_edge(node, prev_node) and not same_scc(
                    comp_id, node, prev_node
                )

                if exits_scc_forward or exits_scc_backward:
                    return True

    return False


def sigma_separated(G, X, Y, S):
    """
    X and Y are sigma-separated by S iff every path between X and Y
    is sigma-blocked by S.
    """
    S = set(S)
    comp_id = scc_map(G)

    UG = G.to_undirected()

    for path in nx.all_simple_paths(UG, X, Y):
        if not sigma_blocked_path(G, path, S, comp_id=comp_id):
            return False

    return True


def is_backdoor_path(G, path, treatment):
    """
    A backdoor path from X to Y starts with an arrow into X:

        X <- ...
    """
    if len(path) < 2:
        return False

    first_neighbor = path[1]
    return G.has_edge(first_neighbor, treatment)


class SigmaBackdoorChecker:
    """
    Reusable checker for many candidate adjustment sets on one graph/X/Y pair.

    The fast path searches reachable open path states instead of enumerating all
    simple paths. Traversal uses the undirected skeleton, while collider and SCC
    checks still use the original directed graph.
    """

    def __init__(self, G, treatment, outcome):
        self.G = G
        self.treatment = treatment
        self.outcome = outcome
        self.comp_id = scc_map(G)
        self.UG = G.to_undirected()
        self.neighbors = {
            node: tuple(self.UG.neighbors(node))
            for node in self.UG.nodes()
        }
        self.descendants_treatment = nx.descendants(G, treatment)
        self._ancestor_cache = {}
        self._validity_cache = {}

    def ancestors_of(self, S):
        out = set(S)
        for node in S:
            if node not in self._ancestor_cache:
                self._ancestor_cache[node] = nx.ancestors(self.G, node)
            out |= self._ancestor_cache[node]
        return out

    def sigma_triple_blocked(self, prev_node, node, next_node, S, anc_S):
        collider = self.G.has_edge(prev_node, node) and self.G.has_edge(
            next_node, node
        )

        if collider:
            return node not in anc_S

        if node in S:
            exits_scc_forward = self.G.has_edge(node, next_node) and not same_scc(
                self.comp_id, node, next_node
            )

            exits_scc_backward = self.G.has_edge(node, prev_node) and not same_scc(
                self.comp_id, node, prev_node
            )

            return exits_scc_forward or exits_scc_backward

        return False

    def sigma_blocked_path(self, path, S, anc_S=None):
        if anc_S is None:
            anc_S = self.ancestors_of(S)

        for i in range(1, len(path) - 1):
            if self.sigma_triple_blocked(
                path[i - 1], path[i], path[i + 1], S, anc_S
            ):
                return True

        return False

    def has_open_sigma_backdoor_path_fast(self, S):
        S = set(S)
        anc_S = self.ancestors_of(S)
        stack = []
        seen = set()

        for neighbor in self.neighbors.get(self.treatment, ()):
            if not self.G.has_edge(neighbor, self.treatment):
                continue

            state = (self.treatment, neighbor)
            if state not in seen:
                seen.add(state)
                stack.append(state)

        while stack:
            prev_node, node = stack.pop()

            if node == self.outcome:
                return True

            for next_node in self.neighbors.get(node, ()):
                if next_node == prev_node or next_node == self.treatment:
                    continue

                if self.sigma_triple_blocked(prev_node, node, next_node, S, anc_S):
                    continue

                state = (node, next_node)
                if state in seen:
                    continue

                seen.add(state)
                stack.append(state)

        return False

    def blocks_all_backdoor_paths_fast(self, S):
        return not self.has_open_sigma_backdoor_path_fast(S)

    def blocks_all_backdoor_paths_slow(self, S):
        S = set(S)
        anc_S = self.ancestors_of(S)

        for path in nx.all_simple_paths(self.UG, self.treatment, self.outcome):
            if not is_backdoor_path(self.G, path, self.treatment):
                continue

            if not self.sigma_blocked_path(path, S, anc_S=anc_S):
                return False

        return True

    def blocks_all_backdoor_paths(self, S, use_fast=True):
        if use_fast:
            return self.blocks_all_backdoor_paths_fast(S)
        return self.blocks_all_backdoor_paths_slow(S)

    def is_valid(self, S, use_fast=True):
        S = set(S)
        cache_key = (frozenset(S), bool(use_fast))
        if cache_key in self._validity_cache:
            return self._validity_cache[cache_key]

        if self.treatment in S or self.outcome in S:
            self._validity_cache[cache_key] = False
            return False

        if S & self.descendants_treatment:
            self._validity_cache[cache_key] = False
            return False

        is_valid = self.blocks_all_backdoor_paths(S, use_fast=use_fast)
        self._validity_cache[cache_key] = is_valid
        return is_valid


def has_open_sigma_backdoor_path_fast(G, treatment, outcome, S):
    """
    Return True if at least one open backdoor path exists.
    """
    return SigmaBackdoorChecker(
        G, treatment, outcome
    ).has_open_sigma_backdoor_path_fast(S)


def sigma_blocks_all_backdoor_paths(G, treatment, outcome, S, use_fast=True):
    """
    Checks whether S sigma-blocks every backdoor path from treatment to outcome.
    """
    return SigmaBackdoorChecker(G, treatment, outcome).blocks_all_backdoor_paths(
        S, use_fast=use_fast
    )


def sigma_blocks_all_backdoor_paths_slow(G, treatment, outcome, S):
    """
    Checks whether S sigma-blocks every backdoor path using path enumeration.
    """
    return sigma_blocks_all_backdoor_paths(
        G, treatment, outcome, S, use_fast=False
    )


def is_valid_sigma_backdoor_set(G, treatment, outcome, S, use_fast=True):
    """
    Sigma-backdoor criterion for a directed possibly cyclic graph.

    A set S is valid if:
    1. no node in S is a descendant of treatment
    2. S sigma-blocks every backdoor path from treatment to outcome
    """
    return SigmaBackdoorChecker(G, treatment, outcome).is_valid(
        S, use_fast=use_fast
    )


def all_valid_sigma_backdoor_sets(
    G, treatment, outcome, minimal_only=False, use_fast=True
):
    """
    Exhaustively enumerate all valid sigma-backdoor adjustment sets.
    """
    checker = SigmaBackdoorChecker(G, treatment, outcome)
    descendants_t = checker.descendants_treatment

    candidates = set(G.nodes()) - {treatment, outcome} - descendants_t

    valid_sets = []

    for S_tuple in powerset(candidates):
        S = set(S_tuple)

        if checker.is_valid(S, use_fast=use_fast):
            valid_sets.append(frozenset(S))

    if minimal_only:
        valid_sets = [S for S in valid_sets if not any(S2 < S for S2 in valid_sets)]

    return [set(S) for S in valid_sets]


def exists_valid_sigma_backdoor_set(G, treatment, outcome, use_fast=True):
    """
    Returns True iff at least one valid sigma-backdoor set exists.
    """
    checker = SigmaBackdoorChecker(G, treatment, outcome)
    descendants_t = checker.descendants_treatment

    candidates = set(G.nodes()) - {treatment, outcome} - descendants_t

    for S_tuple in powerset(candidates):
        if checker.is_valid(set(S_tuple), use_fast=use_fast):
            return True

    return False
