"""Bounded evidence adjustments around the original retrieval order."""


def evidence_order(base_order, lookup, promotion_limit=6, demotion_limit=6):
    base=list(dict.fromkeys(aid for aid in base_order if aid in lookup))
    positions={aid:i for i,aid in enumerate(base)}
    def key(aid):
        result=lookup[aid]
        offset=0
        if not result.get("inspection_uncertain"):
            if result.get("fact_conflict"):offset=min(12,demotion_limit*2)
            elif result.get("inspection_strict_conflict"):offset=demotion_limit
            elif result.get("inspection_tier")==2:offset=-promotion_limit
        # Semantic disagreement alone cannot demote; these ordinal offsets are
        # neither match probabilities nor exclusions or global tiers.
        return (positions[aid]+offset,positions[aid])
    return sorted(base,key=key)
