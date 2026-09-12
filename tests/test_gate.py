from basinrag.eval.gate import decide


def test_decide_convert_c_when_topology_loses():
    scifact = {
        "hybrid_min": {"ndcg@10": 0.60},
        "hybrid_min_topo": {"ndcg@10": 0.601},
    }
    longdoc = {
        "encoder_pure": {"ndcg@10": 0.50},
        "hybrid_min": {"ndcg@10": 0.48},
        "hybrid_min_topo": {"ndcg@10": 0.47},
    }
    basins = {"gate_geometry_ok": True, "median_members": 6.0, "singleton_frac": 0.2}
    out = decide(scifact, longdoc, basins)
    assert out["DECISION"] == "CONVERT_C"


def test_decide_continue_b_when_topology_wins():
    scifact = {
        "hybrid_min": {"ndcg@10": 0.60},
        "hybrid_min_topo": {"ndcg@10": 0.601},
    }
    longdoc = {
        "encoder_pure": {"ndcg@10": 0.40},
        "hybrid_min": {"ndcg@10": 0.41},
        "hybrid_min_topo": {"ndcg@10": 0.43},
        "basinrag_full": {"ndcg@10": 0.44},
    }
    basins = {"gate_geometry_ok": True, "median_members": 8.0, "singleton_frac": 0.1}
    out = decide(scifact, longdoc, basins)
    assert out["DECISION"] == "CONTINUE_B"


def test_decide_invalid_when_basins_too_small():
    out = decide(
        {"hybrid_min": {"ndcg@10": 0.5}, "hybrid_min_topo": {"ndcg@10": 0.5}},
        {"encoder_pure": {"ndcg@10": 0.4}, "hybrid_min": {"ndcg@10": 0.4}, "hybrid_min_topo": {"ndcg@10": 0.5}},
        {"gate_geometry_ok": False, "median_members": 1.0, "singleton_frac": 1.0},
    )
    assert out["DECISION"] == "GATE_INVALID"


def test_decide_warns_on_scifact_leakage():
    out = decide(
        {"hybrid_min": {"ndcg@10": 0.50}, "hybrid_min_topo": {"ndcg@10": 0.53}},
        None,
        None,
    )
    assert out["DECISION"] == "GATE_INVALID"
    assert any("leakage" in w for w in out["warnings"])
