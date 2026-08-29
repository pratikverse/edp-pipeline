"""classify/probe.py -- the linear-probe evidence source (docs/08 Phase 7).
Doesn't require the actual trained artifact (outputs/probe/ is a build
output, not committed) -- exercises the "no artifact yet" abstain path
directly, and the loaded-model path with a tiny stub classifier."""
import numpy as np

from edp.classify.probe import probe_evidence


def test_abstains_when_no_model_file():
    ev = probe_evidence(np.zeros(768, dtype=np.float32), model_path="outputs/does_not_exist/probe.joblib")
    assert not ev.has_evidence


def test_returns_topk_class_scores_from_a_real_model(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=i, scale=0.1, size=(5, 8)) for i in range(3)])
    y = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
    clf = LogisticRegression().fit(X, y)
    model_path = tmp_path / "probe.joblib"
    joblib.dump(clf, model_path)

    query = rng.normal(loc=1, scale=0.1, size=8).astype(np.float32)  # near class "B"'s cluster
    ev = probe_evidence(query, model_path=str(model_path), top_k=2)
    assert ev.has_evidence
    assert len(ev.class_scores) == 2
    assert max(ev.class_scores, key=ev.class_scores.get) == "B"
