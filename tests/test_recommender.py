import pandas as pd
from src.models.recommender import rank_content, score_content

def _catalog():
    return pd.DataFrame([
        # good for persona 2 (meditation/low/short)
        {"content_id":"c1","type":"meditation","duration_min":10,"intensity":"low","difficulty":"beginner","goal_tag":"stress","popularity":0.1},
        # neutral
        {"content_id":"c2","type":"walk","duration_min":20,"intensity":"medium","difficulty":"beginner","goal_tag":"fitness","popularity":0.0},
        # good for persona 1 (hiit/high/25)
        {"content_id":"c3","type":"hiit","duration_min":25,"intensity":"high","difficulty":"advanced","goal_tag":"fitness","popularity":0.0},
    ])

def test_score_prefers_persona_profile():
    row = _catalog().iloc[0]
    s = score_content(row, user_goal="stress", persona=2)
    # same row but mismatch persona
    s_worse = score_content(row, user_goal="stress", persona=1)
    assert s > s_worse

def test_ranker_filters_by_goal_and_falls_back():
    df = _catalog()
    # goal 'stress' should pick c1
    out = rank_content(df, user_goal="stress", persona=2, top_k=2)
    assert out.iloc[0]["content_id"] == "c1"
    # unknown goal → fallback to whole catalog, still ranked
    out2 = rank_content(df, user_goal="unknown", persona=1, top_k=2)
    assert set(out2["content_id"]) <= {"c1","c2","c3"}
    assert "score" in out2.columns
