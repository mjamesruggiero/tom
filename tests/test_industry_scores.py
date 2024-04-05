import os
import pytest
import logging
import datetime
import json

from .context import tom
from tom import industry_scores


logging.basicConfig(level=logging.DEBUG, format="%(lineno)d\t%(message)s")

@pytest.fixture
def ranking_payload():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    claude_file = os.path.join(dir_path, 'data/claude_ranking.txt')
    with open(claude_file, 'r+') as f:
        data = f.read()
    return data


def test_claude_json_can_be_extracted(ranking_payload) -> None:
    expected = [ {
        "ticker": "CPRI",
        "ranking": 1,
        "currentPrice": 43.74,
        "targetPrice": 35.0,
        "rationale": "Despite negative sentiment and downward earnings revisions, Capri Holdings' current price of $43.74 appears overvalued given the significant headwinds facing the luxury goods industry, including slowing economic growth, geopolitical tensions, and changing consumer preferences. The competitive landscape and cyclical risks further threaten CPRI's ability to maintain its market position and profitability. A target price of $35.0 reflects these challenges and the potential for further downside. Investors should consider selling CPRI and reallocating capital to more promising opportunities."
        }]
    payload = json.loads(industry_scores.extract_claude_json(ranking_payload))
    assert payload == expected

def test_mk_datestring_returns_predictable_string() -> None:
    expected = "2020_11_18"
    d = datetime.datetime(2020, 11, 18)
    assert industry_scores.mk_datestring(d) == expected
