"""
루트 레벨 BotManager.
앱 시작 시 모든 봇을 로드하고 키로 꺼내 씀.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))
sys.path.insert(0, str(ROOT / 'src' / 'bupa'))
sys.path.insert(0, str(ROOT / 'src' / 'tricare'))
sys.path.insert(0, str(ROOT / 'src' / 'allianz'))
sys.path.insert(0, str(ROOT / 'src' / 'cigna'))

from insurance_rag  import InsuranceRAGGraph
from bupa_plugin    import BupaPlugin
from tricare_plugin import TriCarePlugin
from allianz_plugin import AllianzPlugin
from cigna_plugin   import CignaPlugin

# 사이드바 표시용 라벨 → 내부 키 매핑
INSURER_OPTIONS = {
    "🏥 Bupa 국제 의료보험":    "bupa",
    "🪖 TRICARE 군인 의료보험": "tricare",
    "🌍 Allianz Care":          "allianz",
    "🩺 Cigna Global":          "cigna",
}


class BotManager:
    """싱글턴 봇 매니저"""

    _instance: BotManager | None = None

    def __new__(cls) -> BotManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("🚀 전체 DB 로드 시작...")
        self._bots = {
            "bupa":    InsuranceRAGGraph(BupaPlugin()).build(),
            "tricare": InsuranceRAGGraph(TriCarePlugin()).build(),
            "allianz": InsuranceRAGGraph(AllianzPlugin()).build(),
            "cigna":   InsuranceRAGGraph(CignaPlugin()).build(),
        }
        self._initialized = True
        print("✅ 전체 봇 로드 완료\n")

    def get(self, key: str):
        bot = self._bots.get(key.lower())
        if bot is None:
            raise ValueError(f"❌ '{key}' 봇 없음. 사용 가능: {list(self._bots.keys())}")
        return bot
