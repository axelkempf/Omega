# ui_engine/datafeeds/dxfeed_manager.py

from ui_engine.datafeeds.base import BaseDatafeedManager


class DXFeedDatafeedManager(BaseDatafeedManager):
    def start(self) -> bool:
        print(f"[{self.alias}] DXFeed-Start simuliert")
        return True

    def stop(self) -> bool:
        print(f"[{self.alias}] DXFeed-Stop simuliert")
        return True

    def status(self) -> str:
        return "Running"
