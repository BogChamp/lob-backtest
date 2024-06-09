from abc import ABC, abstractmethod
from typing import Tuple, Optional
from lobio.lob.limit_order import Order

class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_bid_ask_price() -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def bid_ask_limit_orders() -> Tuple[list[Order], list[Order]]:
        pass

