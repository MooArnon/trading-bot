##########
# Import #
##############################################################################

from abc import abstractmethod
import datetime

import pandas as pd

###########
# Classes #
##############################################################################

class BaseBot:
    def __init__(
            self,
            initial_balance: float = 1000.0,
    ):
        self.set_present_balace = initial_balance

    ##########################################################################
    
    def get_signal_mapper(self) -> dict:
        return {
            -1: "SHORT",
            0: "HOLD",
            1: "LONG",
        }
    
    ##########################################################################
    
    def present_balace(self) -> float:
        return self.__present_balace
    
    ##########################################################################
    
    def set_present_balace(self, present_balace: float) -> float:
        self.__present_balace = present_balace
    
    ##########################################################################

    @abstractmethod
    def search(self, *args, **kawrgs) -> None:
        raise NotImplementedError("Child class must implement seach method")
    
    ##########################################################################
    
    @abstractmethod
    def running_simulation(self, *args, **kawrgs) -> None:
        raise NotImplementedError(
            "Child class must implement running_simulation method"
        )
        
    ##########################################################################
    
    @staticmethod
    def truncate_to_quarter_hour(dt: datetime.datetime) -> datetime.datetime:
        """
        Truncates the given datetime's minutes down to the nearest
        quarter-hour (00, 15, 30, or 45).
        
        Seconds and microseconds are also reset to 0.
        """
        # Use integer division to find the 15-minute block
        # For example:
        # 14 // 15 = 0  -> 0 * 15 = 0
        # 25 // 15 = 1  -> 1 * 15 = 15
        # 46 // 15 = 3  -> 3 * 15 = 45
        new_minute = (dt.minute // 15) * 15
        
        # Return a new datetime object with 
        # truncated minutes, seconds, and microseconds
        return dt.replace(minute=new_minute, second=0, microsecond=0)
    
    ##########################################################################
    
    @abstractmethod
    def get_data(self, *args, **kawrgs) -> None:
        raise NotImplementedError("Child class must implement get_data method")
    
    ##########################################################################
    
    @abstractmethod
    def trasform(self, *args, **kawrgs) -> pd.Series:
        raise NotImplementedError("Child class must implement trasform method")
    
    ##########################################################################
    
##############################################################################
