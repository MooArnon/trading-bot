##########
# Import #
##############################################################################

from .__base import BaseMarket

###########
# Classes #
##############################################################################

class BinanceMarket(BaseMarket):
    def __init__(self):
        pass

    ##########################################################################

    def get_data(
            self, 
            timeframe: str, 
            period: int,
    ) -> None:
        """The mandatory method to get data from market

        Parameters
        ----------
        timeframe: str
            Timeframe to get data
        period : int
            Number of data point in timeframe
            
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Child class must implement seach method")
    
    ##########################################################################

##############################################################################
