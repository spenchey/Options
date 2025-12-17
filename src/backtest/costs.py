"""
Transaction Costs Module

Models commissions, fees, and slippage for backtesting.
"""

from dataclasses import dataclass
from typing import Optional
from .config import COST_PARAMS


@dataclass
class TradeCost:
    """Cost breakdown for a single trade."""
    commissions: float      # Broker commissions
    exchange_fees: float    # Exchange/regulatory fees
    slippage: float         # Estimated market impact
    total: float            # Total cost


class CostModel:
    """
    Transaction cost model for options and stocks.

    Default parameters reflect realistic costs for 2011-2013 period:
    - Options: ~$1.00 commission + $0.25 fees per contract
    - Slippage: Model as percentage of bid-ask spread captured
    """

    def __init__(
        self,
        commission_per_contract: float = COST_PARAMS['commission_per_contract'],
        fee_per_contract: float = COST_PARAMS['fee_per_contract'],
        spread_capture: float = COST_PARAMS['spread_capture'],
        spy_commission_per_share: float = COST_PARAMS['spy_commission_per_share']
    ):
        """
        Args:
            commission_per_contract: Broker commission per option contract
            fee_per_contract: Exchange/regulatory fees per contract
            spread_capture: Fraction of bid-ask spread captured (0.5 = mid)
            spy_commission_per_share: Cost per share for SPY hedges
        """
        self.commission_per_contract = commission_per_contract
        self.fee_per_contract = fee_per_contract
        self.spread_capture = spread_capture
        self.spy_commission_per_share = spy_commission_per_share

    def option_trade_cost(
        self,
        contracts: int,
        bid: float,
        ask: float,
        is_opening: bool = True
    ) -> TradeCost:
        """
        Calculate cost of an option trade.

        Args:
            contracts: Number of contracts (positive for buy, negative for sell)
            bid: Option bid price
            ask: Option ask price
            is_opening: Whether this is opening or closing trade

        Returns:
            TradeCost with breakdown
        """
        abs_contracts = abs(contracts)

        # Commissions and fees
        commissions = abs_contracts * self.commission_per_contract
        exchange_fees = abs_contracts * self.fee_per_contract

        # Slippage from bid-ask spread
        # When selling: we get bid + (spread_capture * spread)
        # When buying: we pay ask - ((1-spread_capture) * spread)
        spread = ask - bid
        if contracts < 0:  # Selling
            # We get less than mid by (1 - spread_capture) * spread
            slippage = abs_contracts * 100 * (1 - self.spread_capture) * spread
        else:  # Buying
            # We pay more than mid by (1 - spread_capture) * spread
            slippage = abs_contracts * 100 * (1 - self.spread_capture) * spread

        total = commissions + exchange_fees + slippage

        return TradeCost(
            commissions=commissions,
            exchange_fees=exchange_fees,
            slippage=slippage,
            total=total
        )

    def strangle_trade_cost(
        self,
        contracts: int,
        put_bid: float,
        put_ask: float,
        call_bid: float,
        call_ask: float,
        is_opening: bool = True
    ) -> TradeCost:
        """
        Calculate cost of a strangle trade (2 legs).

        Args:
            contracts: Number of strangles
            put_bid, put_ask: Put option prices
            call_bid, call_ask: Call option prices
            is_opening: Whether this is opening or closing trade

        Returns:
            TradeCost for the strangle
        """
        put_cost = self.option_trade_cost(contracts, put_bid, put_ask, is_opening)
        call_cost = self.option_trade_cost(contracts, call_bid, call_ask, is_opening)

        return TradeCost(
            commissions=put_cost.commissions + call_cost.commissions,
            exchange_fees=put_cost.exchange_fees + call_cost.exchange_fees,
            slippage=put_cost.slippage + call_cost.slippage,
            total=put_cost.total + call_cost.total
        )

    def spy_trade_cost(self, shares: int) -> TradeCost:
        """
        Calculate cost of SPY hedge trade.

        Args:
            shares: Number of shares (positive for buy, negative for sell)

        Returns:
            TradeCost for SPY trade
        """
        abs_shares = abs(shares)
        commissions = abs_shares * self.spy_commission_per_share

        # Assume minimal slippage for SPY (very liquid)
        # Typical spread is $0.01, assume we get mid
        slippage = abs_shares * 0.005  # Half penny per share

        return TradeCost(
            commissions=commissions,
            exchange_fees=0.0,  # Negligible for stocks
            slippage=slippage,
            total=commissions + slippage
        )

    def estimate_execution_price(
        self,
        bid: float,
        ask: float,
        is_buy: bool
    ) -> float:
        """
        Estimate realistic execution price given bid-ask spread.

        Args:
            bid: Bid price
            ask: Ask price
            is_buy: True if buying, False if selling

        Returns:
            Estimated fill price
        """
        spread = ask - bid
        mid = (bid + ask) / 2

        if is_buy:
            # Pay above mid
            return mid + (1 - self.spread_capture) * spread / 2
        else:
            # Receive below mid
            return mid - (1 - self.spread_capture) * spread / 2

    def calculate_credit_received(
        self,
        contracts: int,
        put_bid: float,
        put_ask: float,
        call_bid: float,
        call_ask: float
    ) -> float:
        """
        Calculate net credit for selling a strangle after costs.

        Args:
            contracts: Number of strangles to sell (should be positive)
            put_bid, put_ask: Put prices
            call_bid, call_ask: Call prices

        Returns:
            Net credit received after transaction costs
        """
        # Gross credit at estimated execution prices
        put_fill = self.estimate_execution_price(put_bid, put_ask, is_buy=False)
        call_fill = self.estimate_execution_price(call_bid, call_ask, is_buy=False)
        gross_credit = contracts * 100 * (put_fill + call_fill)

        # Subtract costs
        costs = self.strangle_trade_cost(
            -contracts,  # Selling
            put_bid, put_ask,
            call_bid, call_ask,
            is_opening=True
        )

        return gross_credit - costs.total

    def calculate_close_cost(
        self,
        contracts: int,
        put_bid: float,
        put_ask: float,
        call_bid: float,
        call_ask: float
    ) -> float:
        """
        Calculate total cost to close a short strangle position.

        Args:
            contracts: Number of strangles to close
            put_bid, put_ask: Put prices
            call_bid, call_ask: Call prices

        Returns:
            Total cost to close (premium paid + transaction costs)
        """
        # Gross debit to buy back
        put_fill = self.estimate_execution_price(put_bid, put_ask, is_buy=True)
        call_fill = self.estimate_execution_price(call_bid, call_ask, is_buy=True)
        gross_debit = contracts * 100 * (put_fill + call_fill)

        # Add costs
        costs = self.strangle_trade_cost(
            contracts,  # Buying
            put_bid, put_ask,
            call_bid, call_ask,
            is_opening=False
        )

        return gross_debit + costs.total


def summarize_costs(
    trades: list,
    cost_model: Optional[CostModel] = None
) -> dict:
    """
    Summarize transaction costs from a list of trades.

    Args:
        trades: List of TradeCost objects
        cost_model: CostModel instance (for reference)

    Returns:
        Summary dict with totals and averages
    """
    if not trades:
        return {
            'total_commissions': 0,
            'total_fees': 0,
            'total_slippage': 0,
            'total_costs': 0,
            'num_trades': 0
        }

    total_commissions = sum(t.commissions for t in trades)
    total_fees = sum(t.exchange_fees for t in trades)
    total_slippage = sum(t.slippage for t in trades)

    return {
        'total_commissions': total_commissions,
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'total_costs': total_commissions + total_fees + total_slippage,
        'num_trades': len(trades),
        'avg_cost_per_trade': (total_commissions + total_fees + total_slippage) / len(trades)
    }


if __name__ == '__main__':
    print("Cost Model Test")
    print("=" * 50)

    model = CostModel()

    # Example: Sell 10 strangles
    print("\nExample: Sell 10 strangles")
    print("  Put: bid=$1.50, ask=$1.60")
    print("  Call: bid=$1.40, ask=$1.50")

    cost = model.strangle_trade_cost(
        contracts=-10,  # Selling
        put_bid=1.50, put_ask=1.60,
        call_bid=1.40, call_ask=1.50
    )

    print(f"\n  Commissions: ${cost.commissions:.2f}")
    print(f"  Exchange fees: ${cost.exchange_fees:.2f}")
    print(f"  Slippage: ${cost.slippage:.2f}")
    print(f"  Total cost: ${cost.total:.2f}")

    # Net credit
    credit = model.calculate_credit_received(
        contracts=10,
        put_bid=1.50, put_ask=1.60,
        call_bid=1.40, call_ask=1.50
    )
    print(f"\n  Gross credit (at mid): ${10 * 100 * (1.55 + 1.45):.2f}")
    print(f"  Net credit after costs: ${credit:.2f}")

    # SPY hedge
    print("\n\nExample: Buy 500 SPY shares to hedge")
    spy_cost = model.spy_trade_cost(500)
    print(f"  Commissions: ${spy_cost.commissions:.2f}")
    print(f"  Slippage: ${spy_cost.slippage:.2f}")
    print(f"  Total: ${spy_cost.total:.2f}")
