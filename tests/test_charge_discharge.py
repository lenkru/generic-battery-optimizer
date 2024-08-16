from helpers import get_profiles
from battery_optimizer.profiles.battery_profile import Battery
from battery_optimizer import optimize
from datetime import datetime
import pandas as pd


class TestChargeDischarge:
    def test_buy_sell(self):
        # Test Data
        time_series = pd.DatetimeIndex(
            [
                datetime(2021, 1, 1, 8, 0, 0),
                datetime(2021, 1, 1, 9, 0, 0),
                datetime(2021, 1, 1, 10, 0, 0),
            ]
        )

        fixed_consumption = {
            "fixed_consumption": pd.DataFrame(
                data={
                    "input_power": [0, 0, 0],
                    "input_price": [0, 0, 0],
                },
                index=time_series,
            )
        }

        # Expected output
        # 1. Zeitschritt ist Buy Preis attraktiv (alles kaufen) - Preis = 1
        buy = {
            "buy": pd.DataFrame(
                data={
                    "input_power": [10, 10, 0],
                    "input_price": [1, 4, 0],
                },
                index=time_series,
            )
        }

        # 2. Zeitschritt ist Sell Preis attraktiv (alles verkaufen) - Preis = 1
        sell = {
            "sell": pd.DataFrame(
                data={
                    "input_power": [10, 10, 0],
                    "input_price": [0, 3, 0],
                },
                index=time_series,
            )
        }

        battery = Battery(
            name="test-battery",
            start_soc=0,
            end_soc=0,
            capacity=10000,
            max_charge_power=10000,
            max_discharge_power=10000,
            charge_efficiency=1,
            discharge_efficiency=1,
        )

        # Optimization
        result = optimize(
            buy_prices=get_profiles(time_series, buy),
            sell_prices=get_profiles(time_series, sell),
            fixed_consumption=get_profiles(time_series, fixed_consumption),
            batteries=[battery],
        )

        result_batteries = pd.DataFrame(
            data={
                "test-battery": [10, -10, 0],
            },
            index=time_series,
        )

        fixed_consumption_result = pd.DataFrame(
            data={
                "fixed_consumption": [0, 0, 0],
            },
            index=time_series,
        )

        buy_result = pd.DataFrame(
            data={
                "buy": [10, 0, 0],
            },
            index=time_series,
        )

        sell_result = pd.DataFrame(
            data={
                "sell": [0, 10, 0],
            },
            index=time_series,
        )

        # Assert power profiles
        pd.testing.assert_frame_equal(result[0], buy_result, check_dtype=False)
        pd.testing.assert_frame_equal(
            result[1], sell_result, check_dtype=False)
        pd.testing.assert_frame_equal(
            result[4], fixed_consumption_result, check_dtype=False
        )

        # Assert battery profiles
        pd.testing.assert_frame_equal(
            result[2], result_batteries, check_dtype=False)

    # def test_buy_from_negative_price(self):
