from battery_optimizer.profiles.profiles import PowerPriceProfile, ProfileStack
import pandas as pd


# Method to create the profiles
def get_profiles(
    time_series: pd.DatetimeIndex, profiles: dict[str, pd.DataFrame]
):
    """Create a ProfileStack from a dictionary of profiles.

    Input price goes first, then input power."""
    opt_profiles = []
    for key, value in profiles.items():
        opt_profiles.append(
            PowerPriceProfile(
                index=time_series,
                price=value["input_price"],
                power=value["input_power"],
                name=key,
            )
        )
    return ProfileStack(opt_profiles)
