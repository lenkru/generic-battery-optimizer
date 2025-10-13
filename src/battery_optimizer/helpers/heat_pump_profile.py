import datetime
import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


def tank_dimensions(volume: int | float):
    """Calculate height and radius of a tank from the volume

    This function calculates the radius and height of a tank
    with a given volume based on typical tank dimensions.
    Warm water tanks usually have a height to diameter ratio of 2:1 to 3:1.
    The mean diameter is determined by the volume and the height is calculated.

    Arguments
    ---------
    volume: int | float
        The volume of the tank in m^3

    Returns
    -------
    radius: float
        The radius of the tank in m
    height: float
        The height of the tank in m
    """
    if volume <= 0:
        return 0, 0

    h2r = (volume / (4 * np.pi)) ** (1 / 3)
    h3r = (volume / (6 * np.pi)) ** (1 / 3)

    radius = (h2r + h3r) / 2

    height = volume / (np.pi * radius**2)
    return radius, height


def heat_loss_tank(
    height: int | float,
    radius: int | float,
    u_value_material: int | float,
    tempretaure_difference: int | float,
) -> float:
    """Transmission heat losses of the tank

    Calculates the transmission heat losses of the tank based on the
    temperature difference between the tank inside and outside.

    Arguments
    ---------
    height: int | float
        The height of the tank in m
    radius: int | float
        The radius of the tank in m
    u_value_material: int | float
        The U-value of the insulation material of the tank in W/(m^2*K)
        This is usually between 0.3 and 0.7 W/(m^2*K)
    tempretaure_difference: int | float
        The temperature difference between the tank and the room temperature
        in K

    Returns
    -------
    float
        The heat loss of the tank in kW
    """
    return (
        (2 * np.pi * radius * (radius + height))
        * u_value_material
        * tempretaure_difference
    ) / 1000


def reverse_resolution(scale: int | float):
    """
    Funktion, welche aus einem int/float Wert, welcher die Periodenlänge in Stunden enthält einen String erzeugt,
    der die Periodendauer beschreibt

    scale:      int/float, welcher Periodenläng in h enthält

    Rückgabe:
    string, welcher die Periodenlänge enthält --> Bsp.: 0.25 --> "15Min"
    """

    value_Min = int(scale * 60)
    res_string = f"{value_Min}Min"
    return res_string


def interpolate_temperature(
    temperature: float | dict[datetime.datetime, float],
    current_period: datetime.datetime,
):
    """Returns the temperature for the current period.

    If the temperature is a float, it is returned as is.
    If the temperature is a dictionary, the temperature for the current period
    is returned either by the exact key if available or it is linearly
    interpolated between the two closest keys.

    ----------
    Variables:

    temperature: float | dict[datetime.datetime, float]
        The temperature for the optimization duration

    current_period: datetime.datetime
        The current period for which the temperature should be returned

    --------
    Returns:

    temperature: float
        The temperature for the current period in K"""
    # Just return the temperature if it is a float
    if isinstance(temperature, float):
        return temperature
    # Return exact temperature if available
    elif current_period in temperature:
        return temperature[current_period]
    # Interpolate temperature if not available
    else:
        temperature[current_period] = None
        series = pd.Series(temperature).sort_index().interpolate(method="time")
        return series[current_period]


def interpolate_heat_energy(
    heat_demand: float | dict[datetime.datetime, float],
    current_period: datetime.datetime,
) -> float:
    """Returns the heat energy demand for the current period.

    If the temperature is a float it is assumed to be a constant heat demand
    in each period in kW.
    """
    if not heat_demand:
        return 0
    # Just return the heat demand if it is a float
    if isinstance(heat_demand, float):
        return heat_demand
    # Return exact heat demand if available
    if current_period in heat_demand:
        return heat_demand[current_period]
    # Interpolate heat demand if not available
    df = pd.Series(heat_demand)
    df[current_period] = None
    df.interpolate(method="time", inplace=True)
    return df[current_period]


def fit_cop_parameters_from_manufacturer_data(cop_data: dict) -> dict:
    """Fit hplib COP parameters (p1-p4) from manufacturer COP data.

    This function uses least-squares regression to fit the hplib COP model
    parameters from manufacturer COP data. The hplib COP equation is:
    COP = p1 * T_source + p2 * T_sink + p3 + p4 * T_amb

    where T_source and T_sink are temperatures in °C, and for Air/Water 
    heat pumps, T_amb = T_source.

    Arguments
    ---------
    cop_data : dict
        Dictionary containing manufacturer COP data with keys:
        - 'temp_source': list of source temperatures in °C
        - 'temp_sink': list of sink temperatures in °C
        - 'cop': list of COP values corresponding to the temperature pairs

    Returns
    -------
    dict
        Dictionary containing the fitted parameters:
        - 'p1_COP [-]': float
        - 'p2_COP [-]': float
        - 'p3_COP [-]': float
        - 'p4_COP [-]': float
        - 'fit_rmse': Root mean square error of the fit
        - 'fit_mape': Mean absolute percentage error of the fit

    Raises
    ------
    ValueError
        If cop_data doesn't contain the required keys or if fitting fails
    """
    required_keys = {'temp_source', 'temp_sink', 'cop'}
    if not all(key in cop_data for key in required_keys):
        raise ValueError(
            f"cop_data must contain all keys: {required_keys}"
        )

    # Extract data
    t_source = np.array(cop_data['temp_source'])
    t_sink = np.array(cop_data['temp_sink'])
    cop_values = np.array(cop_data['cop'])

    if len(t_source) != len(t_sink) or len(t_source) != len(cop_values):
        raise ValueError(
            "All lists in cop_data must have the same length"
        )

    if len(cop_values) < 4:
        raise ValueError(
            "cop_data must contain at least 4 data points for fitting"
        )

    # hplib COP equation: COP = p1 * T_source + p2 * T_sink + p3 + p4 * T_amb
    # For Air/Water heat pumps, T_amb = T_source
    def hplib_cop_function(temps, p1, p2, p3, p4):
        """hplib COP function for curve fitting"""
        t_in, t_out = temps
        t_amb = t_in  # For Air/Water heat pumps
        return p1 * t_in + p2 * t_out + p3 + p4 * t_amb

    try:
        # Fit parameters using least-squares regression
        popt, pcov = curve_fit(
            hplib_cop_function,
            (t_source, t_sink),
            cop_values,
            p0=[0.1, -0.1, 5.0, -0.1],  # initial guess
            maxfev=10000
        )

        p1, p2, p3, p4 = popt

        # Calculate fit quality metrics
        cop_predicted = hplib_cop_function((t_source, t_sink), *popt)
        rmse = float(np.sqrt(np.mean((cop_values - cop_predicted)**2)))
        mape = float(np.mean(np.abs((cop_values - cop_predicted) / cop_values)) * 100)

        log.info(
            f"Fitted COP parameters from manufacturer data: "
            f"p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}, p4={p4:.6f} "
            f"(RMSE={rmse:.4f}, MAPE={mape:.2f}%)"
        )

        return {
            'p1_COP [-]': float(p1),
            'p2_COP [-]': float(p2),
            'p3_COP [-]': float(p3),
            'p4_COP [-]': float(p4),
            'fit_rmse': rmse,
            'fit_mape': mape,
        }

    except Exception as e:
        raise ValueError(
            f"Failed to fit COP parameters from manufacturer data: {e}"
        )
