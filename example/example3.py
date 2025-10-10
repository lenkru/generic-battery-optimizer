"""
Example 2: Optimize a household energy system with a heat pump and a battery.
"""

# Additional installations needed: gurobipy

# Reads in profiles from CSV in example_data/example_profiles.csv
# PV from pvgis: location: latitude = 49.011861, longitude = 8.425412, tilt = 30, orientation: south, type: tmy, kwp: 10
# Temperature from meteostat: location: latitude = 49.011861, longitude = 8.425412, year 2024
# Heat demand: bdew mfh 2024 from demandlib (oemof) with anual consumption: 80000 kWh
# Electricity profile: bdew h25 2024 from demandlib (oemof) with anual consumption: 20000 kWh

import pandas as pd  # Useful to load time series for the indices
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from battery_optimizer.profiles.battery_profile import Battery
from battery_optimizer.profiles.heat_pump import HeatPump
from battery_optimizer.profiles.profiles import ProfileStack, PowerPriceProfile
from battery_optimizer.profile_stack_problem import ProfileStackProblem
from battery_optimizer.solver import Solver
from battery_optimizer.export.model import Exporter
import hplib.hplib as hpl
from battery_optimizer.static.heat_pump import C_TO_K

example_dir = Path(__file__).resolve().parent / "example_data"
df = pd.read_csv(example_dir / "example_profiles.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp")

# Chose time period that shall be optimized
index_df = df.index
df = df.loc[index_df[0:500]]

# Ensure last period has zero heat demand to avoid infeasibility
df.loc[df.index[-1], "heat_demand"] = 0.0

# Buy profiles
grid_buy_profile = PowerPriceProfile(
    index=df.index,
    price=df["grid_buy_profile_price"].values,
    power=df["grid_buy_profile_power"].values * 10000,
    name="Grid buy profile",
)

# PV profile
pv_profile = PowerPriceProfile(
    index=df.index,
    price=df["pv_profile_price"].values,
    power=df["pv_profile_power"].values,
    name="PV generation profile",
)

buy_profile_stack = ProfileStack([grid_buy_profile, pv_profile])

# Sell profile
grid_sell_profile = PowerPriceProfile(
    index=df.index,
    price=0,
    power=0,
    name="Grid sell profile",
)
sell_profile_stack = ProfileStack([grid_sell_profile])

# Fixed consumption profile
consumption_profile = PowerPriceProfile(
    index=df.index,
    power=df["consumption_profile_power"].values,
    name="Consumption profile",
)

consumption_profile_stack = ProfileStack([consumption_profile])

# Battery
household_battery = Battery(
    name="Household battery",
    start_soc=0.5,
    capacity=50000,
    max_charge_power=50000,
    max_discharge_power=50000,
)

# Heat Pump - Using Custom COP Data from Manufacturer
# NEW FEATURE: cop_data parameter allows using manufacturer COP values directly!
# When cop_data is provided, COP values are interpolated from this data instead of hplib.
# This gives more accurate results when you have precise manufacturer data.

cop_data_manufacturer = {
    'temp_sink': [25, 25, 25, 25, 25, 30, 30, 30, 30, 30, 35, 35, 35, 35, 35, 
                  40, 40, 40, 40, 40, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 
                  55, 55, 55, 55, 55, 65, 65, 65, 65, 65],
    'temp_source': [7, 2, -2, -7, -15, 7, 2, -2, -7, -15, 7, 2, -2, -7, -15,
                    7, 2, -2, -7, -15, 7, 2, -2, -7, -15, 7, 2, -2, -7, -15,
                    7, 2, -2, -7, -15, 7, 2, -2, -7, -15],
    'cop': [5.78, 4.71, 4.12, 3.16, 2.81, 5.06, 4.21, 3.79, 3.06, 2.67, 4.5, 3.81, 3.5, 3.02, 2.45,
            4.0, 3.46, 3.26, 2.89, 2.29, 3.55, 3.16, 3.05, 2.77, 2.14, 3.29, 2.8, 2.66, 2.51, 2.02,
            3.06, 2.5, 2.4, 2.29, 1.92, 2.27, 2.06, 1.94, 1.81, 1.62],
}

heat_pump = HeatPump(
    name="hp_daikin_custom_cop",
    type="Generic",  # Type still needed for some internal logic
    id=1,  # Air/Water regulated
    t_in=280.15,  # Reference source temperature: 7°C (required for Generic, but not used when cop_data provided)
    t_out=308.15,  # Reference sink temperature: 35°C
    p_th=11.6,  # Thermal power at reference: 11.6 kW
    cop_data=cop_data_manufacturer,  # ← NEW! Use manufacturer COP data
    flow_temperature=328.15,  # Heating circuit supply 55°C (radiators)
    output_temperature=333.15,  # Max HP outlet ~60°C
    min_electric_power_hp=0.0,  # allow HP to turn off completely
    max_electric_power_hp=10.0,  # max electric power for heat pump
    min_electric_power_hr=0.0,  # allow backup heater to be off
    max_electric_power_hr=12.0,  # max electric power for heating rod
    tank_volume=500,  # thermal energy storage (TES) volume [l]
    tes_start_soc=0.3,  # initial SoC of TES
    outdoor_temperature=df["source_temperature"].to_dict(),  # outdoor temperature
    heat_source_temperature=df["source_temperature"].to_dict(),  # heat source temperature = outdoor temperature
    heat_demand=df["heat_demand"].to_dict(),  # heat demand
    predict_tank_loss=True,  # predict tank loss
)

# Generate heat pump COP and power curves
print("Generating heat pump COP and power curves...")

# Create hplib heat pump object (if not using custom COP data)
if heat_pump.cop_data is not None:
    print("Note: Using custom manufacturer COP data instead of hplib for optimization")
    # For plotting, we still create an hplib model for comparison
    if heat_pump.type == "Generic":
        parameters = hpl.get_parameters(
            model=heat_pump.type,
            group_id=heat_pump.id,
            t_in=heat_pump.t_in - C_TO_K,
            t_out=heat_pump.t_out - C_TO_K,
            p_th=heat_pump.p_th / 1000,
        )
    else:
        parameters = hpl.get_parameters(model=heat_pump.type)
    hpl_heat_pump = hpl.HeatPump(parameters)
    print("  → COP values in optimization will be interpolated from manufacturer data")
    print("  → Plots will show hplib curves for comparison")
else:
    if heat_pump.type == "Generic":
        parameters = hpl.get_parameters(
            model=heat_pump.type,
            group_id=heat_pump.id,
            t_in=heat_pump.t_in - C_TO_K,
            t_out=heat_pump.t_out - C_TO_K,
            p_th=heat_pump.p_th / 1000,
        )
    else:
        parameters = hpl.get_parameters(model=heat_pump.type)
    hpl_heat_pump = hpl.HeatPump(parameters)

# Manufacturer's actual data for Daikin EPRA18DW (for comparison with Generic model)
manufacturer_data = {
    'temp_sink': [25, 25, 25, 25, 25, 30, 30, 30, 30, 30, 35, 35, 35, 35, 35, 
                  40, 40, 40, 40, 40, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 
                  55, 55, 55, 55, 55, 65, 65, 65, 65, 65],
    'temp_source': [7, 2, -2, -7, -15, 7, 2, -2, -7, -15, 7, 2, -2, -7, -15,
                    7, 2, -2, -7, -15, 7, 2, -2, -7, -15, 7, 2, -2, -7, -15,
                    7, 2, -2, -7, -15, 7, 2, -2, -7, -15],
    'cop': [5.78, 4.71, 4.12, 3.16, 2.81, 5.06, 4.21, 3.79, 3.06, 2.67, 4.5, 3.81, 3.5, 3.02, 2.45,
            4.0, 3.46, 3.26, 2.89, 2.29, 3.55, 3.16, 3.05, 2.77, 2.14, 3.29, 2.8, 2.66, 2.51, 2.02,
            3.06, 2.5, 2.4, 2.29, 1.92, 2.27, 2.06, 1.94, 1.81, 1.62],
    'power_thermal': [11.9, 12.3, 11.8, 11.6, 10.2, 11.8, 12.5, 11.9, 11.8, 10.4, 11.6, 12.7, 12.1, 12.7, 11.2,
                      11.8, 12.9, 12.2, 12.7, 11.5, 12.1, 13.0, 12.3, 12.7, 11.7, 12.2, 12.9, 12.9, 12.9, 11.9,
                      12.4, 12.7, 12.9, 12.1, 10.1, 11.8, 12.5, 12.2, 11.9, 11.3],
    'power_electric': [2.060, 2.620, 2.860, 3.660, 3.640, 2.320, 2.980, 3.150, 3.840, 3.900, 2.580, 3.340, 3.450, 4.190, 4.580,
                       2.950, 3.730, 3.740, 4.380, 5.010, 3.390, 4.120, 4.040, 4.580, 5.450, 3.720, 4.590, 4.840, 5.140, 5.890,
                       4.060, 5.070, 5.350, 5.700, 6.320, 5.200, 6.080, 6.290, 6.550, 6.980]
}

# Organize manufacturer data by sink temperature for plotting
mfg_data_by_sink = {}
for i in range(len(manufacturer_data['temp_sink'])):
    t_sink = manufacturer_data['temp_sink'][i]
    if t_sink not in mfg_data_by_sink:
        mfg_data_by_sink[t_sink] = {'source': [], 'cop': [], 'power_th': []}
    mfg_data_by_sink[t_sink]['source'].append(manufacturer_data['temp_source'][i])
    mfg_data_by_sink[t_sink]['cop'].append(manufacturer_data['cop'][i])
    mfg_data_by_sink[t_sink]['power_th'].append(manufacturer_data['power_thermal'][i])

# Define temperature ranges for plotting
source_temps_c = np.arange(-15, 21, 1)  # Source temperature range in °C
sink_temps_c = [35, 45, 55, 60]  # Sink temperatures in °C (common operating points)
ambient_temp_c = -7  # Standard ambient temperature for rating

# Store results for each sink temperature
cop_curves = {}
p_el_curves = {}
p_th_curves = {}

for t_sink_c in sink_temps_c:
    cops = []
    p_els = []
    p_ths = []
    
    for t_source_c in source_temps_c:
        try:
            result = hpl_heat_pump.simulate(
                t_in_primary=t_source_c,
                t_in_secondary=t_sink_c,
                t_amb=ambient_temp_c,
                mode=1
            )
            cops.append(result['COP'])
            p_els.append(result['P_el'])  # Electrical power in W
            p_ths.append(result['P_th'])  # Thermal power in W
        except Exception as e:
            # If simulation fails for certain conditions, use NaN
            cops.append(np.nan)
            p_els.append(np.nan)
            p_ths.append(np.nan)
    
    cop_curves[t_sink_c] = cops
    p_el_curves[t_sink_c] = p_els
    p_th_curves[t_sink_c] = p_ths

print("Heat pump curves generated successfully.")

# Build and solve optimization
opt = ProfileStackProblem(
    buy_prices=buy_profile_stack,
    sell_prices=sell_profile_stack,
    fixed_consumption=consumption_profile_stack,
    batteries=[household_battery],
    heat_pumps=[heat_pump],
)
opt.set_up()
Solver(
    solver="gurobi",
    options={
        "TimeLimit": 600,  # Time limit in seconds
        "MIPGap": 0.01,  # 1% optimality gap tolerance
        "OutputFlag": 1,  # Enable solver output
    },
    tee=True,
).solve(opt.model.model)

# Export results
export = Exporter(opt.model).to_df()
buy_power = export.to_buy()
sell_power = export.to_sell()
battery_power = export.to_battery_power()
battery_soc = export.to_battery_soc()
fixed_consumption = export.to_fixed_consumption()
heat_pump_power = export.to_heat_pump_power()

hp_block = opt.model.model.heat_pumps.component(heat_pump.name).hp_block
tes_soc = pd.Series({
    t: hp_block[t].soc.value for t in df.index
}, name="TES SoC")
print("Optimization successful - plotting results")

# Create mask for first two days
mask_2d = df.index[:48]

# PLOT 1: Demands and Generation (first two days)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(mask_2d, (df.loc[mask_2d, "heat_demand"] * 1000).values, label='Heat Demand (thermal)', linestyle='-',
         linewidth=2, color='darkorange')
ax1.plot(mask_2d, df.loc[mask_2d, "consumption_profile_power"].values, label='Electricity Demand', linestyle='-',
         linewidth=2, color='blue')
ax1.plot(mask_2d, pv_profile.power[mask_2d], label='PV Power', linestyle='-',
         linewidth=2, color='green')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power [W]')
ax1.set_title('Demands and Generation (First 2 Days)')
ax1.legend()
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 2: Device Powers (first two days)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(mask_2d, battery_power.loc[mask_2d, 'Household battery'].values, label='Battery Power', linestyle='-',
         linewidth=2, color='purple')
ax2.plot(mask_2d, buy_power.loc[mask_2d].sum(axis=1).values, label='Buy Power (total)', linestyle='-',
         linewidth=2, color='red')
ax2.plot(mask_2d, heat_pump_power.loc[mask_2d, heat_pump.name].values, label='Heat Pump Power', linestyle='-',
         linewidth=2, color='teal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Power [W]')
ax2.set_title('Device Powers (First 2 Days)')
ax2.legend()
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 3: Battery SOC, TES SOC with Grid Buy Price (first two days)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(mask_2d, battery_soc.loc[mask_2d, 'Household battery'].values, label='Battery SOC', linestyle='-',
         linewidth=2, color='purple')
ax3.plot(mask_2d, [tes_soc.loc[idx] for idx in mask_2d], label='TES SOC', linestyle='-', linewidth=2, color='orange')
ax3.set_xlabel('Time')
ax3.set_ylabel('SOC [0-1]')
ax3_twin = ax3.twinx()
ax3_twin.plot(mask_2d, df.loc[mask_2d, "grid_buy_profile_price"].values, label='Grid Buy Price', linestyle='-', linewidth=2, color='darkred')
ax3_twin.set_ylabel('Price [Cent/kWh]', color='darkred')
ax3_twin.tick_params(axis='y', labelcolor='darkred')
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax3.set_title('Storage SOCs and Grid Buy Price (First 2 Days)')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 4: Heat Pump COP Curves
fig4 = plt.figure(figsize=(12, 6))
ax4 = fig4.add_subplot(111)

# Define colors for each sink temperature
colors = ['blue', 'green', 'orange', 'red']
for i, (t_sink_c, color) in enumerate(zip(sink_temps_c, colors)):
    # Plot hplib simulated curves
    ax4.plot(source_temps_c, cop_curves[t_sink_c], 
             label=f'hplib Sink = {t_sink_c}°C', 
             linestyle='-', linewidth=2, color=color, alpha=0.7)
    
    # Plot manufacturer data points if available for this sink temperature
    if t_sink_c in mfg_data_by_sink:
        ax4.scatter(mfg_data_by_sink[t_sink_c]['source'], 
                   mfg_data_by_sink[t_sink_c]['cop'],
                   label=f'Manufacturer Sink = {t_sink_c}°C',
                   marker='o', s=80, color=color, edgecolors='black', linewidths=1.5, alpha=0.5, zorder=5)

ax4.set_xlabel('Source Temperature [°C]')
ax4.set_ylabel('Coefficient of Performance (COP)')
ax4.set_title(f'Heat Pump COP vs Source Temperature\n({heat_pump.type})\nSolid lines: hplib simulation | Dots: Manufacturer data')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
plt.tight_layout()

# PLOT 5: Heat Pump Thermal Power Curves
fig5 = plt.figure(figsize=(12, 6))
ax5 = fig5.add_subplot(111)

for i, (t_sink_c, color) in enumerate(zip(sink_temps_c, colors)):
    # Convert from W to kW
    p_th_kw = [p/1000 if not np.isnan(p) else np.nan for p in p_th_curves[t_sink_c]]
    # Plot hplib simulated curves
    ax5.plot(source_temps_c, p_th_kw, 
             label=f'hplib Sink = {t_sink_c}°C', 
             linestyle='-', linewidth=2, color=color, alpha=0.7)
    
    # Plot manufacturer data points if available for this sink temperature
    if t_sink_c in mfg_data_by_sink:
        ax5.scatter(mfg_data_by_sink[t_sink_c]['source'], 
                   mfg_data_by_sink[t_sink_c]['power_th'],
                   label=f'Manufacturer Sink = {t_sink_c}°C',
                   marker='o', s=80, color=color, edgecolors='black', linewidths=1.5, alpha=0.5, zorder=5)

ax5.set_xlabel('Source Temperature [°C]')
ax5.set_ylabel('Thermal Power [kW]')
ax5.set_title(f'Heat Pump Thermal Power vs Source Temperature\n({heat_pump.type})\nSolid lines: hplib simulation | Dots: Manufacturer data')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)
plt.tight_layout()

# Show plots
plt.show()