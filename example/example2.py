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
from battery_optimizer.profiles.battery_profile import Battery
from battery_optimizer.profiles.heat_pump import HeatPump
from battery_optimizer.profiles.profiles import ProfileStack, PowerPriceProfile
from battery_optimizer.profile_stack_problem import ProfileStackProblem
from battery_optimizer.solver import Solver
from battery_optimizer.export.model import Exporter

example_dir = Path(__file__).resolve().parent / "example_data"
df = pd.read_csv(example_dir / "example_profiles.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp")

# Chose time period that shall be optimized
index_df = df.index
df = df.loc[index_df[0:100]]

# Ensure last period has zero heat demand to avoid infeasibility
df.loc[df.index[-1], "heat_demand"] = 0.0
df["grid_buy_profile_price"] = df["grid_buy_profile_price"] - 20


# Buy profiles
grid_buy_profile = PowerPriceProfile(
    index=df.index,  # [DatetimeIndex] Time periods
    price=df["grid_buy_profile_price"].values,  # [ct/kWh, cents per kilowatt-hour] Electricity price
    power=df["grid_buy_profile_power"].values * 1000000,  # [W, watts] Maximum power available from grid
    name="Grid buy profile",  # [string] Profile identifier
)

# PV profile (disabled for testing)
pv_profile = PowerPriceProfile(
    index=df.index,  # [DatetimeIndex] Time periods
    price=0,  # [ct/kWh] PV generation price (0 = no cost)
    power=0,  # [W, watts] PV generation power (0 = disabled)
    name="PV generation profile",  # [string] Profile identifier
)

buy_profile_stack = ProfileStack([grid_buy_profile, pv_profile])

# Sell profile
grid_sell_profile = PowerPriceProfile(
    index=df.index,  # [DatetimeIndex] Time periods
    price=0,  # [ct/kWh] Feed-in tariff (0 = no selling)
    power=0,  # [W, watts] Maximum power to sell (0 = no selling)
    name="Grid sell profile",  # [string] Profile identifier
)
sell_profile_stack = ProfileStack([grid_sell_profile])

# Fixed consumption profile

consumption_profile = PowerPriceProfile(
    index=df.index,  # [DatetimeIndex] Time periods
    power=df["consumption_profile_power"].values,  # [W, watts] Fixed electricity consumption
    name="Consumption profile",  # [string] Profile identifier
)

consumption_profile_stack = ProfileStack([consumption_profile])

# Battery (very small for testing TES - essentially disabled)
household_battery = Battery(
    name="Household battery",  # [string] Battery identifier
    start_soc=0.5,  # [0-1, unitless] Initial state of charge (50%)
    capacity=0.01,  # [Wh, watt-hours] Battery capacity (1 Wh = essentially disabled)
    max_charge_power=0.01,  # [W, watts] Maximum charging power
    max_discharge_power=0.01,  # [W, watts] Maximum discharging power
)

# Heat Pump - Vaillant VWL 55/5 (Outdoor Air/Water)
# Temperature hierarchy:
#   flow_temperature (30°C) < output_temperature (50°C) < max_temp_tes (60°C)
#   HP can charge TES from 30°C to 50°C (66.7% of total TES capacity)
#   Heating rod can charge from 50°C to 60°C (remaining 33.3%)
#   TES starts at 0% SOC (30°C) so HP can charge it
#   Large tank (2000L) provides ~2.8 hours of heat storage
heat_pump = HeatPump(
    # === REQUIRED PARAMETERS ===
    name="hp_vaillant_vwl55",  # [string] Unique identifier for the heat pump
    type="VWL 55/5 AS 230V + VWL 57/5 IS",  # [string] Specific Vaillant heat pump model from hplib database
    
    # Note: Generic HP parameters (id, t_in, t_out, p_th) are NOT needed for specific models
    # The model's COP characteristics are loaded from the hplib database
    
    # Temperature settings:
    flow_temperature=303.15,  # [K, Kelvin] Heating circuit supply temp (30.0°C) - MINIMUM usable temperature
    output_temperature=323.15,  # [K, Kelvin] Max HP outlet temp (50.0°C) - maximum temp HP can produce
    
    # Electric power limits:
    max_electric_power_hp=50.0,  # [kW, kilowatts] Maximum electric power consumption of heat pump
    max_electric_power_hr=30.0,  # [kW, kilowatts] Maximum electric power consumption of heating rod/backup heater
    min_electric_power_hp=0,  # [kW, kilowatts] Minimum electric power when HP is on (0 = can turn off completely)
    min_electric_power_hr=0,  # [kW, kilowatts] Minimum electric power when heater is on (0 = can turn off completely)
    
    # Thermal Energy Storage (TES) settings:
    tank_volume=2000,  # [L, liters] Volume of thermal energy storage tank (increased for better time-shifting)
    tes_start_soc=0.0,  # [0-1, unitless] Initial TES state of charge (0=empty at flow_temp, 1=full at max_temp_tes)
    max_temp_tes=333.15,  # [K, Kelvin] Maximum TES temperature (60°C) - HR can heat beyond HP output temp
    
    # Time-series data (required - must be dicts with timezone-aware datetime keys):
    outdoor_temperature=df["source_temperature"].to_dict(),  # [K, Kelvin] Outdoor/ambient temperature
    heat_source_temperature=df["source_temperature"].to_dict(),  # [K, Kelvin] Heat source temp (air/water/brine temp)
    heat_demand=df["heat_demand"].to_dict(),  # [kW, kilowatts] Building heat demand (constant per period)
    
    # Tank loss prediction:
    predict_tank_loss=False,  # [bool] Enable/disable tank heat loss calculation based on tank dimensions
    # tank_u_value=0.6,  # [W/(m²·K)] U-Value of tank insulation (default: 0.6) - only used if predict_tank_loss=True
    
    # === OPTIONAL PARAMETERS (commented out, showing defaults) ===
    # cop_air=3,  # [float, unitless] COP for Air/Air heat pumps (required when type="Air/Air")
    # temp_room=293.15,  # [K, Kelvin] OR dict - Room temperature (20°C) - can be time-varying dict
    # hp_switch_off_temperature=None,  # [K, Kelvin] Outdoor temp at which HP automatically turns off
    # bivalent_temp=None,  # [K, Kelvin] Outdoor temp below which HP provides only 70% of demand (rest from heater)
    # warm_water_demand=None,  # [kW, kilowatts] dict - Domestic hot water demand (constant per period)
    # enforce_end_soc=False,  # [bool] If True, TES must end at same SOC as it started (cyclical constraint)
    
    # Building heat demand estimation (alternative to providing heat_demand directly):
    # u_values_building=None,  # [dict] Wall/roof/window areas [m²] and U-values [W/(m²·K)]
    # living_area=None,  # [m², square meters] Living area - used with u_values_building to estimate heat demand
)

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
        "TimeLimit": 600,  # [seconds] Time limit in seconds
        "MIPGap": 0.01,  # [0-1, unitless] 1% optimality gap tolerance
        "OutputFlag": 1,  # [0-1, unitless] Enable solver output
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

# Extract COP values (using cop_low as the actual COP value)
cop_values = pd.Series({
    t: hp_block[t].cop_low.value for t in df.index
}, name="COP")

# Extract thermal storage power flows and losses
tes_input_power_hp = pd.Series({
    t: hp_block[t].heat_supply_hp_to_tes.value for t in df.index
}, name="TES Input from HP [kW]")

tes_input_power_hr = pd.Series({
    t: hp_block[t].heat_supply_HR.value for t in df.index
}, name="TES Input from Heating Rod [kW]")

tes_output_power = pd.Series({
    t: hp_block[t].heat_supply_TES_Demand.value for t in df.index
}, name="TES Output Power [kW]")

tes_losses = pd.Series({
    t: hp_block[t].heat_loss_tank.value for t in df.index
}, name="TES Losses [kW]")

# Extract HP heat distribution
hp_heat_to_demand = pd.Series({
    t: hp_block[t].heat_supply_hp_to_demand.value for t in df.index
}, name="HP Heat to Demand [kW]")

hp_heat_total = pd.Series({
    t: hp_block[t].heat_supply_hp_total.value for t in df.index
}, name="HP Total Heat [kW]")

print("Optimization successful - plotting results")

# Diagnostic: Check TES energy balance
print("\n=== TES Energy Balance Diagnostics ===")
print(f"TES SOC range: {tes_soc.min():.3f} to {tes_soc.max():.3f}")
print(f"\nHeat Pump Distribution:")
print(f"  HP Total Heat - sum: {hp_heat_total.sum():.2f} kW, max: {hp_heat_total.max():.2f} kW")
print(f"  HP Heat to TES - sum: {tes_input_power_hp.sum():.2f} kW, max: {tes_input_power_hp.max():.2f} kW")
print(f"  HP Heat to Demand (direct) - sum: {hp_heat_to_demand.sum():.2f} kW, max: {hp_heat_to_demand.max():.2f} kW")
print(f"\nThermal Storage Flows:")
print(f"  TES Input from HP - sum: {tes_input_power_hp.sum():.2f} kW, max: {tes_input_power_hp.max():.2f} kW")
print(f"  TES Input from HR - sum: {tes_input_power_hr.sum():.2f} kW, max: {tes_input_power_hr.max():.2f} kW")
print(f"  TES Output to Demand - sum: {tes_output_power.sum():.2f} kW, max: {tes_output_power.max():.2f} kW")
print(f"  TES Losses - sum: {tes_losses.sum():.2f} kW, max: {tes_losses.max():.2f} kW")
print(f"  Net TES power (HP+HR-Output-Loss): {(tes_input_power_hp + tes_input_power_hr - tes_output_power - tes_losses).sum():.2f} kW")
print("======================================\n")

# Create mask for first two days
mask_2d = df.index[:500]

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

# PLOT 4: Power Components Overview (first two days)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(mask_2d, battery_power.loc[mask_2d, 'Household battery'].values, label='battery_power', linestyle='-',
         linewidth=2, color='purple')
ax4.plot(mask_2d, buy_power.loc[mask_2d].sum(axis=1).values, label='buy_power', marker='x',
         linestyle='', markersize=6, color='red')
ax4.plot(mask_2d, heat_pump_power.loc[mask_2d, heat_pump.name].values, label='heat_pump_power', linestyle='-',
         linewidth=2, color='teal')
ax4.plot(mask_2d, fixed_consumption.loc[mask_2d].sum(axis=1).values, label='fixed_consumption', linestyle='-',
         linewidth=2, color='blue')
# Calculate total demand (consumption + heat pump power)
total_demand = fixed_consumption.loc[mask_2d].sum(axis=1).values + heat_pump_power.loc[mask_2d, heat_pump.name].values
ax4.plot(mask_2d, total_demand, label='total_demand (fixed_consumption + heat_pump_power)', linestyle='--',
         linewidth=2, color='darkviolet')
ax4.set_xlabel('Time')
ax4.set_ylabel('Power [W]')
ax4.set_title('Power Components Overview (First 2 Days)')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 5: Outdoor Temperature and COP (first two days)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
# Convert temperature from Kelvin to Celsius for better readability
outdoor_temp_celsius = df.loc[mask_2d, "source_temperature"].values - 273.15
ax5.plot(mask_2d, outdoor_temp_celsius, label='Outdoor Temperature', linestyle='-',
         linewidth=2, color='steelblue')
ax5.set_xlabel('Time')
ax5.set_ylabel('Temperature [°C]', color='steelblue')
ax5.tick_params(axis='y', labelcolor='steelblue')
ax5_twin = ax5.twinx()
ax5_twin.plot(mask_2d, [cop_values.loc[idx] for idx in mask_2d], label='COP', linestyle='--', 
              linewidth=2, color='crimson')
ax5_twin.set_ylabel('COP', color='crimson')
ax5_twin.tick_params(axis='y', labelcolor='crimson')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax5.set_title('Outdoor Temperature and COP (First 2 Days)')
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 6: Thermal Storage Power Flows and Losses (first two days)
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.plot(mask_2d, [tes_input_power_hp.loc[idx] for idx in mask_2d], label='Input from Heat Pump', 
         linestyle='-', linewidth=2, color='forestgreen')
ax6.plot(mask_2d, [tes_input_power_hr.loc[idx] for idx in mask_2d], label='Input from Heating Rod', 
         linestyle='-', linewidth=2, color='gold')
ax6.plot(mask_2d, [tes_output_power.loc[idx] for idx in mask_2d], label='Output to Demand', 
         linestyle='-', linewidth=2, color='darkorange')
ax6.plot(mask_2d, [tes_losses.loc[idx] for idx in mask_2d], label='TES Losses', 
         linestyle='--', linewidth=2, color='red')
ax6.set_xlabel('Time')
ax6.set_ylabel('Power [kW]')
ax6.set_title('Thermal Storage Power Flows and Losses (First 2 Days)')
ax6.legend()
ax6.grid(True, alpha=0.3)
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# PLOT 7: Heat Pump Heat Distribution (first two days)
fig7 = plt.figure()
ax7 = fig7.add_subplot(111)
ax7.plot(mask_2d, [hp_heat_total.loc[idx] for idx in mask_2d], label='HP Total Heat Output', 
         linestyle='-', linewidth=2.5, color='purple')
ax7.plot(mask_2d, [hp_heat_to_demand.loc[idx] for idx in mask_2d], label='HP Heat to Demand (direct)', 
         linestyle='-', linewidth=2, color='blue')
ax7.plot(mask_2d, [tes_input_power_hp.loc[idx] for idx in mask_2d], label='HP Heat to TES', 
         linestyle='-', linewidth=2, color='forestgreen')
ax7.plot(mask_2d, df.loc[mask_2d, "heat_demand"].values, label='Heat Demand', 
         linestyle='--', linewidth=2, color='red', alpha=0.7)
ax7.set_xlabel('Time')
ax7.set_ylabel('Power [kW]')
ax7.set_title('Heat Pump Heat Distribution (First 2 Days)')
ax7.legend()
ax7.grid(True, alpha=0.3)
plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# Show plots
plt.show()
print("\nPlots displayed")
print("Check TES SOC and power flows to understand the optimization behavior")