# Using Manufacturer COP Data with hplib

## Overview

The heat pump optimizer now supports using manufacturer COP data to create more accurate heat pump models. When you provide manufacturer COP data, the system automatically fits custom hplib parameters (p1-p4) using least-squares regression, creating a model tailored to your specific heat pump.

## How It Works

### hplib COP Model

hplib uses a linear regression model to calculate COP:

```
COP = p1 * T_source + p2 * T_sink + p3 + p4 * T_amb
```

Where:
- `T_source`: Source temperature (e.g., outdoor air) in °C
- `T_sink`: Sink temperature (heat pump outlet) in °C  
- `T_amb`: Ambient temperature in °C (for Air/Water heat pumps, T_amb = T_source)
- `p1, p2, p3, p4`: Regression parameters

### What Gets Modified

When you provide `cop_data`, the optimizer:
1. Fits the 4 COP parameters (`p1_COP`, `p2_COP`, `p3_COP`, `p4_COP`) from your manufacturer data
2. Updates the hplib model with these custom parameters
3. Uses this customized model for all COP calculations during optimization

**Note:** Only COP parameters are fitted. Thermal power (P_th) and electrical power (P_el) parameters remain from the base model unless you also provide power data (not yet implemented).

## Usage Example

### 1. Prepare Your Manufacturer COP Data

```python
cop_data_manufacturer = {
    # Sink temperatures (heat pump outlet) in °C
    'temp_sink': [25, 25, 25, 30, 30, 30, 35, 35, 35, 40, 40, 40, 45, 45, 45, 50, 50, 50, 55, 55, 55],
    
    # Source temperatures (e.g., outdoor air) in °C
    'temp_source': [7, 2, -7, 7, 2, -7, 7, 2, -7, 7, 2, -7, 7, 2, -7, 7, 2, -7, 7, 2, -7],
    
    # Corresponding COP values
    'cop': [5.78, 4.71, 3.16, 5.06, 4.21, 3.06, 4.5, 3.81, 3.02, 4.0, 3.46, 2.89, 3.55, 3.16, 2.77, 3.29, 2.8, 2.51, 3.06, 2.5, 2.29],
}
```

**Requirements:**
- At least 4 data points (more is better for accuracy)
- All three lists must have the same length
- Temperatures in °C (not Kelvin)

### 2. Create Heat Pump with Custom COP Data

```python
from battery_optimizer.profiles.heat_pump import HeatPump

heat_pump = HeatPump(
    name="my_heat_pump",
    type="Generic",  # Use Generic to allow custom parameters
    id=1,  # 1 = Air/Water regulated (base parameter set)
    t_in=280.15,  # Reference source temp: 7°C in Kelvin
    t_out=328.15,  # Reference sink temp: 55°C in Kelvin
    p_th=10000,  # Reference thermal power in W
    flow_temperature=328.15,  # 55°C
    output_temperature=333.15,  # 60°C  
    max_electric_power_hp=10.0,  # kW
    max_electric_power_hr=5.0,  # kW
    tank_volume=500,  # liters
    outdoor_temperature=temp_profile,
    heat_source_temperature=temp_profile,
    heat_demand=demand_profile,
    cop_data=cop_data_manufacturer,  # ← Add your manufacturer data here
)
```

### 3. Run Optimization

The optimizer will automatically:
- Fit COP parameters from your manufacturer data
- Log the fit quality (RMSE and MAPE)
- Use the custom model for all calculations

```python
opt = ProfileStackProblem(
    buy_prices=buy_profile_stack,
    sell_prices=sell_profile_stack,
    heat_pumps=[heat_pump],
    ...
)
opt.set_up()
```

You'll see output like:
```
Fitting COP parameters for heat pump 'my_heat_pump' from manufacturer data...
Heat pump 'my_heat_pump': Custom COP parameters fitted (RMSE=0.2892, MAPE=7.42%)
```

## Fit Quality Metrics

- **RMSE** (Root Mean Square Error): Average deviation of fitted COP from manufacturer COP
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

Typical values:
- MAPE < 5%: Excellent fit
- MAPE 5-10%: Good fit (typical for most heat pumps)
- MAPE > 10%: Consider providing more data points or checking data quality

## Tips for Best Results

1. **Provide diverse data points**: Include various combinations of source and sink temperatures
2. **Cover the operating range**: Include temperatures your heat pump will actually encounter
3. **More data = better fit**: 20-40 data points typically give good results
4. **Check manufacturer specs**: Ensure temperatures and COP values are from the same test conditions

## Comparison: With vs. Without Manufacturer Data

### Without manufacturer data:
```python
heat_pump = HeatPump(
    name="hp",
    type="Bosch Compress 7000 LW 28",  # Uses generic hplib model
    ...
)
```
→ Uses hplib's generic regression parameters for this model family

### With manufacturer data:
```python
heat_pump = HeatPump(
    name="hp",
    type="Generic",
    cop_data=cop_data_manufacturer,  # Uses your specific data
    ...
)
```
→ Creates a custom model fitted to your exact heat pump specifications

## Advanced: Adding Power Data (Future Enhancement)

Currently, only COP parameters are fitted from manufacturer data. Thermal power (P_th) and electrical power (P_el) parameters use the base Generic model.

A future enhancement could allow fitting these parameters as well:

```python
# Potential future feature
cop_and_power_data = {
    'temp_source': [...],
    'temp_sink': [...],
    'cop': [...],
    'p_th': [...],  # Thermal power in W
    'p_el': [...],  # Electrical power in W
}
```

## Troubleshooting

### "cop_data must contain at least 4 data points"
- Provide at least 4 temperature/COP combinations

### "All lists in cop_data must have the same length"
- Ensure `temp_source`, `temp_sink`, and `cop` have the same number of elements

### High MAPE (>15%)
- Check that temperatures are in °C (not Kelvin)
- Verify COP values are reasonable (typically 2-6 for air-source heat pumps)
- Add more diverse data points covering the full operating range

### Unexpected COP values during optimization
- Check that `flow_temperature` and `output_temperature` are set correctly
- Verify that `heat_source_temperature` matches your COP data's source temperature definition

## Reference

See `example/example3.py` for a complete working example.

