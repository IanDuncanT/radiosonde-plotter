# {
#   'alt': 12551.81606,
#   'batt': 2.8,
#   'burst_timer': 65535,
#   'datetime': '2026-05-30T11:39:23.001000Z',
#   'frame': 4810,
#   'frequency': 404.799775,
#   'heading': 87.63714,
#   'humidity': 19.3,
#   'lat': 32.97627,
#   'launch_site': '72249',
#   'launch_site_range_estimate': 73.89688762052161,
#   'lon': -96.94974,
#   'manufacturer': 'Vaisala',
#   'position': '32.97627,-96.94974',
#   'ref_datetime': 'GPS',
#   'ref_position': 'GPS',
#   'rs41_mainboard': 'RSM415',
#   'rs41_mainboard_fw': '20701',
#   'sats': 12,
#   'serial': 'X5110802',
#   'snr': 17.2,
#   'software_name': 'radiosonde_auto_rx',
#   'software_version': '1.8.1',
#   'subtype': 'RS41-NG',
#   'temp': -57.2,
#   'time_received': '2026-05-30T11:39:07.181604Z',
#   'tx_frequency': 404.8,
#   'type': 'RS41',
#   'uploader_alt': 191.0,
#   'uploader_antenna': '1/4 wave monopole',
#   'uploader_callsign': 'K5MGY',
#   'uploader_position': '32.638008,-97.109726',
#   'user-agent': 'Amazon CloudFront',
#   'vel_h': 25.26184,
#   'vel_v': 4.8619
# }

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import scipy.stats as stats
import datetime
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d
import sys
import traceback

import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily not installed. Map tiles will be skipped.")

sonde_id = 'X4643492'
sonde_office = 'FWD'
sonde_time = 'June 19th 2026 0Z'


# =============================================================================
# Helper Functions
# =============================================================================

def isaBarometricFormula(alt_m):
    """ISA standard barometric formula. Returns pressure in Pascals."""
    P0, T0, L = 101325.0, 288.15, -0.0065
    g, M, R = 9.80665, 0.0289644, 8.31446
    h_trop = 11000.0

    T11 = T0 + L * h_trop
    P11 = P0 * (T0 / T11) ** (g * M / (R * L))

    if alt_m <= h_trop:
        return P0 * (T0 / (T0 + L * alt_m)) ** (g * M / (R * L))
    else:
        return P11 * np.exp(-g * M * (alt_m - h_trop) / (R * T11))


def magnusFormula(temp_c, rh_pct):
    """Magnus formula. Returns dew point in degrees C."""
    a, b = 17.625, 243.04
    alpha = np.log(rh_pct / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


# =============================================================================
# Load Raw Data
# =============================================================================

with open(f'{sonde_id}.json') as f:
    raw_data = json.load(f)

print(f"Raw packets loaded: {len(raw_data)}")

# =============================================================================
# Step 1: Parse into list of lists
# Column layout:
#  0  datetime
#  1  lat
#  2  lon
#  3  alt (m)
#  4  pressure (hPa)
#  5  temp (°C)
#  6  rh (%)
#  7  dewpoint (°C)
#  8  vel_v (m/s)
#  9  vel_h (m/s)
# 10  heading (°)
# 11  u (m/s)   — meteorological convention (negated)
# 12  v (m/s)   — meteorological convention (negated)
# 13  frame
# 14  sats
# 15  snr
# =============================================================================

filtered_data = [
    [
        datetime.datetime.strptime(i['datetime'], "%Y-%m-%dT%H:%M:%S.%fZ"),
        float(i['lat'])       if 'lat'      in i else None,
        float(i['lon'])       if 'lon'      in i else None,
        float(i['alt'])       if 'alt'      in i else None,
        float(isaBarometricFormula(float(i['alt'])) / 100.0) if 'alt' in i else None,
        float(i['temp'])      if 'temp'     in i else None,
        float(i['humidity'])  if 'humidity' in i else None,
        float(magnusFormula(float(i['temp']), float(i['humidity']))) if 'temp' in i and 'humidity' in i else None,
        float(i['vel_v'])     if 'vel_v'    in i else None,
        float(i['vel_h'])     if 'vel_h'    in i else None,
        float(i['heading'])   if 'heading'  in i else None,
        float(1 * i['vel_h']) * np.sin(np.radians(float(i['heading']))) if 'vel_h' in i and 'heading' in i else None,
        float(1 * i['vel_h']) * np.cos(np.radians(float(i['heading']))) if 'vel_h' in i and 'heading' in i else None,
        int(i['frame'])       if 'frame'    in i else None,
        int(i['sats'])        if 'sats'     in i else None,
        float(i['snr'])       if 'snr'      in i else None,
    ]
    for i in raw_data
]

filtered_data = np.array(filtered_data, dtype=object)

# =============================================================================
# Step 2: Sort by datetime
# =============================================================================

filtered_data = filtered_data[np.argsort(filtered_data[:, 0])]
print(f"After sort: {len(filtered_data)}")

# =============================================================================
# Step 3: Filter out rows where frame is None, then deduplicate by frame
#         keeping the row with the highest SNR per frame.
#         Cast frame to int before comparison to avoid object/string issues.
# =============================================================================

frame_mask = np.array([x is not None for x in filtered_data[:, 13]])
filtered_data = filtered_data[frame_mask]
print(f"After removing None frames: {len(filtered_data)}")

frames = filtered_data[:, 13].astype(int)   # cast to int for reliable comparison
unique_frames = np.unique(frames)

deduped_rows = []
for frame in unique_frames:
    mask = frames == frame
    frame_rows = filtered_data[mask]
    snr_vals = np.array([float(x) if x is not None else -np.inf for x in frame_rows[:, 15]])
    best_idx = np.argmax(snr_vals)
    deduped_rows.append(frame_rows[best_idx])

filtered_data = np.array(deduped_rows, dtype=object)
print(f"After deduplication by frame: {len(filtered_data)}")

# =============================================================================
# Step 4: Filter low GPS satellite count (sats < 4)
# =============================================================================

sat_mask = np.array([x is not None and int(x) >= 4 for x in filtered_data[:, 14]])
filtered_data = filtered_data[sat_mask]
print(f"After GPS sat filter (>=4): {len(filtered_data)}")

# =============================================================================
# Step 5: Filter rows missing temp / humidity / dewpoint
# =============================================================================

thermo_mask = np.array([
    filtered_data[:, 5][i] is not None and
    filtered_data[:, 6][i] is not None and
    filtered_data[:, 7][i] is not None
    for i in range(len(filtered_data))
])
filtered_data = filtered_data[thermo_mask]
print(f"After thermo filter: {len(filtered_data)}")

# =============================================================================
# Step 6: Split ascent using burst altitude (max altitude index)
#         More robust than vel_v > 0 — works even when vel_v is missing.
# =============================================================================

alt_col = filtered_data[:, 3].astype(float)
burst_idx = np.argmax(alt_col)
ascent_data = filtered_data[:burst_idx + 1]
print(f"After ascent split at burst (idx={burst_idx}, alt={alt_col[burst_idx]:.1f} m): {len(ascent_data)}")

# =============================================================================
# Step 7: Extract sounding arrays, apply median filter, enforce monotonic pressure
# =============================================================================

altitude_raw    = ascent_data[:, 3].astype(float)
pressure_raw    = ascent_data[:, 4].astype(float)
temperature_raw = ascent_data[:, 5].astype(float)
dewpoint_raw    = ascent_data[:, 7].astype(float)
u_raw           = ascent_data[:, 11].astype(float)
v_raw           = ascent_data[:, 12].astype(float)

# After median filter, interpolate onto a clean pressure grid
pressure_filt = medfilt(pressure_raw, kernel_size=5)

# Build a clean grid from surface pressure down to minimum, with fixed spacing
p_surface = pressure_filt[0]
p_top     = pressure_filt[-1]
clean_grid = np.arange(p_surface, p_top, -0.5)  # 0.5 hPa steps, strictly decreasing

# Interpolate all variables onto the clean grid
# np.interp requires increasing x, so flip, interpolate, done
altitude    = np.interp(clean_grid, pressure_filt[::-1], altitude_raw[::-1])
temperature = np.interp(clean_grid, pressure_filt[::-1], temperature_raw[::-1])
dewpoint    = np.interp(clean_grid, pressure_filt[::-1], dewpoint_raw[::-1])
u_wind      = np.interp(clean_grid, pressure_filt[::-1], u_raw[::-1])
v_wind      = np.interp(clean_grid, pressure_filt[::-1], v_raw[::-1])
pressure    = clean_grid  # already perfectly monotonic

# Smooth u/v before coarsening for hodograph — window of ~20 points on 0.5 hPa grid = 10 hPa smoothing
u_wind_smooth = uniform_filter1d(u_wind, size=20)
v_wind_smooth = uniform_filter1d(v_wind, size=20)


u_wind_smooth_coarse = u_wind_smooth[::10]
v_wind_smooth_coarse = v_wind_smooth[::10]
altitude_coarse = altitude[::10]


# Attach units
altitude    = altitude    * units.meter
pressure    = pressure    * units.mbar
temperature = temperature * units.degC
dewpoint    = dewpoint    * units.degC
u_wind      = u_wind      * units('m/s')
v_wind      = v_wind      * units('m/s')

u_wind_smooth_coarse = u_wind_smooth_coarse * units('m/s')
v_wind_smooth_coarse = v_wind_smooth_coarse * units('m/s')
altitude_coarse = altitude_coarse * units.meter

print(f"Final sounding points: {len(pressure)}")

# =============================================================================
# Plotting
# =============================================================================

# Create a figure and skew-t plot object — wider for map panel on far right
fig = plt.figure(figsize=(26, 12))
skew = SkewT(fig, rotation=45, rect=(0.03, 0.05, 0.34, 0.90))

# Set the limits of the pressure and temperature axis. The temperature goes lower, but since its skewed, it looks cleaner to limit it further
skew.ax.set_adjustable('datalim')
skew.ax.set_ylim(1000, 50)
skew.ax.set_xlim(-20, 30)

# Set some better labels than the default to increase readability
skew.ax.set_xlabel(f'Temperature ({temperature.units:~P})', weight='bold')
skew.ax.set_ylabel(f'Pressure ({pressure.units:~P})', weight='bold')

# Set the facecolor of the skew-t object and the figure to white
fig.set_facecolor('#ffffff')
skew.ax.set_facecolor('#ffffff')

# Create an isotherm pattern on the graph
x1 = np.linspace(-100, 40, 8)
x2 = np.linspace(-90, 50, 8)
y = [1100, 50]
for i in range(8):
    skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='gray', alpha=0.02, zorder=1)

# Plot the temperature and dew point on the skew-t
skew.plot(pressure, temperature, 'r', lw=3, label='Temperature')
skew.plot(pressure, dewpoint, 'g', lw=3, label='Dew Point')

# Use some pthon math to 'resample' the wind barbs for a cleaner output with increased readability.
interval = np.logspace(1.7, 3, 40) * units.hPa
idx = mpcalc.resample_nn_1d(pressure, interval)
skew.plot_barbs(pressure=pressure[idx], u=u_wind[idx], v=v_wind[idx])

# Add the special lines to the Skew-T Log-P diagram
skew.ax.axvline(0 * units.degC, linestyle='--', color='blue', alpha=0.3)
skew.plot_dry_adiabats(lw=1, alpha=0.3)
skew.plot_moist_adiabats(lw=1, alpha=0.3)
skew.plot_mixing_lines(lw=1, alpha=0.3)

# Calculate LCL height and plot as a black dot
lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
skew.plot(pressure, prof, 'k', linewidth=2, label='SB Parcel Path')

# Shade areas of CAPE and CIN
skew.shade_cin(pressure, temperature, prof, dewpoint, alpha=0.2, label='SBCIN')
skew.shade_cape(pressure, temperature, prof, alpha=0.2, label='SBCAPE')

# Create axis and the hodograph plot object — middle column, top half
hodo_ax = plt.axes((0.37, 0.45, 0.28, 0.50))
h = Hodograph(hodo_ax, component_range=40.)

# Add two grid increments to the hodograph
h.add_grid(increment=20, ls='-', lw=1.5, alpha=0.5)
h.add_grid(increment=10, ls='--', lw=1, alpha=0.2)

# Remove several elements to increase readability
h.ax.set_box_aspect(1)
h.ax.set_yticklabels([])
h.ax.set_xticklabels([])
h.ax.set_xticks([])
h.ax.set_yticks([])
h.ax.set_xlabel(' ')
h.ax.set_ylabel(' ')

# Add custom ticks to the hodograph
plt.xticks(np.arange(0, 0, 1))
plt.yticks(np.arange(0, 0, 1))
for i in range(10, 120, 10):
    h.ax.annotate(str(i), (i, 0), xytext=(0, 2), textcoords='offset pixels',
                  clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)
for i in range(10, 120, 10):
    h.ax.annotate(str(i), (0, i), xytext=(0, 2), textcoords='offset pixels',
                  clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)

# Plot hodograph with continuous altitude color scale
h.plot_colormapped(u_wind_smooth_coarse, v_wind_smooth_coarse, c=altitude_coarse, lw=3,
                   cmap='viridis', label='Wind')

# Add a colorbar for the altitude scale
sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(
        vmin=altitude_coarse.magnitude.min(),
        vmax=altitude_coarse.magnitude.max()
    )
)
sm.set_array([])

# Color bar exists for the location plot, so it is redundant here.
# cbar = plt.colorbar(sm, ax=hodo_ax, orientation='vertical',
#                     pad=0.01, shrink=0.8, aspect=30)
# cbar.set_label('Altitude (m)', fontsize=9)
# cbar.ax.tick_params(labelsize=8)

# Bunkers storm motion markers
RM, LM, MW = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, altitude)
for vec, label in [(RM, 'RM'), (LM, 'LM'), (MW, 'MW')]:
    h.ax.plot(vec[0].m, vec[1].m, 'k+', markersize=10, markeredgewidth=2)
    h.ax.text(vec[0].m + 1, vec[1].m + 1, label, weight='bold',
              fontsize=11, alpha=0.7)

h.ax.arrow(0, 0, RM[0].m - 0.3, RM[1].m - 0.3, linewidth=2, color='black',
           alpha=0.2, label='Bunkers RM Vector',
           length_includes_head=True, head_width=2)

hodoleg = h.ax.legend(loc='upper left')

# Create a rectangle for listing parameters — middle column, bottom half
fig.patches.extend([plt.Rectangle((0.375, 0.05), 0.225, 0.37,
                                  edgecolor='black', facecolor='white',
                                  linewidth=1, alpha=1, transform=fig.transFigure,
                                  figure=fig)])

# Severe weather indices
kindex = mpcalc.k_index(pressure, temperature, dewpoint)
total_totals = mpcalc.total_totals_index(pressure, temperature, dewpoint)

# Mixed layer parcel properties
ml_t, ml_td = mpcalc.mixed_layer(pressure, temperature, dewpoint, depth=50 * units.hPa)
ml_p, _, _ = mpcalc.mixed_parcel(pressure, temperature, dewpoint, depth=50 * units.hPa)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure, temperature, dewpoint, depth=50 * units.hPa)

# Most unstable parcel properties
mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(pressure, temperature, dewpoint, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(pressure, temperature, dewpoint, depth=50 * units.hPa)

# Estimate height of LCL in meters from hydrostatic thickness (for sig_tor)
new_p = np.append(pressure[pressure > lcl_pressure], lcl_pressure)
new_t = np.append(temperature[pressure > lcl_pressure], lcl_temperature)
lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

# Compute Surface-based CAPE
sbcape, sbcin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)

# Compute SRH
(u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, altitude)
*_, total_helicity1 = mpcalc.storm_relative_helicity(altitude, u_wind, v_wind, depth=1 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)
*_, total_helicity3 = mpcalc.storm_relative_helicity(altitude, u_wind, v_wind, depth=3 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)
*_, total_helicity6 = mpcalc.storm_relative_helicity(altitude, u_wind, v_wind, depth=6 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)

# Copmute Bulk Shear components and then magnitude
ubshr1, vbshr1 = mpcalc.bulk_shear(pressure, u_wind, v_wind, height=altitude, depth=1 * units.km)
bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
ubshr3, vbshr3 = mpcalc.bulk_shear(pressure, u_wind, v_wind, height=altitude, depth=3 * units.km)
bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
ubshr6, vbshr6 = mpcalc.bulk_shear(pressure, u_wind, v_wind, height=altitude, depth=6 * units.km)
bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

# Use all computed pieces to calculate the Significant Tornado parameter
sig_tor = mpcalc.significant_tornado(sbcape, lcl_height,
                                     total_helicity3, bshear3).to_base_units()

# Perform the calculation of supercell composite if an effective layer exists
super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

# Plot the parameters in the box
plt.figtext(0.385, 0.37, 'SBCAPE: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.37, f'{sbcape:.0f~P}', weight='bold',
            fontsize=13, color='orangered', ha='right')
plt.figtext(0.385, 0.34, 'SBCIN: ', weight='bold',
            fontsize=13, color='black', ha='left')
plt.figtext(0.495, 0.34, f'{sbcin:.0f~P}', weight='bold',
            fontsize=13, color='lightblue', ha='right')
plt.figtext(0.385, 0.29, 'MLCAPE: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.29, f'{mlcape:.0f~P}', weight='bold',
            fontsize=13, color='orangered', ha='right')
plt.figtext(0.385, 0.26, 'MLCIN: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.26, f'{mlcin:.0f~P}', weight='bold',
            fontsize=13, color='lightblue', ha='right')
plt.figtext(0.385, 0.21, 'MUCAPE: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.21, f'{mucape:.0f~P}', weight='bold',
            fontsize=13, color='orangered', ha='right')
plt.figtext(0.385, 0.18, 'MUCIN: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.18, f'{mucin:.0f~P}', weight='bold',
            fontsize=13, color='lightblue', ha='right')
plt.figtext(0.385, 0.13, 'TT-INDEX: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.13, f'{total_totals:.0f~P}', weight='bold',
            fontsize=13, color='orangered', ha='right')
plt.figtext(0.385, 0.10, 'K-INDEX: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.495, 0.10, f'{kindex:.0f~P}', weight='bold',
            fontsize=13, color='orangered', ha='right')

plt.figtext(0.505, 0.37, '0-1km SRH: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.37, f'{total_helicity1:.0f~P}',
            weight='bold', fontsize=13, color='navy', ha='right')
plt.figtext(0.505, 0.34, '0-1km SHEAR: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.34, f'{bshear1:.0f~P}', weight='bold',
            fontsize=13, color='blue', ha='right')
plt.figtext(0.505, 0.29, '0-3km SRH: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.29, f'{total_helicity3:.0f~P}',
            weight='bold', fontsize=13, color='navy', ha='right')
plt.figtext(0.505, 0.26, '0-3km SHEAR: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.26, f'{bshear3:.0f~P}', weight='bold',
            fontsize=13, color='blue', ha='right')
plt.figtext(0.505, 0.21, '0-6km SRH: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.21, f'{total_helicity6:.0f~P}',
            weight='bold', fontsize=13, color='navy', ha='right')
plt.figtext(0.505, 0.18, '0-6km SHEAR: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.18, f'{bshear6:.0f~P}', weight='bold',
            fontsize=13, color='blue', ha='right')
plt.figtext(0.505, 0.13, 'SIG TORNADO: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.13, f'{sig_tor[0]:.0f~P}', weight='bold', fontsize=13,
            color='orangered', ha='right')
plt.figtext(0.505, 0.10, 'SUPERCELL COMP: ', weight='bold', fontsize=13,
            color='black', ha='left')
plt.figtext(0.595, 0.10, f'{super_comp[0]:.0f~P}', weight='bold', fontsize=13,
            color='orangered', ha='right')

# =============================================================================
# Trajectory Map — far right panel
# =============================================================================

# Pull lat/lon from the full filtered_data (includes descent for full trajectory)
# Use ascent_data for ascent path, and full filtered_data for descent
traj_lat = filtered_data[:, 1].astype(float)
traj_lon = filtered_data[:, 2].astype(float)
traj_alt = filtered_data[:, 3].astype(float)

# Remove any rows where lat/lon is NaN
traj_valid = np.isfinite(traj_lat) & np.isfinite(traj_lon) & np.isfinite(traj_alt)
traj_lat = traj_lat[traj_valid]
traj_lon = traj_lon[traj_valid]
traj_alt = traj_alt[traj_valid]

# Key indices
launch_idx = 0
burst_map_idx = np.argmax(traj_alt)
landing_idx = len(traj_alt) - 1

map_ax = fig.add_axes((0.635, 0.05, 0.345, 0.90))
map_ax.set_aspect('equal', adjustable='datalim')

# Plot trajectory colored by altitude using same viridis colormap
alt_norm = plt.Normalize(vmin=traj_alt.min(), vmax=traj_alt.max())
cmap = plt.cm.viridis

# Draw trajectory as line segments colored by altitude
from matplotlib.collections import LineCollection
points = np.array([traj_lon, traj_lat]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=alt_norm, linewidth=3, zorder=3)
lc.set_array(traj_alt[:-1])
map_ax.add_collection(lc)

# Set map extent with padding
pad = 0.15
map_ax.set_xlim(traj_lon.min() - pad, traj_lon.max() + pad)
map_ax.set_ylim(traj_lat.min() - pad, traj_lat.max() + pad)

# Add contextily map tiles (CartoDB Positron — clean, light background)
if HAS_CONTEXTILY:
    try:
        ctx.add_basemap(
            map_ax,
            crs='EPSG:4326',
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom='auto',
            zorder=1
        )
    except Exception as e:
        print(f"Warning: Could not load map tiles: {e}")
        map_ax.set_facecolor('#e8f4f8')

# Launch marker — green upward triangle
map_ax.plot(traj_lon[launch_idx], traj_lat[launch_idx],
            marker='^', color='limegreen', markersize=14,
            markeredgecolor='black', markeredgewidth=1.5,
            zorder=5, label='Launch')
map_ax.annotate('Launch', (traj_lon[launch_idx], traj_lat[launch_idx]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=9, weight='bold', color='limegreen',
                path_effects=[
                    __import__('matplotlib.patheffects', fromlist=['withStroke'])
                    .withStroke(linewidth=2, foreground='black')
                ], zorder=6)

# Burst marker — red star
map_ax.plot(traj_lon[burst_map_idx], traj_lat[burst_map_idx],
            marker='*', color='red', markersize=16,
            markeredgecolor='black', markeredgewidth=1.5,
            zorder=5, label=f'Burst ({traj_alt[burst_map_idx]:.0f} m)')
map_ax.annotate(f'Burst\n{traj_alt[burst_map_idx]:.0f} m',
                (traj_lon[burst_map_idx], traj_lat[burst_map_idx]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=9, weight='bold', color='red',
                path_effects=[
                    __import__('matplotlib.patheffects', fromlist=['withStroke'])
                    .withStroke(linewidth=2, foreground='black')
                ], zorder=6)

# Landing marker — orange downward triangle
map_ax.plot(traj_lon[landing_idx], traj_lat[landing_idx],
            marker='v', color='orange', markersize=14,
            markeredgecolor='black', markeredgewidth=1.5,
            zorder=5, label='Landing')
map_ax.annotate('Landing', (traj_lon[landing_idx], traj_lat[landing_idx]),
                xytext=(6, -14), textcoords='offset points',
                fontsize=9, weight='bold', color='orange',
                path_effects=[
                    __import__('matplotlib.patheffects', fromlist=['withStroke'])
                    .withStroke(linewidth=2, foreground='black')
                ], zorder=6)

# Colorbar for altitude on the map
sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=alt_norm)
sm_map.set_array([])
cbar_map = plt.colorbar(sm_map, ax=map_ax, orientation='vertical',
                        pad=0.01, shrink=0.8, aspect=30)
cbar_map.set_label('Altitude (m)', fontsize=10, weight='bold')
cbar_map.ax.tick_params(labelsize=9)

map_ax.set_xlabel('Longitude', weight='bold', fontsize=10)
map_ax.set_ylabel('Latitude', weight='bold', fontsize=10)
map_ax.set_title('Balloon Trajectory', weight='bold', fontsize=12)
map_ax.legend(loc='lower left', fontsize=9, framealpha=0.8)
map_ax.tick_params(labelsize=8)
map_ax.axes.get_xaxis().set_visible(False)
map_ax.axes.get_yaxis().set_visible(False)

# Add legends to the skew and hodograph
skewleg = skew.ax.legend(loc='upper left')
hodoleg = h.ax.legend(loc='upper left')

# Add plot title — centered across full figure
plt.figtext(0.5, 0.97, f'Radiosonde {sonde_id} ({sonde_office}) | {sonde_time}',
            weight='bold', fontsize=20, ha='center')

# Show the plot
plt.savefig(f'sonde_graph_{sonde_id}.png', dpi=300, bbox_inches='tight')
# plt.show()