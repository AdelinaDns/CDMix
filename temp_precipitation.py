# import ee
# import geemap
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime, timedelta

# # ============================================================
# # Autentificare & Initializare GEE
# # ============================================================

# ee.Authenticate()
# ee.Initialize(project='disertatie-496115')

# # ============================================================
# # 1. Load asset
# # ============================================================

# slovakia = ee.FeatureCollection('projects/disertatie-496115/assets/sections')

# YEARS  = [2020, 2021, 2022, 2023, 2024, 2025]
# COLORS = ['#1E88E5', '#1E88E5', '#1E88E5', '#1E88E5', '#1E88E5', '#AA2F24']

# # Banda de umiditate a solului din SMAP L4:
# #   'sm_surface'  -> strat superficial 0-5 cm   (reactioneaza rapid la seceta/caldura)
# #   'sm_rootzone' -> zona radacinilor 0-100 cm  (raspuns mai lent, integrat)
# SM_BAND = 'sm_surface'

# # ============================================================
# # FLAG: seteaza True pentru a afisa media anilor cu aceeasi culoare
# #       vs. 2025, sau False pentru a afisa toti anii individual
# # ============================================================

# SHOW_MEAN_VS_2025 = True  

# # Determina grupurile de culori automat din COLORS + YEARS
# def get_color_groups(years, colors):
#     """Grupeaza anii dupa culoare."""
#     groups = {}
#     for year, color in zip(years, colors):
#         groups.setdefault(color, []).append(year)
#     return groups  # {culoare: [lista_ani]}

# # ============================================================
# # 2. Genereaza ferestre bisaptamanale (14 zile) pentru un an
# # ============================================================

# def get_biweekly_windows(year: int) -> list:
#     windows = []
#     start = datetime(year, 1, 1)
#     end_of_year = datetime(year, 12, 31)
#     while start <= end_of_year:
#         end = min(start + timedelta(days=13), end_of_year)
#         windows.append((start, end))
#         start = end + timedelta(days=1)
#     return windows


# def datetime_to_ee(dt: datetime) -> ee.Date:
#     return ee.Date(dt.strftime('%Y-%m-%d'))


# # ============================================================
# # 3. Helper functions
# # ============================================================

# def get_biweekly_lst(year: int, band: str) -> list:
#     results = []
#     windows = get_biweekly_windows(year)
#     for start_dt, end_dt in windows:
#         col = (
#             ee.ImageCollection('MODIS/061/MOD11A2')
#             .filterDate(datetime_to_ee(start_dt),
#                         datetime_to_ee(end_dt).advance(1, 'day'))
#             .select(band)
#         )
#         if col.size().getInfo() == 0:
#             continue
#         val = (
#             col.mean()
#             .multiply(0.02)
#             .subtract(273.15)
#             .reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=slovakia.geometry(),
#                 scale=1000,
#                 maxPixels=1e9
#             )
#             .get(band)
#             .getInfo()
#         )
#         if val is not None:
#             results.append({'date': start_dt, 'value': val})
#     print(f"  LST {band} {year}: {len(results)} ferestre cu date")
#     return results


# def get_biweekly_precip(year: int) -> list:
#     results = []
#     windows = get_biweekly_windows(year)
#     for start_dt, end_dt in windows:
#         col = (
#             ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
#             .filterDate(datetime_to_ee(start_dt),
#                         datetime_to_ee(end_dt).advance(1, 'day'))
#         )
#         if col.size().getInfo() == 0:
#             continue
#         val = (
#             col.sum()
#             .reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=slovakia.geometry(),
#                 scale=5566,
#                 maxPixels=1e9
#             )
#             .values()
#             .get(0)
#             .getInfo()
#         )
#         if val is not None:
#             results.append({'date': start_dt, 'value': val})
#     print(f"  Precip {year}: {len(results)} ferestre cu date")
#     return results


# def get_biweekly_soil_moisture(year: int, band: str = SM_BAND) -> list:
#     """
#     Umiditate volumetrica a solului din SMAP L4 (NASA/SMAP/SPL4SMGP/007).
#     Date la 3 ore -> mediem peste fereastra de 14 zile (col.mean()), apoi
#     media spatiala peste geometria Slovaciei. Valoarea e in m3/m3 (fara
#     factor de scalare).
#     """
#     results = []
#     windows = get_biweekly_windows(year)
#     for start_dt, end_dt in windows:
#         col = (
#             ee.ImageCollection('NASA/SMAP/SPL4SMGP/008')
#             .filterDate(datetime_to_ee(start_dt),
#                         datetime_to_ee(end_dt).advance(1, 'day'))
#             .select(band)
#         )
#         if col.size().getInfo() == 0:
#             continue
#         val = (
#             col.mean()
#             .reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=slovakia.geometry(),
#                 scale=10000,        # grila SMAP ~9 km
#                 maxPixels=1e9
#             )
#             .get(band)
#             .getInfo()
#         )
#         if val is not None:
#             results.append({'date': start_dt, 'value': val})
#     print(f"  Soil Moisture {band} {year}: {len(results)} ferestre cu date")
#     return results


# def compute_group_mean(years_in_group: list, data_dict: dict) -> list:
#     """
#     Calculeaza media valorilor pentru un grup de ani,
#     normalizate la aceeasi axa temporala (year=2000).
#     Returneaza lista {date, value}.
#     """
#     # Colecteaza toate valorile per fereastra normalizata
#     window_map = {}  # date_norm -> [valori]
#     for year in years_in_group:
#         for entry in data_dict.get(year, []):
#             key = entry['date'].replace(year=2000)
#             window_map.setdefault(key, []).append(entry['value'])

#     # Calculeaza media per fereastra
#     result = []
#     for date_norm in sorted(window_map.keys()):
#         vals = window_map[date_norm]
#         result.append({'date': date_norm, 'value': np.mean(vals)})
#     return result


# # ============================================================
# # 4. Extract data pentru toti anii
# # ============================================================

# data_lst_day   = {}
# data_lst_night = {}
# data_precip    = {}
# data_sm        = {}

# for year in YEARS:
#     print(f"\nExtragere date {year}...")
#     data_lst_day[year]   = get_biweekly_lst(year,   'LST_Day_1km')
#     data_lst_night[year] = get_biweekly_lst(year,   'LST_Night_1km')
#     data_precip[year]    = get_biweekly_precip(year)
#     data_sm[year]        = get_biweekly_soil_moisture(year)

# print("\nToate datele extrase!")

# # ============================================================
# # 5. Plot
# # ============================================================

# fig, axes = plt.subplots(4, 1, figsize=(16, 20))

# title_suffix = 'medie 2020–2024 vs. 2025' if SHOW_MEAN_VS_2025 else '2020–2025'
# fig.suptitle(
#     f'Slovakia — LST Day, LST Night, Precipitatii & Umiditate sol bisaptamanal ({title_suffix})',
#     fontsize=15, fontweight='bold', y=0.99
# )

# MONTH_ABBR = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun',
#                'Iul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# def month_ticks():
#     return [datetime(2000, m, 1) for m in range(1, 13)]

# sm_label = 'Surface (0-5 cm)' if SM_BAND == 'sm_surface' else 'Root zone (0-100 cm)'

# DATASETS = [
#     (data_lst_day,   'Biweekly Mean LST Day',                       'LST Day (°C)',           'o', True),
#     (data_lst_night, 'Biweekly Mean LST Night',                      'LST Night (°C)',         's', True),
#     (data_precip,    'Biweekly Mean Precipitation',                  'Precipitation (mm)',     '^', False),
#     (data_sm,        f'Biweekly Mean Soil Moisture — {sm_label}',    'Soil Moisture (m³/m³)',  'D', False),
# ]

# color_groups = get_color_groups(YEARS, COLORS)
# # Ex: {'#1E88E5': [2020, 2021, 2022, 2023, 2024], '#AA2F24': [2025]}

# for ax, (data_dict, title, ylabel, marker, show_zero) in zip(axes, DATASETS):

#     if SHOW_MEAN_VS_2025:
#         # --- Mod MEAN vs 2025 ---
#         for color, years_in_group in color_groups.items():
#             mean_data = compute_group_mean(years_in_group, data_dict)
#             if not mean_data:
#                 continue

#             norm_dates = [d['date'] for d in mean_data]
#             values     = [d['value'] for d in mean_data]

#             if len(years_in_group) > 1:
#                 label = f"Medie {min(years_in_group)}–{max(years_in_group)}"
#                 lw, alpha, zorder = 2.5, 1.0, 3
#             else:
#                 label = str(years_in_group[0])
#                 lw, alpha, zorder = 2.0, 0.95, 4

#             ax.plot(norm_dates, values,
#                     color=color, linewidth=lw,
#                     marker=marker, markersize=4,
#                     label=label, alpha=alpha, zorder=zorder)

#     else:
#         # --- Mod INDIVIDUAL (comportamentul original) ---
#         for year, color in zip(YEARS, COLORS):
#             entries = data_dict.get(year, [])
#             if not entries:
#                 continue
#             norm_dates = [d['date'].replace(year=2000) for d in entries]
#             values     = [d['value'] for d in entries]
#             ax.plot(norm_dates, values,
#                     color=color, linewidth=1.8,
#                     marker=marker, markersize=3,
#                     label=str(year))

#     ax.set_title(title, fontsize=12)
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(month_ticks())
#     ax.set_xticklabels(MONTH_ABBR, fontsize=10)
#     ax.legend(title='An', loc='upper left', ncol=5)
#     ax.grid(True, linestyle='--', alpha=0.5)
#     if show_zero:
#         ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

# plt.tight_layout()
# fname = 'slovakia_biweekly_mean_vs_2025.png' if SHOW_MEAN_VS_2025 else 'slovakia_biweekly_2020_2025.png'
# plt.savefig(fname, dpi=150, bbox_inches='tight')
# plt.show()
# print(f"Graficele salvate in {fname}")

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# Autentificare & Initializare GEE
# ============================================================

ee.Authenticate()
ee.Initialize(project='disertatie-496115')

# ============================================================
# 1. Load assets
# ============================================================

pastis_tiles = ee.FeatureCollection('projects/disertatie-496115/assets/PASTIS_tiles')
slovakia     = ee.FeatureCollection('projects/disertatie-496115/assets/sections')

# Banda de umiditate a solului din SMAP L4:
#   'sm_surface'  -> strat superficial 0-5 cm
#   'sm_rootzone' -> zona radacinilor 0-100 cm
SM_BAND = 'sm_surface'

# Definitia regiunilor: fiecare cu anul ei si culoarea ei
REGIONS = {
    'PASTIS Tiles 2017': {'fc': pastis_tiles, 'year': 2017, 'color': '#FF6B35'},
    'Slovakia 2020':     {'fc': slovakia,     'year': 2020, 'color': '#1E88E5'},
}

MONTH_ABBR = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun',
               'Iul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ============================================================
# 2. Genereaza ferestre bisaptamanale (14 zile) pentru un an
# ============================================================

def get_biweekly_windows(year: int) -> list:
    windows = []
    start = datetime(year, 1, 1)
    end_of_year = datetime(year, 12, 31)
    while start <= end_of_year:
        end = min(start + timedelta(days=13), end_of_year)
        windows.append((start, end))
        start = end + timedelta(days=1)
    return windows


def datetime_to_ee(dt: datetime) -> ee.Date:
    return ee.Date(dt.strftime('%Y-%m-%d'))


# ============================================================
# 3. Helper functions (primesc geometria regiunii ca parametru)
# ============================================================

def get_biweekly_lst(year: int, band: str, region) -> list:
    results = []
    for start_dt, end_dt in get_biweekly_windows(year):
        col = (
            ee.ImageCollection('MODIS/061/MOD11A2')
            .filterDate(datetime_to_ee(start_dt),
                        datetime_to_ee(end_dt).advance(1, 'day'))
            .select(band)
        )
        if col.size().getInfo() == 0:
            continue
        val = (
            col.mean()
            .multiply(0.02)
            .subtract(273.15)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=1000,
                maxPixels=1e9
            )
            .get(band)
            .getInfo()
        )
        if val is not None:
            results.append({'date': start_dt, 'value': val})
    return results


def get_biweekly_precip(year: int, region) -> list:
    results = []
    for start_dt, end_dt in get_biweekly_windows(year):
        col = (
            ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
            .filterDate(datetime_to_ee(start_dt),
                        datetime_to_ee(end_dt).advance(1, 'day'))
        )
        if col.size().getInfo() == 0:
            continue
        val = (
            col.sum()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=5566,
                maxPixels=1e9
            )
            .values()
            .get(0)
            .getInfo()
        )
        if val is not None:
            results.append({'date': start_dt, 'value': val})
    return results


def get_biweekly_soil_moisture(year: int, region, band: str = SM_BAND) -> list:
    """SMAP L4 (NASA/SMAP/SPL4SMGP/008), m3/m3, scala nativa 11 km."""
    results = []
    for start_dt, end_dt in get_biweekly_windows(year):
        col = (
            ee.ImageCollection('NASA/SMAP/SPL4SMGP/008')
            .filterDate(datetime_to_ee(start_dt),
                        datetime_to_ee(end_dt).advance(1, 'day'))
            .select(band)
        )
        if col.size().getInfo() == 0:
            continue
        val = (
            col.mean()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=11000,
                maxPixels=1e9
            )
            .get(band)
            .getInfo()
        )
        if val is not None:
            results.append({'date': start_dt, 'value': val})
    return results


# ============================================================
# 4. Extract data pentru fiecare regiune
# ============================================================

data = {name: {} for name in REGIONS}

for name, cfg in REGIONS.items():
    print(f"\nExtragere {name}...")
    region = cfg['fc'].geometry()
    year   = cfg['year']

    data[name]['lst_day']   = get_biweekly_lst(year, 'LST_Day_1km',   region)
    data[name]['lst_night'] = get_biweekly_lst(year, 'LST_Night_1km', region)
    data[name]['precip']    = get_biweekly_precip(year, region)
    data[name]['sm']        = get_biweekly_soil_moisture(year, region)

    print(f"  LST Day: {len(data[name]['lst_day'])} | "
          f"LST Night: {len(data[name]['lst_night'])} | "
          f"Precip: {len(data[name]['precip'])} | "
          f"Soil Moisture: {len(data[name]['sm'])} ferestre")

print("\nToate datele extrase!")

# ============================================================
# 5. Plot
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 20))
fig.suptitle('PASTIS Tiles (2017) vs Slovakia (2020) — Bisaptamanal',
             fontsize=15, fontweight='bold', y=0.99)

def month_ticks():
    return [datetime(2000, m, 1) for m in range(1, 13)]

sm_label = 'Surface (0-5 cm)' if SM_BAND == 'sm_surface' else 'Root zone (0-100 cm)'

DATASETS = [
    ('lst_day',   'Biweekly Mean LST Day',                       'LST Day (°C)',           'o', True),
    ('lst_night', 'Biweekly Mean LST Night',                      'LST Night (°C)',         's', True),
    ('precip',    'Biweekly Mean Precipitation',                  'Precipitation (mm)',     '^', False),
    ('sm',        f'Biweekly Mean Soil Moisture — {sm_label}',    'Soil Moisture (m³/m³)',  'D', False),
]

for ax, (key, title, ylabel, marker, show_zero) in zip(axes, DATASETS):
    for name, cfg in REGIONS.items():
        entries = data[name][key]
        if not entries:
            continue

        norm_dates = [d['date'].replace(year=2000) for d in entries]
        values     = [d['value'] for d in entries]

        ax.plot(norm_dates, values,
                color=cfg['color'], linewidth=1.8,
                marker=marker, markersize=4,
                label=name)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(month_ticks())
    ax.set_xticklabels(MONTH_ABBR, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    if show_zero:
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

plt.tight_layout()
plt.savefig('pastis_vs_slovakia_biweekly.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graficele salvate in pastis_vs_slovakia_biweekly.png")

# ============================================================
# 6. Harta
# ============================================================

m = geemap.Map()
m.centerObject(pastis_tiles, 6)
m.addLayer(
    pastis_tiles.style({'color': 'FF6B35', 'fillColor': 'FF6B3525', 'width': 2}),
    {}, 'PASTIS Tiles (2017)'
)
m.addLayer(
    slovakia.style({'color': '1E88E5', 'fillColor': '1E88E525', 'width': 2}),
    {}, 'Slovakia (2020)'
)
m.save('map_pastis_slovakia.html')
print("Harta salvata in map_pastis_slovakia.html")