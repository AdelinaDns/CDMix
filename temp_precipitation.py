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
# 1. Load asset
# ============================================================

slovakia = ee.FeatureCollection('projects/disertatie-496115/assets/sections')

YEARS  = [2020, 2021, 2022, 2023, 2024]
COLORS = ['#1E88E5', '#43A047', '#E53935', '#FB8C00', '#8E24AA']

# ============================================================
# 2. Genereaza ferestre bisaptamanale (14 zile) pentru un an
# ============================================================

def get_biweekly_windows(year: int) -> list:
    """Returneaza lista de (start_date, end_date) pentru ferestre de 14 zile."""
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
# 3. Helper functions
# ============================================================

def get_biweekly_lst(year: int, band: str) -> list:
    """
    MODIS LST bisaptamanal — medie per fereastra de 14 zile.
    Returneaza lista {date, value}.
    """
    results = []
    windows = get_biweekly_windows(year)

    for start_dt, end_dt in windows:
        col = (
            ee.ImageCollection('MODIS/061/MOD11A2')
            .filterDate(datetime_to_ee(start_dt),
                        datetime_to_ee(end_dt).advance(1, 'day'))
            .select(band)
        )

        count = col.size().getInfo()
        if count == 0:
            continue

        val = (
            col.mean()
            .multiply(0.02)
            .subtract(273.15)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=slovakia.geometry(),
                scale=1000,
                maxPixels=1e9
            )
            .get(band)
            .getInfo()
        )

        if val is not None:
            results.append({'date': start_dt, 'value': val})

    print(f"  LST {band} {year}: {len(results)} ferestre cu date")
    return results


def get_biweekly_precip(year: int) -> list:
    """
    CHIRPS precipitatii bisaptamanal — suma per fereastra de 14 zile.
    Returneaza lista {date, value}.
    """
    results = []
    windows = get_biweekly_windows(year)

    for start_dt, end_dt in windows:
        col = (
            ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
            .filterDate(datetime_to_ee(start_dt),
                        datetime_to_ee(end_dt).advance(1, 'day'))
        )

        count = col.size().getInfo()
        if count == 0:
            continue

        val = (
            col.sum()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=slovakia.geometry(),
                scale=5566,
                maxPixels=1e9
            )
            .values()
            .get(0)
            .getInfo()
        )

        if val is not None:
            results.append({'date': start_dt, 'value': val})

    print(f"  Precip {year}: {len(results)} ferestre cu date")
    return results


def to_plot_arrays(data_list: list):
    """Transforma lista {date, value} in arrays pentru plot."""
    dates  = [d['date'] for d in data_list]
    values = [d['value'] for d in data_list]
    return dates, values


# ============================================================
# 4. Extract data pentru toti anii
# ============================================================

data_lst_day   = {}
data_lst_night = {}
data_precip    = {}

for year in YEARS:
    print(f"\nExtragere date {year}...")
    data_lst_day[year]   = get_biweekly_lst(year,   'LST_Day_1km')
    data_lst_night[year] = get_biweekly_lst(year,   'LST_Night_1km')
    data_precip[year]    = get_biweekly_precip(year)

print("\nToate datele extrase!")
# ============================================================
# 5. Plot
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 15))
fig.suptitle('Slovakia — LST Day, LST Night & Precipitatii bisaptamanal (2020–2024)',
             fontsize=15, fontweight='bold', y=0.99)

MONTH_ABBR = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun',
               'Iul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def month_ticks():
    return [datetime(2000, m, 1) for m in range(1, 13)]

DATASETS = [
    (data_lst_day,   'Biweekly Mean LST Day',      'LST Day (°C)',       'o', True),
    (data_lst_night, 'Biweekly Mean LST Night',     'LST Night (°C)',     's', True),
    (data_precip,    'Biweekly Mean Precipitation', 'Precipitation (mm)', '^', False),
]

for ax, (data_dict, title, ylabel, marker, show_zero) in zip(axes, DATASETS):
    for year, color in zip(YEARS, COLORS):
        entries = data_dict[year]
        if not entries:
            continue

        # Normalizam la anul 2000 ca sa se suprapuna pe axa X
        norm_dates = [d['date'].replace(year=2000) for d in entries]
        values     = [d['value'] for d in entries]

        ax.plot(norm_dates, values,
                color=color, linewidth=1.8,
                marker=marker, markersize=3,
                label=str(year))

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(month_ticks())
    ax.set_xticklabels(MONTH_ABBR, fontsize=10)
    ax.legend(title='An', loc='upper left', ncol=5)
    ax.grid(True, linestyle='--', alpha=0.5)
    if show_zero:
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

plt.tight_layout()
plt.savefig('slovakia_biweekly_2020_2024.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graficele salvate in slovakia_biweekly_2020_2024.png")