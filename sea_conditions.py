"""Condizioni meteo-marine al Poetto da Open-Meteo: onde, temperatura dell'acqua, vento previsto.

Sono stime di modello su una griglia al largo, non osservazioni a riva:
servono come contesto e controllo incrociato del modello sulla webcam.
"""

import datetime
import logging
from collections import Counter
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10
TIMEZONE_NAME = "Europe/Rome"
FORECAST_DAYS = 2

# Soglie indicative di altezza d'onda (m) per descrivere il mare a parole
CALM_WAVE_HEIGHT = 0.5
ROUGH_WAVE_HEIGHT = 1.0

# Fasce orarie delle previsioni: (ora di inizio inclusa, ora di fine esclusa)
FORECAST_SLOTS = {"mattina": (8, 13), "pomeriggio": (13, 19)}

CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]


@dataclass
class MarineNow:
    """Condizioni marine attuali"""

    wave_height: float  # m
    wave_period: float  # s
    sea_temperature: float | None  # °C


@dataclass
class SlotForecast:
    """Previsione aggregata per una fascia oraria di un giorno"""

    day: datetime.date
    slot_name: str
    max_wave_height: float  # m
    mean_wind_speed: float  # km/h
    wind_direction: str  # direzione cardinale più frequente nella fascia


def fetch_json(url: str, params: dict) -> dict | None:
    """GET su Open-Meteo; restituisce il JSON o None in caso di errore (loggato)"""
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.warning(f"Open-Meteo request failed ({url}): {e}")
    except ValueError as e:
        log.warning(f"Open-Meteo returned invalid JSON ({url}): {e}")
    return None


def degrees_to_cardinal(degrees: float) -> str:
    """Converte gradi (0-360) in direzione cardinale (N, NE, ...)"""
    return CARDINAL_DIRECTIONS[round(degrees % 360 / 45)]


def describe_waves(wave_height: float) -> str:
    """Descrizione a parole dell'altezza d'onda"""
    if wave_height < CALM_WAVE_HEIGHT:
        return "calmo"
    if wave_height < ROUGH_WAVE_HEIGHT:
        return "poco mosso"
    return "mosso"


def get_marine_now(lat: float, lon: float) -> MarineNow | None:
    """Onde e temperatura dell'acqua attuali, None se non disponibili"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wave_height,wave_period,sea_surface_temperature",
        "timezone": TIMEZONE_NAME,
    }
    data = fetch_json(MARINE_URL, params)
    current = data.get("current") if data else None
    if not current or current.get("wave_height") is None:
        log.warning(f"Marine data not available: {data}")
        return None
    return MarineNow(
        wave_height=current["wave_height"],
        wave_period=current.get("wave_period") or 0.0,
        sea_temperature=current.get("sea_surface_temperature"),
    )


def get_hourly_forecast(lat: float, lon: float) -> list[dict] | None:
    """Previsione oraria unita (onde + vento) per FORECAST_DAYS giorni.

    Ogni elemento: {"time": datetime, "wave_height", "wind_speed", "wind_direction"}.
    """
    common = {"latitude": lat, "longitude": lon, "forecast_days": FORECAST_DAYS, "timezone": TIMEZONE_NAME}
    marine = fetch_json(MARINE_URL, {**common, "hourly": "wave_height"})
    weather = fetch_json(
        FORECAST_URL, {**common, "hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "kmh"}
    )
    if not marine or not weather or "hourly" not in marine or "hourly" not in weather:
        return None

    wind_by_time = {}
    weather_hourly = weather["hourly"]
    for index, time_text in enumerate(weather_hourly["time"]):
        wind_by_time[time_text] = (
            weather_hourly["wind_speed_10m"][index],
            weather_hourly["wind_direction_10m"][index],
        )

    hours = []
    marine_hourly = marine["hourly"]
    for index, time_text in enumerate(marine_hourly["time"]):
        wave_height = marine_hourly["wave_height"][index]
        wind = wind_by_time.get(time_text)
        # Ore senza dati (null nel JSON) vengono saltate
        if wave_height is None or wind is None or None in wind:
            continue
        hours.append(
            {
                "time": datetime.datetime.fromisoformat(time_text),
                "wave_height": wave_height,
                "wind_speed": wind[0],
                "wind_direction": wind[1],
            }
        )
    return hours


def aggregate_slots(hours: list[dict], now: datetime.datetime) -> list[SlotForecast]:
    """Raggruppa le ore per giorno e fascia (FORECAST_SLOTS), saltando le fasce già finite"""
    slots = []
    days = sorted({hour["time"].date() for hour in hours})
    for day in days:
        for slot_name, (start_hour, end_hour) in FORECAST_SLOTS.items():
            if day == now.date() and now.hour >= end_hour:
                continue
            slot_hours = [h for h in hours if h["time"].date() == day and start_hour <= h["time"].hour < end_hour]
            if not slot_hours:
                continue
            wind_speeds = [h["wind_speed"] for h in slot_hours]
            directions = Counter(degrees_to_cardinal(h["wind_direction"]) for h in slot_hours)
            slots.append(
                SlotForecast(
                    day=day,
                    slot_name=slot_name,
                    max_wave_height=max(h["wave_height"] for h in slot_hours),
                    mean_wind_speed=sum(wind_speeds) / len(wind_speeds),
                    wind_direction=directions.most_common(1)[0][0],
                )
            )
    return slots


def get_forecast(lat: float, lon: float, now: datetime.datetime) -> list[SlotForecast] | None:
    """Previsioni per fasce orarie da adesso a fine domani, None se non disponibili"""
    hours = get_hourly_forecast(lat, lon)
    if hours is None:
        return None
    return aggregate_slots(hours, now)


if __name__ == "__main__":
    from zoneinfo import ZoneInfo

    from get_wind_info import LAT, LON

    logging.basicConfig(level=logging.INFO)
    print(get_marine_now(LAT, LON))
    for slot in get_forecast(LAT, LON, datetime.datetime.now(ZoneInfo(TIMEZONE_NAME))) or []:
        print(slot)
