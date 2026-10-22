import logging

import requests

log = logging.getLogger(__name__)

# Coordinate del Poetto di Cagliari
LAT = 39.194167 
LON = 9.160833

# Funzione per ottenere il vento
def get_wind(lat: float, lon: float) -> str | None:
    """
    Restituisce velocità e direzione del vento per le coordinate specificate,
    oppure None se i dati non sono disponibili (il bot in quel caso li omette).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&windspeed_unit=kmh"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather = data.get('current_weather')
        if weather:
            wind_speed = weather['windspeed']  # km/h
            wind_deg = weather['winddirection']  # gradi

            # Converti gradi in direzione cardinali
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
            idx = round(wind_deg / 45)
            direction_text = directions[idx]

            return f"Vento al Poetto: {wind_speed} km/h, Direzione: {direction_text} ({wind_deg}°)"
        else:
            log.warning("Open-Meteo response without current_weather")
            return None
    except requests.RequestException as e:
        log.warning(f"Wind request failed: {e}")
        return None

# Esempio di utilizzo
if __name__ == "__main__":
    print(get_wind(LAT, LON))
