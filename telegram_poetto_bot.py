#!/usr/bin/env python3
import os
import time
import random
import asyncio
import tempfile
from dataclasses import dataclass
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18
import requests
from urllib.parse import urljoin, urlparse
import logging
import datetime
from zoneinfo import ZoneInfo
import gdown

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.constants import ChatAction
from telegram.error import BadRequest, Forbidden, NetworkError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
    PicklePersistence,
    filters,
)

from get_wind_info import LAT, LON, get_wind
from sea_conditions import (
    CALM_WAVE_HEIGHT,
    ROUGH_WAVE_HEIGHT,
    MarineNow,
    SlotForecast,
    describe_waves,
    get_forecast,
    get_marine_now,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
# httpx logs every request URL at INFO, and Telegram URLs contain the bot token
logging.getLogger("httpx").setLevel(logging.WARNING)


# TELEGRAM TOKEN
from pathlib import Path

# Load .env file if exists
env_path = Path(".env")
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
    print("Loaded environment variables from .env")
else:
    print(".env not found, reading environment variables directly")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("⚠️ TELEGRAM_TOKEN not found! Set it in .env (local) or in environment variables (production)")

print("Telegram token loaded successfully")


# CONFIG
M3U8_URL = "https://cdn-002.whatsupcams.com/hls/it_cagliari01.m3u8"
DOWNLOAD_DIR = "./temp"  # each analysis works in its own subfolder
DATA_DIR = "./data"
PERSISTENCE_PATH = os.path.join(DATA_DIR, "bot_state.pickle")
TIMEZONE = ZoneInfo("Europe/Rome")
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "r3d_18_poetto.pth")
MODEL_GDRIVE_ID = "1ASd7VdbSrXXs9fa3EpHFed0W2NlUugq9"
TIMEOUT = 15
RETRIES = 2

FRAME_COUNT = 16
ROI = (430, 600, 430 + 1400, 600 + 350)
# ROI = (730, 300, 730 + 1150, 300 + 350) # new ROI based on recent cam movement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ["mare_calmo", "mare_mosso"]

# A single 6s segment gives noisy predictions: average over the most recent ones
N_SEGMENTS = 3
# Below this averaged confidence the result is reported as uncertain
CONFIDENCE_THRESHOLD = 0.70
UNCERTAIN_LABEL = "incerto"

# Random messages
MESSAGES_CALM = [
    "🏊 Perfetto per nuotare! Il mare è bellissimo oggi!",
    "☀️ Condizioni ideali! Corri al mare!",
    "🌊 Giornata fantastica per un bagno!",
    "😎 Le onde sono miti, via al nuoto!",
    "🏖️ Mare tranquillo, è il momento di divertirsi!",
]

MESSAGES_CALM += [
    "🏖️ Il mare è una tavola! Perfetto per un tuffo da manuale!",
    "😎 Onde? Zero. Relax? Totale. Costume e via!",
    "☀️ Mare calmissimo, ti aspetta per una nuotata da sogno!",
    "🐚 È così tranquillo che potresti quasi dormire a galla!",
    "🌊 Condizioni top: acqua limpida, zero scuse… tutti in acqua!",
]

MESSAGES_CALM += [
    "🏊‍♂️ Mare piatto come una piscina! Tuffati senza pensarci!",
    "🌴 Acqua calma e cristallina… sembra una pubblicità della Maldive!",
    "🦀 Anche i granchi oggi si rilassano: mare perfetto!",
    "💦 Nemmeno un’onda all’orizzonte… condizioni da manuale!",
    "☕ Mare così tranquillo che potresti portarti la tazzina e bere un caffè in acqua!",
]

MESSAGES_ROUGH = [
    "⚠️ Mare mosso, meglio non nuotare oggi...",
    "🌪️ Onde forti, rimanda il bagno!",
    "⛔ Condizioni pericolose, stai a terra!",
    "😢 Peccato! Il mare è agitato oggi.",
    "🚫 Mare turbolento, non è sicuro nuotare.",
]

MESSAGES_ROUGH += [
    "🌪️ Oggi il mare sembra un frullatore… meglio rimandare il tuffo!",
    "⚠️ Onde agitate in arrivo! Il mare ha deciso di fare il DJ oggi.",
    "🚫 Non è il giorno giusto per sfidare Nettuno… resta a terra!",
    "😬 Mare un po’ nervoso oggi. Meglio lasciarlo sbollire!",
    "🌊 Oggi il mare è in modalità 'lavatrice': bagno rimandato!",
]

MESSAGES_ROUGH += [
    "🌊 Oggi il mare ha il suo caratterino… meglio lasciarlo sfogare!",
    "😅 Le onde fanno più rumore del vicino con la motosega… niente bagno!",
    "🌀 Se entri oggi, esci domani… e forse in un’altra spiaggia!",
    "🐉 Mare agitato: sembra che ci sia un drago sott’acqua!",
    "🚷 Il mare oggi non è dell’umore giusto… meglio il lettino e un gelato!",
]

MESSAGES_UNCERTAIN = [
    "🤔 Il mare oggi è difficile da leggere… dai un'occhiata tu prima di tuffarti!",
    "🧐 Condizioni incerte: guarda il video e decidi tu!",
    "🌫️ Il modello non è sicuro… meglio controllare di persona!",
    "🤷 Né calmo né mosso: il mare oggi fa il misterioso!",
]


# SETTINGS
# Per-chat settings live in context.chat_data (saved to disk by PicklePersistence):
#   'notify_time': "HH:MM" or None (daily notification disabled)
#   'output_format': "frame" or "video"
DEFAULT_OUTPUT_FORMAT = "video"
QUICK_NOTIFY_TIMES = ["07:00", "08:00", "09:00", "10:00"]

BOT_COMMANDS = [
    BotCommand("analizza", "Analizza il mare adesso"),
    BotCommand("previsioni", "Onde e vento previsti per oggi e domani"),
    BotCommand("notifica", "Imposta la notifica giornaliera"),
    BotCommand("stop", "Disattiva la notifica giornaliera"),
    BotCommand("output", "Scegli foto o video"),
    BotCommand("start", "Mostra aiuto e impostazioni"),
]

# Chats with an analysis in progress, to ignore repeated requests
chats_in_progress: set[int] = set()
# One analysis at a time: the model runs on CPU and parallel runs only slow each other down
analysis_lock = asyncio.Lock()

# SETUP DIRECTORIES
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def download_model_from_gdrive():
    """Download model from Google Drive if not present"""
    if os.path.exists(MODEL_PATH):
        log.info(f"Model already exists at {MODEL_PATH}")
        return True
    
    log.info(f"Model not found at {MODEL_PATH}. Downloading from Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            log.info("Model downloaded successfully!")
            return True
        else:
            log.error("Model download failed")
            return False
    except Exception as e:
        log.error(f"Error downloading model: {e}")
        return False


# SETUP MODEL
log.info("Checking model availability...")
if not download_model_from_gdrive():
    raise RuntimeError("Failed to download model from Google Drive")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# No pretrained weights: they would be downloaded at every start and then overwritten by ours
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
log.info("Model loaded successfully")


# UTILITY FUNCTIONS
def fetch_text(url):
    """Fetch text content from URL with retries"""
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            log.error(f"Fetch error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None


def parse_m3u8_for_ts(m3u8_text, base_url):
    """Parse M3U8 playlist to extract .ts file URLs"""
    lines = [l.strip() for l in m3u8_text.splitlines() if l.strip() and not l.startswith("#")]
    ts_urls = []
    for l in lines:
        if l.lower().endswith(".ts"):
            ts_urls.append(l if bool(urlparse(l).netloc) else urljoin(base_url, l))
    return ts_urls


def download_file(url, out_path):
    """Download file from URL with retries"""
    for attempt in range(RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            log.error(f"Download error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return False


def download_segments(out_dir: str, n_segments: int = N_SEGMENTS) -> list[str]:
    """Download the n most recent segments from M3U8 stream into out_dir.

    Returns local paths, most recent first (empty list on failure).
    """
    text = fetch_text(M3U8_URL)
    if not text:
        return []
    base = M3U8_URL.rsplit("/", 1)[0] + "/"
    ts_urls = parse_m3u8_for_ts(text, base)
    if not ts_urls:
        return []

    # In a live playlist the most recent segments are at the end
    recent_urls = list(reversed(ts_urls[-n_segments:]))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_paths = []
    for index, url in enumerate(recent_urls):
        out_path = os.path.join(out_dir, f"{timestamp}_{index}_cagliari01.ts")
        if download_file(url, out_path):
            video_paths.append(out_path)
        else:
            log.warning(f"Skipping segment {url}: download failed")
    log.info(f"Downloaded {len(video_paths)}/{len(recent_urls)} segments")
    return video_paths


def load_video(video_path, frames_per_clip=FRAME_COUNT, crop_region=ROI):
    """Load and preprocess video for model inference"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video not readable")
    
    indices = torch.linspace(0, total_frames - 1, frames_per_clip).long().tolist()
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            if crop_region:
                x1, y1, x2, y2 = crop_region
                frame = frame[y1:y2, x1:x2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
    cap.release()
    frames = torch.stack(frames).permute(1, 0, 2, 3)
    return frames.unsqueeze(0)


def predict_probs(video_path: str) -> torch.Tensor:
    """Run model inference on a single video, return class probabilities (shape [n_classes])"""
    video_tensor = load_video(video_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs[0].cpu()


def predict(video_paths: list[str]) -> tuple[str, float, str]:
    """Average class probabilities over several videos.

    Returns (label, confidence, top_class): label is UNCERTAIN_LABEL when the
    averaged confidence of top_class is below CONFIDENCE_THRESHOLD.
    """
    if not video_paths:
        raise ValueError("No videos to analyze")

    probs_per_video = []
    for video_path in video_paths:
        probs = predict_probs(video_path)
        probs_text = ", ".join(f"{name}={p:.2f}" for name, p in zip(class_names, probs.tolist()))
        log.info(f"Segment {video_path}: {probs_text}")
        probs_per_video.append(probs)

    mean_probs = torch.stack(probs_per_video).mean(dim=0)
    top_index = int(torch.argmax(mean_probs))
    top_class = class_names[top_index]
    confidence = mean_probs[top_index].item()
    label = top_class if confidence >= CONFIDENCE_THRESHOLD else UNCERTAIN_LABEL
    log.info(f"Averaged over {len(video_paths)} segments: {top_class} ({confidence:.2f}) -> {label}")
    return label, confidence, top_class


def extract_frame(video_path):
    """Extract middle frame from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    frame_path = video_path.replace(".ts", "_frame.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def convert_ts_to_mp4(ts_path):
    """Convert .ts video to .mp4 format"""
    mp4_path = ts_path.replace(".ts", ".mp4")
    log.info(f"Converting {ts_path} to MP4...")
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-i', ts_path, '-c:v', 'libx264', '-c:a', 'aac', '-y', mp4_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log.info(f"Conversion complete: {mp4_path}")
            return mp4_path
        else:
            log.error(f"FFmpeg error: {result.stderr}")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(ts_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            log.info(f"Conversion complete (OpenCV fallback): {mp4_path}")
            return mp4_path
    except Exception as e:
        log.error(f"Conversion error: {e}")
        return None


# ANALYSIS PIPELINE (blocking: run it in a worker thread)
@dataclass
class SeaAnalysis:
    """Result of one sea analysis"""
    label: str  # class name or UNCERTAIN_LABEL
    confidence: float  # averaged probability of top_class
    top_class: str
    n_segments: int
    media_path: str  # frame (.jpg) or video (.mp4) to send to the user
    wind_text: str | None
    marine: MarineNow | None


def analyze_sea(work_dir: str, output_format: str) -> SeaAnalysis:
    """Download recent segments into work_dir, run the model and prepare the media to send.

    Raises RuntimeError if the download or the media preparation fails.
    """
    video_paths = download_segments(work_dir)
    if not video_paths:
        raise RuntimeError("Failed to download video segments")

    label, confidence, top_class = predict(video_paths)

    # Show the user the most recent segment
    latest_video = video_paths[0]
    if output_format == "frame":
        media_path = extract_frame(latest_video)
    else:
        media_path = convert_ts_to_mp4(latest_video)
    if media_path is None:
        raise RuntimeError(f"Failed to prepare {output_format} from {latest_video}")

    wind_text = get_wind(LAT, LON)
    marine = get_marine_now(LAT, LON)
    return SeaAnalysis(label, confidence, top_class, len(video_paths), media_path, wind_text, marine)


# SETTINGS AND MESSAGES
OUTPUT_FORMAT_NAMES = {"frame": "foto", "video": "video"}


def get_chat_settings(chat_data: dict) -> dict:
    """Fill missing settings with defaults; returns the same chat_data dict"""
    chat_data.setdefault("notify_time", None)
    chat_data.setdefault("output_format", DEFAULT_OUTPUT_FORMAT)
    return chat_data


def describe_notify_time(notify_time: str | None) -> str:
    """Human readable notification status"""
    if notify_time is None:
        return "non attiva"
    return f"ogni giorno alle {notify_time}"


def notify_confirmation(notify_time: str | None) -> str:
    """Reply text after the notification time has been changed"""
    if notify_time is None:
        return "🔕 Notifica giornaliera disattivata. Riattivala con /notifica"
    return f"✓ Riceverai l'analisi del mare ogni giorno alle {notify_time}. Disattivala con /stop"


def format_label(class_name: str) -> str:
    """'mare_calmo' -> 'MARE CALMO'"""
    return class_name.replace("_", " ").upper()


def build_caption(analysis: SeaAnalysis, scheduled_time: str | None = None) -> str:
    """Build the HTML caption sent with the frame/video.

    scheduled_time is set only for daily notifications, to remind how to turn them off.
    """
    if analysis.label == UNCERTAIN_LABEL:
        random_msg = random.choice(MESSAGES_UNCERTAIN)
        tendency = format_label(analysis.top_class).lower()
        result_text = f"<b>INCERTO</b>\nTendenza: {tendency} ({analysis.confidence * 100:.0f}%)"
    elif "calmo" in analysis.label:
        random_msg = random.choice(MESSAGES_CALM)
        result_text = f"<b>{format_label(analysis.label)}</b>\nAffidabilità: {analysis.confidence * 100:.0f}%"
    else:
        random_msg = random.choice(MESSAGES_ROUGH)
        result_text = f"<b>{format_label(analysis.label)}</b>\nAffidabilità: {analysis.confidence * 100:.0f}%"

    lines = ["🌊 <b>Spiaggia del Poetto - Cagliari</b>", "", result_text, f"📹 Media su {analysis.n_segments} clip"]
    if analysis.marine:
        lines.append(format_marine_now(analysis.marine))
    if analysis.wind_text:
        lines.append(f"💨 {analysis.wind_text}")
    mismatch_note = marine_mismatch_note(analysis.label, analysis.marine)
    if mismatch_note:
        lines += ["", mismatch_note]
    lines += ["", random_msg]
    if scheduled_time:
        lines += ["", f"⏰ Notifica giornaliera delle {scheduled_time} · /stop per disattivarla"]
    return "\n".join(lines)


def format_marine_now(marine: MarineNow) -> str:
    """'🌡️ Acqua 23 °C · 〰️ Onde 0.5 m (poco mosso)'"""
    waves_text = f"〰️ Onde {marine.wave_height:.1f} m ({describe_waves(marine.wave_height)})"
    if marine.sea_temperature is None:
        return waves_text
    return f"🌡️ Acqua {marine.sea_temperature:.0f} °C · {waves_text}"


def marine_mismatch_note(label: str, marine: MarineNow | None) -> str | None:
    """Warn when the webcam model and the Open-Meteo wave estimate clearly disagree"""
    if marine is None:
        return None
    if "calmo" in label and marine.wave_height >= ROUGH_WAVE_HEIGHT:
        return f"⚠️ Le previsioni stimano onde da {marine.wave_height:.1f} m: guarda bene il video prima di entrare!"
    if "mosso" in label and marine.wave_height < CALM_WAVE_HEIGHT:
        return f"ℹ️ Le previsioni stimano onde di soli {marine.wave_height:.1f} m: guarda il video per conferma."
    return None


def day_label(day: datetime.date, today: datetime.date) -> str:
    """'Oggi', 'Domani' or 'dd/mm'"""
    if day == today:
        return "Oggi"
    if day == today + datetime.timedelta(days=1):
        return "Domani"
    return day.strftime("%d/%m")


def format_forecast(slots: list[SlotForecast], marine: MarineNow | None, today: datetime.date) -> str:
    """HTML message for /previsioni"""
    lines = ["🔮 <b>Previsioni Poetto</b>", ""]
    for slot in slots:
        lines.append(
            f"<b>{day_label(slot.day, today)} {slot.slot_name}</b>: {describe_waves(slot.max_wave_height)}, "
            f"onde fino a {slot.max_wave_height:.1f} m · vento {slot.mean_wind_speed:.0f} km/h {slot.wind_direction}"
        )
    if marine and marine.sea_temperature is not None:
        lines += ["", f"🌡️ Acqua adesso: {marine.sea_temperature:.0f} °C"]
    lines += ["", "<i>Stime Open-Meteo al largo, non osservazioni a riva. Per com'è adesso usa /analizza</i>"]
    return "\n".join(lines)


def parse_notify_time(text: str) -> str | None:
    """Parse '7', '7:30', '07.30' or '7 30' into 'HH:MM'; None if not a valid time"""
    cleaned = text.strip().replace(".", ":").replace(" ", ":")
    parts = cleaned.split(":")
    if len(parts) == 1:
        parts.append("0")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def notify_keyboard() -> InlineKeyboardMarkup:
    """Quick choices for the daily notification time"""
    time_buttons = [InlineKeyboardButton(t, callback_data=f"notify:{t}") for t in QUICK_NOTIFY_TIMES]
    off_button = InlineKeyboardButton("🔕 Disattiva", callback_data="notify:off")
    return InlineKeyboardMarkup([time_buttons, [off_button]])


def again_keyboard() -> InlineKeyboardMarkup:
    """Button attached to each result to run a new analysis"""
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Analizza di nuovo", callback_data="inference_now")]])


# SCHEDULING
def daily_job_name(chat_id: int) -> str:
    """Name of the daily notification job of a chat"""
    return f"daily_inference_{chat_id}"


def remove_daily_job(job_queue: JobQueue, chat_id: int) -> None:
    """Remove the daily notification job of a chat, if any"""
    for job in job_queue.get_jobs_by_name(daily_job_name(chat_id)):
        job.schedule_removal()


def schedule_daily_job(job_queue: JobQueue, chat_id: int, notify_time: str) -> None:
    """(Re)schedule the daily notification at notify_time ("HH:MM", Europe/Rome time).

    run_daily follows the local clock, so DST changes don't shift the notification.
    """
    remove_daily_job(job_queue, chat_id)
    hour_text, minute_text = notify_time.split(":")
    job_queue.run_daily(
        scheduled_inference,
        time=datetime.time(int(hour_text), int(minute_text), tzinfo=TIMEZONE),
        chat_id=chat_id,
        name=daily_job_name(chat_id),
    )
    log.info(f"Daily notification for chat {chat_id} scheduled at {notify_time}")


def update_notify_time(context: ContextTypes.DEFAULT_TYPE, chat_id: int, notify_time: str | None) -> None:
    """Save the notification time of a chat and (re)schedule or remove its job"""
    settings = get_chat_settings(context.chat_data)
    settings["notify_time"] = notify_time
    if notify_time is None:
        remove_daily_job(context.job_queue, chat_id)
        log.info(f"Daily notification for chat {chat_id} disabled")
    else:
        schedule_daily_job(context.job_queue, chat_id, notify_time)


# TELEGRAM HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    settings = get_chat_settings(context.chat_data)
    log.info(f"Bot started by chat_id: {update.effective_chat.id}")
    await update.message.reply_text(
        "🌊 <b>Poetto Sea State Bot</b>\n\n"
        "Ti dico com'è il mare al Poetto analizzando la webcam in tempo reale.\n\n"
        "<b>Comandi:</b>\n"
        "/analizza - Analizza il mare adesso\n"
        "/previsioni - Onde e vento previsti per oggi e domani\n"
        "/notifica - Ricevi l'analisi ogni giorno a un orario\n"
        "/stop - Disattiva la notifica giornaliera\n"
        "/output - Scegli se ricevere foto o video\n\n"
        "📊 <b>Le tue impostazioni:</b>\n"
        f"• Formato: {OUTPUT_FORMAT_NAMES[settings['output_format']]}\n"
        f"• Notifica: {describe_notify_time(settings['notify_time'])}",
        parse_mode="HTML",
    )


async def inference(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analizza (and legacy /inference) command"""
    chat_id = update.effective_chat.id
    log.info(f"Inference requested by chat_id: {chat_id}")
    await run_inference(context, chat_id)


async def forecast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /previsioni command"""
    now = datetime.datetime.now(TIMEZONE)
    slots = await asyncio.to_thread(get_forecast, LAT, LON, now)
    if not slots:
        await update.message.reply_text("❌ Previsioni non disponibili al momento, riprova più tardi.")
        return
    marine = await asyncio.to_thread(get_marine_now, LAT, LON)
    await update.message.reply_text(format_forecast(slots, marine, now.date()), parse_mode="HTML")


async def notify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /notifica (and legacy /settings) command"""
    settings = get_chat_settings(context.chat_data)
    context.user_data["awaiting_notify_time"] = True
    await update.message.reply_text(
        f"⏰ Notifica attuale: {describe_notify_time(settings['notify_time'])}\n\n"
        "Scegli un orario o scrivilo tu (es. 7:30):",
        reply_markup=notify_keyboard(),
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command"""
    update_notify_time(context, update.effective_chat.id, None)
    await update.message.reply_text(notify_confirmation(None))


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free text: notification time after /notifica, otherwise a short hint"""
    if not context.user_data.get("awaiting_notify_time"):
        await update.message.reply_text("Usa /analizza per vedere com'è il mare adesso, o /start per l'aiuto.")
        return

    notify_time = parse_notify_time(update.message.text)
    if notify_time is None:
        await update.message.reply_text("❌ Orario non valido. Scrivilo come 7:30 oppure 18:00")
        return
    context.user_data.pop("awaiting_notify_time", None)
    update_notify_time(context, update.effective_chat.id, notify_time)
    await update.message.reply_text(notify_confirmation(notify_time))


async def output_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /output command"""
    settings = get_chat_settings(context.chat_data)
    keyboard = [
        [InlineKeyboardButton("🖼️ Foto", callback_data="output:frame")],
        [InlineKeyboardButton("🎬 Video", callback_data="output:video")],
    ]
    await update.message.reply_text(
        f"Formato attuale: <b>{OUTPUT_FORMAT_NAMES[settings['output_format']]}</b>\n\nCome vuoi ricevere il risultato?",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard callbacks ("action:value")"""
    query = update.callback_query
    chat_id = update.effective_chat.id
    settings = get_chat_settings(context.chat_data)
    action, _, value = query.data.partition(":")
    await query.answer()

    if action == "output" and value in OUTPUT_FORMAT_NAMES:
        settings["output_format"] = value
        await query.edit_message_text(text=f"✓ Formato: {OUTPUT_FORMAT_NAMES[value]}")
    elif action == "notify":
        context.user_data.pop("awaiting_notify_time", None)
        notify_time = None if value == "off" else value
        update_notify_time(context, chat_id, notify_time)
        await query.edit_message_text(text=notify_confirmation(notify_time))
    elif action == "inference_now":
        # Remove the button so the same result can't be tapped twice
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except BadRequest as e:
            log.warning(f"Could not remove button for chat {chat_id}: {e}")
        await run_inference(context, chat_id)
    else:
        log.warning(f"Unknown callback data from chat {chat_id}: {query.data}")


# INFERENCE
async def send_analysis(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, analysis: SeaAnalysis, scheduled_time: str | None
) -> None:
    """Send the frame or video with the result caption"""
    caption = build_caption(analysis, scheduled_time)
    is_photo = analysis.media_path.endswith(".jpg")
    chat_action = ChatAction.UPLOAD_PHOTO if is_photo else ChatAction.UPLOAD_VIDEO
    await context.bot.send_chat_action(chat_id=chat_id, action=chat_action)

    with open(analysis.media_path, "rb") as media_file:
        if is_photo:
            await context.bot.send_photo(
                chat_id=chat_id, photo=media_file, caption=caption, parse_mode="HTML", reply_markup=again_keyboard()
            )
        else:
            await context.bot.send_video(
                chat_id=chat_id,
                video=media_file,
                caption=caption,
                parse_mode="HTML",
                supports_streaming=True,
                reply_markup=again_keyboard(),
            )


async def run_inference(context: ContextTypes.DEFAULT_TYPE, chat_id: int, scheduled_time: str | None = None) -> None:
    """Run the full analysis for a chat and send the result"""
    if chat_id in chats_in_progress:
        await context.bot.send_message(chat_id=chat_id, text="⏳ Sto già analizzando il mare, un attimo di pazienza…")
        return

    chats_in_progress.add(chat_id)
    settings = get_chat_settings(context.chat_data)
    status_message = None
    try:
        status_message = await context.bot.send_message(
            chat_id=chat_id, text="⏳ Scarico le immagini della webcam e analizzo il mare…"
        )
        # Per-request folder: concurrent analyses never touch each other's files
        with tempfile.TemporaryDirectory(dir=DOWNLOAD_DIR) as work_dir:
            async with analysis_lock:
                analysis = await asyncio.to_thread(analyze_sea, work_dir, settings["output_format"])
            log.info(f"Chat {chat_id}: {analysis.label} ({analysis.confidence * 100:.1f}%)")
            await send_analysis(context, chat_id, analysis, scheduled_time)
    except Forbidden:
        # The user blocked the bot: stop sending daily notifications
        log.warning(f"Chat {chat_id} blocked the bot, disabling its daily notification")
        update_notify_time(context, chat_id, None)
    except Exception:
        # Top-level boundary of the handler: log the full traceback and tell the user
        log.exception(f"Inference failed for chat {chat_id}")
        await context.bot.send_message(
            chat_id=chat_id, text="❌ Non sono riuscito ad analizzare il mare. Riprova tra qualche minuto."
        )
    finally:
        chats_in_progress.discard(chat_id)
        if status_message is not None:
            await delete_message_quietly(status_message)


async def delete_message_quietly(message) -> None:
    """Delete a status message, logging (not raising) if Telegram refuses"""
    try:
        await message.delete()
    except (BadRequest, Forbidden) as e:
        log.warning(f"Could not delete status message: {e}")


# SCHEDULED JOB
async def scheduled_inference(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run inference at the scheduled time"""
    chat_id = context.job.chat_id
    settings = get_chat_settings(context.chat_data)
    log.info(f"Daily notification at {settings['notify_time']} for chat {chat_id}")
    await run_inference(context, chat_id, scheduled_time=settings["notify_time"])


async def post_init(application: Application) -> None:
    """Restore daily jobs from persisted chat settings and register the command menu"""
    restored = 0
    for chat_id, chat_data in application.chat_data.items():
        notify_time = chat_data.get("notify_time")
        if notify_time:
            schedule_daily_job(application.job_queue, chat_id, notify_time)
            restored += 1
    log.info(f"Restored {restored} daily notifications from {PERSISTENCE_PATH}")

    # The command menu is a nicety: a slow Telegram API must not stop the bot from starting
    try:
        await application.bot.set_my_commands(BOT_COMMANDS)
    except NetworkError as e:
        log.warning(f"Could not register the command menu: {e}")


# MAIN
if __name__ == "__main__":
    log.info("Starting Telegram bot...")
    persistence = PicklePersistence(filepath=PERSISTENCE_PATH)
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .persistence(persistence)
        # Without this, one running analysis blocks every other user's update
        .concurrent_updates(True)
        .post_init(post_init)
        .build()
    )
    if app.job_queue is None:
        raise RuntimeError("Job queue not available. Install with: pip install python-telegram-bot[job-queue]")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(["analizza", "inference"], inference))
    app.add_handler(CommandHandler("previsioni", forecast_command))
    app.add_handler(CommandHandler(["notifica", "settings"], notify_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("output", output_format))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    log.info("✓ Bot running. Waiting for commands...")
    app.run_polling()
