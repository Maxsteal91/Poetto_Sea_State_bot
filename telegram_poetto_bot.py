#!/usr/bin/env python3
import os
import time
import random
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import requests
from urllib.parse import urljoin, urlparse
import logging
import shutil
import datetime
import pytz

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# CONFIG
TELEGRAM_TOKEN = "8253440827:AAGj4EPEnfSpUGFPTuyr3T0W-GbXqI9hSTM"
M3U8_URL = "https://cdn-002.whatsupcams.com/hls/it_cagliari01.m3u8"
DOWNLOAD_DIR = "./temp"
MODEL_PATH = "./model/r3d_18_poetto.pth"
TIMEOUT = 15
RETRIES = 2

FRAME_COUNT = 16
ROI = (430, 600, 430 + 1400, 600 + 350)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ["mare_calmo", "mare_mosso"]

# Messaggi casuali
MESSAGES_CALM = [
    "🏊 Perfetto per nuotare! Il mare è bellissimo oggi!",
    "☀️ Condizioni ideali! Corri al mare!",
    "🌊 Giornata fantastica per un bagno!",
    "😎 Le onde sono miti, via al nuoto!",
    "🏖️ Mare tranquillo, è il momento di divertirsi!",
]

MESSAGES_ROUGH = [
    "⚠️ Mare mosso, meglio non nuotare oggi...",
    "🌪️ Onde forti, rimanda il bagno!",
    "⛔ Condizioni pericolose, stai a terra!",
    "😢 Peccato! Il mare è agitato oggi.",
    "🚫 Mare turbolento, non è sicuro nuotare.",
]

# STATE
user_chat_id = None
user_notify_hour = 9
user_notify_minute = 30
user_output_format = "frame"
scheduled_job_id = None

# SETUP MODELLO
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# UTILITY
def fetch_text(url):
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            log.error(f"Errore fetch (tentativo {attempt+1}): {e}")
            time.sleep(1)
    return None

def parse_m3u8_for_ts(m3u8_text, base_url):
    lines = [l.strip() for l in m3u8_text.splitlines() if l.strip() and not l.startswith("#")]
    ts_urls = []
    for l in lines:
        if l.lower().endswith(".ts"):
            ts_urls.append(l if bool(urlparse(l).netloc) else urljoin(base_url, l))
    return ts_urls

def download_file(url, out_path):
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
            log.error(f"Errore download (tentativo {attempt+1}): {e}")
            time.sleep(1)
    return False

def download_video():
    text = fetch_text(M3U8_URL)
    if not text:
        return None
    base = M3U8_URL.rsplit("/", 1)[0] + "/"
    ts_urls = parse_m3u8_for_ts(text, base)
    if not ts_urls:
        return None
    
    url = random.choice(ts_urls)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DOWNLOAD_DIR, f"{timestamp}_cagliari01.ts")
    if download_file(url, out_path):
        return out_path
    return None

def load_video(video_path, frames_per_clip=FRAME_COUNT, crop_region=ROI):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video non leggibile")
    
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

def predict(video_path):
    video_tensor = load_video(video_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return class_names[pred_class], probs[0][pred_class].item()

def extract_frame(video_path):
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

def cleanup_temp():
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def convert_ts_to_mp4(ts_path):
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

# TELEGRAM HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_chat_id
    user_chat_id = update.effective_chat.id
    log.info(f"Bot avviato da user_id: {user_chat_id}")
    
    await update.message.reply_text(
        "🌊 <b>Sea Monitoring Bot</b>\n\n"
        "Monitora lo stato del mare in tempo reale.\n\n"
        "<b>Comandi disponibili:</b>\n"
        "/inference - Analizza il mare adesso\n"
        "/settings - Imposta orario notifica automatica\n"
        "/output - Scegli formato output (frame/video)\n"
        "/start - Mostra questo messaggio\n\n"
        f"📊 <b>Stato attuale:</b>\n"
        f"• Output: {user_output_format.upper()}\n"
        f"• Notifica fissa: {user_notify_hour:02d}:{user_notify_minute:02d}",
        parse_mode="HTML"
    )

async def inference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_chat_id
    user_chat_id = update.effective_chat.id
    log.info(f"Inferenza richiesta da user_id: {user_chat_id}")
    await update.message.reply_text("⏳ Downloading and analyzing...")
    await run_inference(context, is_scheduled=False)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"⏰ Orario attuale: {user_notify_hour:02d}:{user_notify_minute:02d}\n\nInserisci l'ora (0-23):"
    )
    context.user_data['setting_mode'] = 'hour'

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_notify_hour, user_notify_minute, scheduled_job_id
    
    if 'setting_mode' not in context.user_data:
        return
    
    try:
        value = int(update.message.text)
        
        if context.user_data['setting_mode'] == 'hour':
            if not 0 <= value <= 23:
                await update.message.reply_text("❌ Inserisci un'ora valida (0-23)")
                return
            user_notify_hour = value
            context.user_data['setting_mode'] = 'minute'
            await update.message.reply_text(f"✓ Ora: {value:02d}\n\nInserisci i minuti (0-59):")
        
        elif context.user_data['setting_mode'] == 'minute':
            if not 0 <= value <= 59:
                await update.message.reply_text("❌ Inserisci minuti validi (0-59)")
                return
            user_notify_minute = value
            del context.user_data['setting_mode']
            
            # Rimuovi tutti i job precedenti
            for job in context.job_queue.jobs():
                if job.name == "scheduled_inference":
                    job.schedule_removal()
            
            # Calcola secondi fino al prossimo orario
            tz = pytz.timezone('Europe/Rome')
            now = datetime.datetime.now(tz)
            next_run = now.replace(hour=user_notify_hour, minute=user_notify_minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += datetime.timedelta(days=1)
            delay = (next_run - now).total_seconds()
            
            job = context.job_queue.run_repeating(
                scheduled_inference,
                interval=86400,
                first=delay,
                name="scheduled_inference"
            )
            
            log.info(f"Notifica aggiornata alle {user_notify_hour:02d}:{user_notify_minute:02d} (ora italiana)")
            await update.message.reply_text(f"✓ Notifica fissa aggiornata alle {user_notify_hour:02d}:{user_notify_minute:02d}\n\n📅 La prossima notifica arriverà a quell'orario.")
    except ValueError:
        await update.message.reply_text("❌ Inserisci un numero valido")

async def output_format(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_output_format
    keyboard = [
        [InlineKeyboardButton("🖼️ Frame", callback_data="output_frame")],
        [InlineKeyboardButton("🎬 Video", callback_data="output_video")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Output attuale: <b>{user_output_format.upper()}</b>\n\nScegli formato:",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_output_format
    query = update.callback_query
    await query.answer()
    
    if query.data == "output_frame":
        user_output_format = "frame"
        await query.edit_message_text(text="✓ Output: Frame")
    elif query.data == "output_video":
        user_output_format = "video"
        await query.edit_message_text(text="✓ Output: Full Video (MP4)")
    elif query.data == "inference_now":
        await query.edit_message_text(text="⏳ Analyzing...")
        await run_inference(context, is_scheduled=False)

# INFERENZA
async def run_inference(context: ContextTypes.DEFAULT_TYPE, is_scheduled=True):
    try:
        log.info("Starting video download...")
        video_path = download_video()
        if not video_path:
            log.error("Video download failed")
            await context.bot.send_message(chat_id=user_chat_id, text="❌ Error: Failed to download video")
            return
        
        log.info(f"Video downloaded: {video_path}. Starting inference...")
        pred, prob = predict(video_path)
        log.info(f"Inference complete: {pred} ({prob*100:.1f}%)")
        
        # Scegli messaggio random
        if "calmo" in pred.lower():
            random_msg = random.choice(MESSAGES_CALM)
        else:
            random_msg = random.choice(MESSAGES_ROUGH)
        
        message = f"🌊 <b>Spiaggia Poetto - Cagliari</b>\n\n<b>{pred.upper()}</b>\n{prob*100:.1f}% confidence\n\n{random_msg}\n\n⏰ Prossima notifica: {user_notify_hour:02d}:{user_notify_minute:02d}"
        
        if user_output_format == "frame":
            frame_path = extract_frame(video_path)
            with open(frame_path, "rb") as f:
                await context.bot.send_photo(chat_id=user_chat_id, photo=f, caption=message, parse_mode="HTML")
        else:
            mp4_path = convert_ts_to_mp4(video_path)
            if mp4_path:
                with open(mp4_path, "rb") as f:
                    await context.bot.send_video(chat_id=user_chat_id, video=f, caption=message, parse_mode="HTML")
            else:
                await context.bot.send_message(chat_id=user_chat_id, text="❌ Error: Failed to convert video")
        
        # Aggiungi pulsante per inferenza immediata
        if not is_scheduled:
            keyboard = [[InlineKeyboardButton("🔄 Inferenza ora", callback_data="inference_now")]]
            await context.bot.send_message(
                chat_id=user_chat_id,
                text="Vuoi fare un'altra analisi?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    except Exception as e:
        log.error(f"Inference error: {e}")
        await context.bot.send_message(chat_id=user_chat_id, text=f"❌ Error: {str(e)}")
    finally:
        log.info("Cleaning temp folder...")
        cleanup_temp()

# JOB PERIODICO
async def scheduled_inference(context: ContextTypes.DEFAULT_TYPE):
    if user_chat_id:
        log.info(f"Notifica automatica alle {user_notify_hour:02d}:{user_notify_minute:02d}")
        await run_inference(context, is_scheduled=True)

# MAIN
if __name__ == "__main__":
    log.info("Avvio bot Telegram...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("inference", inference))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("output", output_format))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Job queue per notifiche a orario fisso
    job_queue = app.job_queue
    if job_queue:
        tz = pytz.timezone('Europe/Rome')
        now = datetime.datetime.now(tz)
        next_run = now.replace(hour=user_notify_hour, minute=user_notify_minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += datetime.timedelta(days=1)
        delay = (next_run - now).total_seconds()
        
        job = job_queue.run_repeating(
            scheduled_inference,
            interval=86400,
            first=delay,
            name="scheduled_inference"
        )
        log.info(f"✓ Job scheduler configurato: notifica alle {user_notify_hour:02d}:{user_notify_minute:02d} (ora italiana)")
    else:
        log.warning("Job queue non disponibile. Installa con: pip install python-telegram-bot[job-queue]")
    
    log.info("✓ Bot in esecuzione. In attesa di comandi...")
    app.run_polling()