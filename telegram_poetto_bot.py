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
import gdown

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


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
DOWNLOAD_DIR = "./temp"
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


# STATE - Per-user settings using dictionaries
user_settings = {}  # {chat_id: {'notify_hour': 9, 'notify_minute': 30, 'output_format': 'frame'}}

# Default settings
DEFAULT_NOTIFY_HOUR = 8
DEFAULT_NOTIFY_MINUTE = 0
DEFAULT_OUTPUT_FORMAT = "video"


def get_user_settings(chat_id):
    """Get or create user settings"""
    if chat_id not in user_settings:
        user_settings[chat_id] = {
            'notify_hour': DEFAULT_NOTIFY_HOUR,
            'notify_minute': DEFAULT_NOTIFY_MINUTE,
            'output_format': DEFAULT_OUTPUT_FORMAT
        }
    return user_settings[chat_id]

# SETUP DIRECTORIES
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
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

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
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


def download_video():
    """Download random video segment from M3U8 stream"""
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


def predict(video_path):
    """Run model inference on video"""
    video_tensor = load_video(video_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return class_names[pred_class], probs[0][pred_class].item()


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


def cleanup_temp():
    """Clean temporary directory"""
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)


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


# TELEGRAM HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    log.info(f"Bot started by user_id: {chat_id}")
    
    await update.message.reply_text(
        "🌊 <b>Sea Monitoring Bot</b>\n\n"
        "Monitor sea conditions in real-time.\n\n"
        "<b>Available commands:</b>\n"
        "/inference - Analyze sea now\n"
        "/settings - Set automatic notification time\n"
        "/output - Choose output format (frame/video)\n"
        "/start - Show this message\n\n"
        f"📊 <b>Current status:</b>\n"
        f"• Output: {settings['output_format'].upper()}\n"
        f"• Scheduled notification: {settings['notify_hour']:02d}:{settings['notify_minute']:02d}",
        parse_mode="HTML"
    )


async def inference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /inference command"""
    chat_id = update.effective_chat.id
    log.info(f"Inference requested by user_id: {chat_id}")
    await update.message.reply_text("⏳ Downloading and analyzing...")
    await run_inference(context, chat_id, is_scheduled=False)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /settings command"""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    await update.message.reply_text(
        f"⏰ Current time: {settings['notify_hour']:02d}:{settings['notify_minute']:02d}\n\nEnter hour (0-23):"
    )
    context.user_data['setting_mode'] = 'hour'


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text input for settings"""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    
    if 'setting_mode' not in context.user_data:
        return
    
    try:
        value = int(update.message.text)
        
        if context.user_data['setting_mode'] == 'hour':
            if not 0 <= value <= 23:
                await update.message.reply_text("❌ Enter a valid hour (0-23)")
                return
            settings['notify_hour'] = value
            context.user_data['setting_mode'] = 'minute'
            await update.message.reply_text(f"✓ Hour: {value:02d}\n\nEnter minutes (0-59):")
        
        elif context.user_data['setting_mode'] == 'minute':
            if not 0 <= value <= 59:
                await update.message.reply_text("❌ Enter valid minutes (0-59)")
                return
            settings['notify_minute'] = value
            del context.user_data['setting_mode']
            
            # Remove all previous jobs for this user
            job_name = f"scheduled_inference_{chat_id}"
            for job in context.job_queue.jobs():
                if job.name == job_name:
                    job.schedule_removal()
            
            # Calculate seconds until next scheduled time
            tz = pytz.timezone('Europe/Rome')
            now = datetime.datetime.now(tz)
            next_run = now.replace(hour=settings['notify_hour'], minute=settings['notify_minute'], second=0, microsecond=0)
            if next_run <= now:
                next_run += datetime.timedelta(days=1)
            delay = (next_run - now).total_seconds()
            
            # Schedule job with user-specific name and data
            context.job_queue.run_repeating(
                scheduled_inference,
                interval=86400,
                first=delay,
                name=job_name,
                chat_id=chat_id,
                data={'chat_id': chat_id}
            )
            
            log.info(f"Notification updated to {settings['notify_hour']:02d}:{settings['notify_minute']:02d} for user {chat_id}")
            await update.message.reply_text(f"✓ Scheduled notification updated to {settings['notify_hour']:02d}:{settings['notify_minute']:02d}\n\n📅 Next notification will arrive at that time.")
    except ValueError:
        await update.message.reply_text("❌ Enter a valid number")


async def output_format(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /output command"""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    keyboard = [
        [InlineKeyboardButton("🖼️ Frame", callback_data="output_frame")],
        [InlineKeyboardButton("🎬 Video", callback_data="output_video")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Current output: <b>{settings['output_format'].upper()}</b>\n\nChoose format:",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks"""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    query = update.callback_query
    await query.answer()
    
    if query.data == "output_frame":
        settings['output_format'] = "frame"
        await query.edit_message_text(text="✓ Output: Frame")
    elif query.data == "output_video":
        settings['output_format'] = "video"
        await query.edit_message_text(text="✓ Output: Full Video (MP4)")
    elif query.data == "inference_now":
        await query.edit_message_text(text="⏳ Analyzing...")
        await run_inference(context, chat_id, is_scheduled=False)


# INFERENCE
async def run_inference(context: ContextTypes.DEFAULT_TYPE, chat_id: int, is_scheduled=True):
    """Run complete inference pipeline"""
    settings = get_user_settings(chat_id)
    
    try:
        log.info("Starting video download...")
        video_path = download_video()
        if not video_path:
            log.error("Video download failed")
            await context.bot.send_message(chat_id=chat_id, text="❌ Error: Failed to download video")
            return
        
        log.info(f"Video downloaded: {video_path}. Starting inference...")
        pred, prob = predict(video_path)
        log.info(f"Inference complete: {pred} ({prob*100:.1f}%)")
        
        # Choose random message
        if "calmo" in pred.lower():
            random_msg = random.choice(MESSAGES_CALM)
        else:
            random_msg = random.choice(MESSAGES_ROUGH)
        
        message = f"🌊 <b>Spiaggia Poetto - Cagliari</b>\n\n<b>{pred.upper()}</b>\n{prob*100:.1f}% confidence\n\n{random_msg}\n\n⏰ Next notification: {settings['notify_hour']:02d}:{settings['notify_minute']:02d}"
        
        if settings['output_format'] == "frame":
            frame_path = extract_frame(video_path)
            with open(frame_path, "rb") as f:
                await context.bot.send_photo(chat_id=chat_id, photo=f, caption=message, parse_mode="HTML")
        else:
            mp4_path = convert_ts_to_mp4(video_path)
            if mp4_path:
                with open(mp4_path, "rb") as f:
                    await context.bot.send_video(chat_id=chat_id, video=f, caption=message, parse_mode="HTML")
            else:
                await context.bot.send_message(chat_id=chat_id, text="❌ Error: Failed to convert video")
        
        # Add button for immediate inference
        if not is_scheduled:
            keyboard = [[InlineKeyboardButton("🔄 Analyze again", callback_data="inference_now")]]
            await context.bot.send_message(
                chat_id=chat_id,
                text="Want to run another analysis?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    except Exception as e:
        log.error(f"Inference error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")
    finally:
        log.info("Cleaning temp folder...")
        cleanup_temp()


# SCHEDULED JOB
async def scheduled_inference(context: ContextTypes.DEFAULT_TYPE):
    """Run inference at scheduled time"""
    chat_id = context.job.chat_id
    settings = get_user_settings(chat_id)
    log.info(f"Automatic notification at {settings['notify_hour']:02d}:{settings['notify_minute']:02d} for user {chat_id}")
    await run_inference(context, chat_id, is_scheduled=True)


# MAIN
if __name__ == "__main__":
    log.info("Starting Telegram bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("inference", inference))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("output", output_format))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Job queue for scheduled notifications - no initial job, users set their own schedules
    job_queue = app.job_queue
    if job_queue:
        log.info("✓ Job scheduler ready (users can set notifications via /settings)")
    else:
        log.warning("Job queue not available. Install with: pip install python-telegram-bot[job-queue]")
    
    log.info("✓ Bot running. Waiting for commands...")
    app.run_polling()