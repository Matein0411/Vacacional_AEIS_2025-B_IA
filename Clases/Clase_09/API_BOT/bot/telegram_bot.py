import aiohttp
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL")

FEATURES = [
    "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability", "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max", "histogram_number_of_peaks",
    "histogram_number_of_zeroes", "histogram_mean", "histogram_variance", "histogram_tendency"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """🤖 Bot de Predicción de Salud Fetal

Envía 19 valores separados por comas:
📝 Ejemplo: 120,0.002,0.0,0.006,0.003,0.0,0.0,73,0.5,43,2.4,64,62,126,2,0,136,641,1"""
    await update.message.reply_text(message)

async def predict_fetal_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        values = [x.strip() for x in update.message.text.strip().split(',')]
        
        if len(values) != 19:
            await update.message.reply_text(f"❌ Necesito 19 valores, recibí {len(values)}")
            return
        
        try:
            float_values = [float(v) for v in values]
        except ValueError:
            await update.message.reply_text("❌ Todos los valores deben ser números")
            return
        
        data = dict(zip(FEATURES, float_values))
        msg = await update.message.reply_text("🔄 Procesando...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(API_URL, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        emoji = {"Normal": "✅", "Suspect": "⚠️", "Pathological": "🚨"}
                        
                        text = f"""{emoji.get(result['prediction_label'], '🤖')} **{result['prediction_label']}**
🎯 Confianza: {result['confidence']:.1%}"""
                        
                        await msg.edit_text(text, parse_mode='Markdown')
                    else:
                        await msg.edit_text(f"❌ Error API: {response.status}")
            except Exception as e:
                await msg.edit_text(f"❌ Error: {str(e)}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict_fetal_health))
    
    print("🤖 Bot iniciado")
    print(f"🔗 API: {API_URL}")
    app.run_polling()

if __name__ == "__main__":
    main()
