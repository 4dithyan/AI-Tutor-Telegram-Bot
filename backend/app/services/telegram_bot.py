import os
import uuid
import asyncio
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from app.config import settings
from app.services.vision import vision_service
from app.services.document_processor import clean_text, extract_text_from_pdf
from app.services.chunker import semantic_chunk_text
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.services.rag import process_rag_query, generate_notes_from_docs
from app.services.quiz_service import classify_topics, generate_quiz
from app.utils.logging import api_logger

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TelegramBotManager:
    def __init__(self):
        self.token = settings.telegram_bot_token
        self.app: Optional[Application] = None
        self.quiz_sessions = {}

    async def start(self):
        if not self.token:
            api_logger.warning("No TELEGRAM_BOT_TOKEN provided. Telegram bot will not start.")
            return

        api_logger.info("Initializing Telegram Bot Service...")
        try:
            self.app = Application.builder().token(self.token).build()

            # Command Handlers
            self.app.add_handler(CommandHandler("start", self._start_cmd))
            self.app.add_handler(CommandHandler("help", self._help_cmd))
            self.app.add_handler(CommandHandler("notes", self._notes_cmd))
            self.app.add_handler(CommandHandler("list", self._list_cmd))
            self.app.add_handler(CommandHandler("docs", self._list_cmd))
            self.app.add_handler(CommandHandler("clear", self._clear_cmd))
            self.app.add_handler(CommandHandler("quiz", self._quiz_cmd))
            self.app.add_handler(CommandHandler("topics", self._topics_cmd))

            # Callback & Message Handlers
            self.app.add_handler(CallbackQueryHandler(self._handle_callback))
            self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
            self.app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

            # Initialize & start polling in background
            await self.app.initialize()
            await self.app.start()

            # Delete any conflicting external webhooks
            try:
                await self.app.bot.delete_webhook(drop_pending_updates=True)
                api_logger.info("Cleared conflicting Telegram webhooks successfully.")
            except Exception as e:
                api_logger.warning(f"Could not delete webhook: {e}")

            # Register Telegram Bot Command Menu
            try:
                commands = [
                    BotCommand("start", "Start AI Tutor Assistant"),
                    BotCommand("topics", "View Chapters & Topics"),
                    BotCommand("quiz", "Take Interactive Quiz"),
                    BotCommand("notes", "Generate Study Notes"),
                    BotCommand("list", "View Uploaded Documents"),
                    BotCommand("clear", "Clear My Documents"),
                    BotCommand("help", "Show Help Menu")
                ]
                await self.app.bot.set_my_commands(commands)
                api_logger.info("Telegram Bot Command Menu registered successfully!")
            except Exception as e:
                api_logger.warning(f"Failed to set bot command menu: {e}")

            await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            api_logger.info("Telegram Bot is running and listening for messages!")
        except Exception as e:
            api_logger.error(f"Failed to start Telegram Bot: {e}")

    async def stop(self):
        if self.app:
            api_logger.info("Stopping Telegram Bot Service...")
            try:
                if self.app.updater and self.app.updater.running:
                    await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                api_logger.info("Telegram Bot stopped cleanly.")
            except Exception as e:
                api_logger.error(f"Error stopping Telegram Bot: {e}")

    async def _start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "🎓 <b>Welcome to AI Tutor Agent Bot!</b>\n\n"
            '👋 I am your AI Textbook Tutor developed by <a href="https://adithyan-portfolio.pages.dev">Adithyan</a>.\n\n'
            "<b>What you can do:</b>\n"
            "📷 <b>Send a Photo</b> of a textbook page\n"
            "📄 <b>Send a PDF Document</b> (up to 30MB) of a textbook or chapter\n"
            "💬 <b>Ask Any Question</b> about your uploaded materials\n"
            "🏷️ <b>Type /topics</b> to view classified chapters & topics\n"
            "🎯 <b>Type /quiz</b> to take an interactive topic quiz\n"
            "📚 <b>Type /list</b> to view & delete your uploaded documents\n"
            "📝 <b>Type /notes</b> to generate structured study notes\n"
            "🗑️ <b>Type /clear</b> to delete all your uploaded materials\n\n"
            "Send an image or PDF to get started!"
        )
        await update.message.reply_text(welcome_text, parse_mode="HTML", disable_web_page_preview=True)

    async def _help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._start_cmd(update, context)

    async def _quiz_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        status_msg = await update.message.reply_text("🔎 Analyzing textbook topics for quiz selection...")
        try:
            loop = asyncio.get_running_loop()
            topics = await loop.run_in_executor(None, classify_topics, user_id)
            
            keyboard = [[InlineKeyboardButton("🎯 Overall Textbook Quiz", callback_data="picktopic:Overall")]]
            for t in topics:
                title = t["title"]
                keyboard.append([InlineKeyboardButton(f"📘 {title[:35]}", callback_data=f"picktopic:{title}")])
                
            reply_markup = InlineKeyboardMarkup(keyboard)
            await status_msg.edit_text("🎯 <b>Select a Topic for your Quiz:</b>", reply_markup=reply_markup, parse_mode="HTML")
        except Exception as e:
            await status_msg.edit_text(f"❌ Failed to fetch topics: {str(e)}")

    async def _topics_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        status_msg = await update.message.reply_text("🏷️ Classifying textbook topics...")
        try:
            loop = asyncio.get_running_loop()
            topics = await loop.run_in_executor(None, classify_topics, user_id)
            if not topics:
                await status_msg.edit_text("📂 No textbook documents uploaded yet. Send a photo or PDF to extract topics!")
                return
            
            msg = "🏷️ <b>Classified Textbook Topics & Chapters:</b>\n\n"
            keyboard = []
            for idx, t in enumerate(topics, 1):
                msg += f"{idx}. 📘 <b>{t['title']}</b>\n"
                keyboard.append([InlineKeyboardButton(f"Quiz: {t['title'][:30]}", callback_data=f"picktopic:{t['title']}")])
            
            keyboard.insert(0, [InlineKeyboardButton("🎯 Take Overall Quiz", callback_data="picktopic:Overall")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await status_msg.edit_text(msg, reply_markup=reply_markup, parse_mode="HTML")
        except Exception as e:
            await status_msg.edit_text(f"❌ Failed to classify topics: {str(e)}")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            pass

        user_id = f"tg_{update.effective_user.id}"
        data = query.data

        # 1. Pick Topic -> Ask Question Count
        if data.startswith("picktopic:"):
            topic_name = data.split("picktopic:", 1)[1]
            keyboard = [
                [
                    InlineKeyboardButton("✨ Auto", callback_data=f"startquiz:{topic_name}:0"),
                    InlineKeyboardButton("5 Qs", callback_data=f"startquiz:{topic_name}:5"),
                    InlineKeyboardButton("10 Qs", callback_data=f"startquiz:{topic_name}:10"),
                    InlineKeyboardButton("15 Qs", callback_data=f"startquiz:{topic_name}:15")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f"🎯 Topic: <b>{topic_name}</b>\n\nSelect question count (or ✨ Auto to adapt to content size):",
                reply_markup=reply_markup,
                parse_mode="HTML"
            )

        # 2. Start Quiz -> Generate & Render Question 1
        elif data.startswith("startquiz:"):
            parts = data.split(":")
            topic_name = parts[1]
            count = int(parts[2])
            
            label_text = f"{count}-question" if count > 0 else "✨ Auto-adaptive"
            await query.edit_message_text(f"⏳ Generating <b>{label_text} quiz</b> for <b>{topic_name}</b>... Please wait.", parse_mode="HTML")
            
            try:
                loop = asyncio.get_running_loop()
                quiz_data = await loop.run_in_executor(None, generate_quiz, topic_name, count, user_id)
                questions = quiz_data.get("questions", [])
                
                if not questions:
                    await query.edit_message_text("❌ Could not generate quiz questions. Please upload more textbook content!")
                    return
                
                self.quiz_sessions[user_id] = {
                    "topic": topic_name,
                    "count": len(questions),
                    "questions": questions,
                    "current_idx": 0,
                    "score": 0
                }
                
                await self._render_quiz_question(query, user_id)
            except Exception as e:
                api_logger.error(f"Telegram quiz generation error: {e}")
                await query.edit_message_text(f"❌ Quiz generation error: {str(e)}")

        # 3. Answer Validation
        elif data.startswith("ansquiz:"):
            parts = data.split(":")
            q_idx = int(parts[1])
            selected_key = parts[2]
            
            session = self.quiz_sessions.get(user_id)
            if not session or q_idx >= len(session["questions"]):
                await query.edit_message_text("⚠️ Quiz session expired. Type /quiz to start a new one.")
                return
            
            q = session["questions"][q_idx]
            raw_corr = str(q.get("correct") or q.get("answer") or q.get("correct_answer") or "A").strip().upper()
            correct_key = "A"
            for k in ["A", "B", "C", "D"]:
                if k in raw_corr:
                    correct_key = k
                    break
            
            is_correct = (selected_key == correct_key)
            if is_correct:
                session["score"] += 1
                header = "✅ <b>Correct!</b>"
            else:
                header = f"❌ <b>Incorrect.</b> (Correct answer: <b>{correct_key}</b>)"
                
            msg = f"{header}\n\n"
            msg += f"<b>Q{q_idx + 1}. {q['question']}</b>\n\n"
            msg += f"Your Choice: <b>{selected_key}</b>\n\n"
            msg += f"💡 <i>Explanation:</i> {q.get('explanation', '')}"
            
            keyboard = []
            if q_idx + 1 < session["count"]:
                keyboard.append([InlineKeyboardButton("Next Question ➔", callback_data=f"nextq:{q_idx + 1}")])
            else:
                keyboard.append([InlineKeyboardButton("See Results 🏆", callback_data="finishquiz")])
                
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="HTML")

        # 4. Next Question
        elif data.startswith("nextq:"):
            q_idx = int(data.split(":")[1])
            session = self.quiz_sessions.get(user_id)
            if not session:
                await query.edit_message_text("⚠️ Quiz session expired. Type /quiz to start a new one.")
                return
            session["current_idx"] = q_idx
            await self._render_quiz_question(query, user_id)

        # 5. Finish Quiz -> Score Card
        elif data == "finishquiz":
            session = self.quiz_sessions.get(user_id)
            if not session:
                await query.edit_message_text("⚠️ Quiz session expired. Type /quiz to start a new one.")
                return
                
            total = session["count"]
            score = session["score"]
            percentage = round((score / total) * 100)
            
            msg = (
                f"🎉 <b>Quiz Completed!</b>\n\n"
                f"Topic: <b>{session['topic']}</b>\n"
                f"Final Score: <b>{score} / {total}</b> ({percentage}%)\n\n"
            )
            if percentage >= 80:
                msg += "🌟 Excellent work! You mastered this topic."
            elif percentage >= 50:
                msg += "👍 Good job! Review the topic notes to score 100%."
            else:
                msg += "📖 Keep practicing! Upload more textbook notes to study."
                
            keyboard = [
                [InlineKeyboardButton("🔄 Retake Quiz", callback_data=f"startquiz:{session['topic']}:{total}")],
                [InlineKeyboardButton("📘 Select New Topic", callback_data="picktopic:Overall")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="HTML")

        # Delete Document Callback
        elif data.startswith("deldoc:"):
            doc_id = data.split("deldoc:", 1)[1]
            try:
                vector_store.delete_document(doc_id)
                await query.edit_message_text("🗑️ <b>Document deleted from your knowledge base!</b>", parse_mode="HTML")
            except Exception as e:
                await query.edit_message_text(f"❌ Failed to delete document: {str(e)}")

    async def _render_quiz_question(self, query, user_id: str):
        session = self.quiz_sessions.get(user_id)
        if not session:
            await query.edit_message_text("⚠️ Quiz session expired.")
            return
            
        q_idx = session["current_idx"]
        q = session["questions"][q_idx]
        total = session["count"]
        
        msg = f"🎯 <b>Question {q_idx + 1} of {total}</b> (Topic: <i>{session['topic']}</i>)\n\n"
        msg += f"<b>{q['question']}</b>\n\n"
        
        options = q.get("options", [])
        option_keys = ["A", "B", "C", "D"]
        
        # Display full options formatted cleanly in the message body
        for i, opt in enumerate(options[:4]):
            opt_key = option_keys[i]
            opt_clean = str(opt).strip()
            if opt_clean.startswith(f"{opt_key})") or opt_clean.startswith(f"{opt_key}."):
                opt_clean = opt_clean[2:].strip()
            msg += f"<b>{opt_key})</b> {opt_clean}\n\n"
            
        msg += "<i>Tap your answer option below:</i>"
        
        # Clean prominent option buttons
        keyboard = [
            [
                InlineKeyboardButton("Option A", callback_data=f"ansquiz:{q_idx}:A"),
                InlineKeyboardButton("Option B", callback_data=f"ansquiz:{q_idx}:B")
            ],
            [
                InlineKeyboardButton("Option C", callback_data=f"ansquiz:{q_idx}:C"),
                InlineKeyboardButton("Option D", callback_data=f"ansquiz:{q_idx}:D")
            ]
        ]
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="HTML")

    async def _list_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        try:
            docs = vector_store.list_documents(user_id=user_id)
            if not docs:
                await update.message.reply_text("📂 No documents uploaded to your knowledge base yet.\n\nSend a photo or PDF file to get started!")
                return
            
            msg = "📚 *Your Uploaded Textbook Documents:*\n\n"
            keyboard = []
            for idx, d in enumerate(docs, 1):
                msg += f"{idx}. 📄 `{d['filename']}`\n"
                keyboard.append([InlineKeyboardButton(f"🗑️ Delete {d['filename'][:20]}", callback_data=f"deldoc:{d['document_id']}")])
            
            msg += "\nYou can ask me questions about any of your documents!"
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to list documents: {str(e)}")

    async def _clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        try:
            vector_store.clear_user_documents(user_id=user_id)
            await update.message.reply_text("🗑️ *Your knowledge base has been cleared!* All your uploaded documents and vector chunks have been deleted.", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to clear knowledge base: {str(e)}")

    async def _notes_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        status_msg = await update.message.reply_text("📝 Generating study notes from your uploaded materials... Please wait.")
        try:
            loop = asyncio.get_running_loop()
            notes = await loop.run_in_executor(None, generate_notes_from_docs, None, None, user_id)
            
            # Send in chunks if notes exceed Telegram length limit (4000 chars)
            if len(notes) > 4000:
                for i in range(0, len(notes), 4000):
                    await update.message.reply_text(notes[i:i+4000])
            else:
                await status_msg.edit_text(notes)
        except Exception as e:
            api_logger.error(f"Telegram notes command error: {e}")
            await status_msg.edit_text(f"❌ Failed to generate notes: {str(e)}")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        status_msg = await update.message.reply_text("📷 Image received! Processing content with Vision OCR model...")
        
        try:
            photo = update.message.photo[-1]
            tg_file = await photo.get_file()
            
            filename = f"tg_photo_{uuid.uuid4().hex[:8]}.jpg"
            file_path = os.path.join(UPLOAD_DIR, filename)
            await tg_file.download_to_drive(file_path)
            
            doc_id = str(uuid.uuid4())
            loop = asyncio.get_running_loop()
            
            # Run vision extraction and indexing
            def process_pipeline():
                raw_text = vision_service.extract_text_from_image(file_path)
                if not raw_text.strip():
                    raise ValueError("No text extracted from image.")
                cleaned = clean_text(raw_text)
                chunks = semantic_chunk_text(cleaned)
                embeddings = embedding_service.embed_texts(chunks)
                vector_store.insert_chunks(
                    document_id=doc_id,
                    original_filename=filename,
                    chunks=chunks,
                    embeddings=embeddings,
                    user_id=user_id
                )
                return len(chunks)
            
            chunks_count = await loop.run_in_executor(None, process_pipeline)
            await status_msg.edit_text(
                f"✅ *Successfully processed photo!*\n\n"
                f"📌 Extracted & indexed *{chunks_count} content chunks* into your personal knowledge base.\n\n"
                f"You can now ask me any question about this content!",
                parse_mode="Markdown"
            )
        except Exception as e:
            api_logger.error(f"Telegram photo processing error: {e}")
            await status_msg.edit_text(f"❌ Processing failed: {str(e)}")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = f"tg_{update.effective_user.id}"
        doc = update.message.document
        orig_filename = doc.file_name or f"doc_{uuid.uuid4().hex[:8]}.pdf"
        
        # Check size limit (30MB)
        if doc.file_size and doc.file_size > (settings.max_upload_size_mb * 1024 * 1024):
            await update.message.reply_text(f"❌ File `{orig_filename}` ({round(doc.file_size / (1024*1024), 1)}MB) exceeds maximum limit of {settings.max_upload_size_mb}MB.", parse_mode="Markdown")
            return

        status_msg = await update.message.reply_text(f"📄 Document *{orig_filename}* ({round((doc.file_size or 0)/(1024*1024), 1)}MB) received! Processing...", parse_mode="Markdown")
        
        try:
            tg_file = await doc.get_file()
            filename = f"tg_{uuid.uuid4().hex[:6]}_{orig_filename}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            await tg_file.download_to_drive(file_path)
            
            doc_id = str(uuid.uuid4())
            loop = asyncio.get_running_loop()
            
            def process_pipeline():
                if filename.lower().endswith(".pdf"):
                    raw_text = extract_text_from_pdf(file_path)
                else:
                    raw_text = vision_service.extract_text_from_image(file_path)
                    
                if not raw_text.strip():
                    raise ValueError("No text extracted from document.")
                cleaned = clean_text(raw_text)
                chunks = semantic_chunk_text(cleaned)
                embeddings = embedding_service.embed_texts(chunks)
                vector_store.insert_chunks(
                    document_id=doc_id,
                    original_filename=orig_filename,
                    chunks=chunks,
                    embeddings=embeddings,
                    user_id=user_id
                )
                return len(chunks)
            
            chunks_count = await loop.run_in_executor(None, process_pipeline)
            await status_msg.edit_text(
                f"✅ *Successfully processed document `{orig_filename}`!*\n\n"
                f"📌 Extracted & indexed *{chunks_count} content chunks* into your personal knowledge base.\n\n"
                f"You can now ask me questions based on this textbook!",
                parse_mode="Markdown"
            )
        except Exception as e:
            api_logger.error(f"Telegram document processing error: {e}")
            await status_msg.edit_text(f"❌ Document processing failed: {str(e)}")

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        question = update.message.text.strip()
        if not question:
            return
        
        user_id = f"tg_{update.effective_user.id}"
        lower_q = question.lower()
        
        # Check simple greetings
        if lower_q in ["hi", "hello", "hey", "hlo", "hiii", "/start"]:
            welcome_reply = (
                '👋 <b>Hello! I am AI Tutor developed by <a href="https://adithyan-portfolio.pages.dev">Adithyan</a>.</b>\n\n'
                'I am your intelligent textbook study assistant!\n\n'
                '💬 <b>Ask me any question directly in chat!</b>\n'
                '📷 <b>Send a photo</b> of a textbook page\n'
                '📄 <b>Send a PDF document</b> (up to 30MB)\n\n'
                'Type /help to view all available commands.'
            )
            await update.message.reply_text(welcome_reply, parse_mode="HTML", disable_web_page_preview=True)
            return
            
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            loop = asyncio.get_running_loop()
            answer, sources_data = await loop.run_in_executor(None, process_rag_query, question, user_id)
            
            reply_text = answer
            if sources_data:
                source_files = sorted(list(set(s["original_filename"] for s in sources_data)))
                reply_text += f"\n\n📚 <b>Sources:</b> {', '.join(source_files)}"
                
            try:
                await update.message.reply_text(reply_text, parse_mode="HTML", disable_web_page_preview=True)
            except Exception:
                await update.message.reply_text(reply_text, disable_web_page_preview=True)
        except Exception as e:
            api_logger.error(f"Telegram text chat error: {e}")
            await update.message.reply_text(f"Sorry, an error occurred: {str(e)}")

telegram_bot_service = TelegramBotManager()
