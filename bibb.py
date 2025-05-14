#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice-Driven Name Cycler with CSV Logging

Listens for the phrase "who is he" on the default microphone.
Cycles through replies: Vaibhav → Manas → Dev → Vaibhav → …
Logs every interaction to interactions.csv with:
    - Timestamp
    - Recognized text
    - Reply given

Requirements:
    pip install SpeechRecognition pyttsx3 pyaudio
"""

import os
import sys
import time
import signal
import logging
import threading
import queue
import csv
from datetime import datetime

import speech_recognition as sr
import pyttsx3


class Config:
    """
    Holds all configuration parameters for the VoiceCycler.
    Extendable to load from external files if desired.
    """
    PHRASE = "who is he"

    REPLIES = ["Vaibhav", "Manas", "Dev"]

    CSV_FILE = "interactions.csv"

    LOG_FILE = "voice_cycler.log"

    AMBIENT_DURATION = 0.8

    LISTEN_PHRASE_TIME = 5.0

    TTS_RATE = 150    
    TTS_VOLUME = 1.0 

    MAX_RETRY = 3

    QUEUE_MAXSIZE = 32



def setup_logging(log_file: str) -> logging.Logger:
    """
    Initialize and return a logger that writes to both console and file.

    :param log_file: Path to the log file.
    :return: Configured Logger object.
    """
    logger = logging.getLogger("VoiceCycler")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.debug("Logger initialized.")
    return logger


class CSVLogger:
    """
    Handles creating and appending interaction records to a CSV file.
    """

    def __init__(self, file_path: str, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger

        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "recognized_text", "reply"])
                self.logger.info(f"Created new CSV file with header: {self.file_path}")
            except Exception as e:
                self.logger.error(f"Failed to create CSV file: {e}")

    def log_interaction(self, recognized_text: str, reply: str):
        """
        Append a single interaction record to the CSV.

        :param recognized_text: The raw text recognized from speech.
        :param reply: The reply that was given.
        """
        timestamp = datetime.now().isoformat()
        try:
            with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, recognized_text, reply])
            self.logger.debug(f"Logged CSV row: [{timestamp}, {recognized_text}, {reply}]")
        except Exception as e:
            self.logger.error(f"Error writing to CSV: {e}")



class TTSEngine:
    """
    Wraps the pyttsx3 engine to speak text asynchronously.
    """

    def __init__(self, rate: int, volume: float, logger: logging.Logger):
        self.logger = logger
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self.lock = threading.Lock()
            self.logger.info(f"TTS Engine initialized (rate={rate}, volume={volume})")
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            raise

    def speak(self, text: str):
        """
        Speak text in a background thread to avoid blocking the main loop.
        """
        def _speak_task():
            with self.lock:
                self.logger.info(f"TTS speaking: {text}")
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    self.logger.error(f"TTS engine error: {e}")

        thread = threading.Thread(target=_speak_task, daemon=True)
        thread.start()



class VoiceCycler:
    """
    Manages microphone listening, speech recognition, phrase detection,
    reply cycling, TTS feedback, and CSV logging.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize recognizer, microphone, TTS, CSV logger, and internal state.
        """
        self.config = config
        self.logger = logger
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.tts = TTSEngine(rate=config.TTS_RATE,
                             volume=config.TTS_VOLUME,
                             logger=logger)
        self.csv_logger = CSVLogger(file_path=config.CSV_FILE,
                                    logger=logger)
        self.count = 0
        self.running = False
        self.audio_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.stop_listening = None

        self._install_signal_handlers()

        self._calibrate_ambient()

    def _install_signal_handlers(self):
        """
        Set up SIGINT and SIGTERM handlers to stop the cycler gracefully.
        """

        def _handler(signum, frame):
            self.logger.info(f"Received signal {signum}, stopping...")
            self.running = False

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self.logger.debug("Signal handlers installed for SIGINT/SIGTERM.")

    def _calibrate_ambient(self):
        """
        Listen to the microphone for a short time to adjust for ambient noise.
        """
        with self.mic as source:
            self.logger.info(f"Calibrating microphone for ambient noise "
                             f"({self.config.AMBIENT_DURATION}s)...")
            self.recognizer.adjust_for_ambient_noise(source,
                                                     duration=self.config.AMBIENT_DURATION)
            self.logger.info("Ambient noise calibration complete.")

    def listen_in_background(self):
        """
        Start background listening. Audio snippets get queued for processing.
        """
        def callback(recognizer, audio):
            """
            Called by the background listener with captured audio.
            """
            try:
            
                self.audio_queue.put_nowait(audio)
                self.logger.debug("Audio snippet queued.")
            except queue.Full:
                self.logger.warning("Audio queue full; dropping snippet.")

        self.stop_listening = self.recognizer.listen_in_background(
            self.mic,
            callback,
            phrase_time_limit=self.config.LISTEN_PHRASE_TIME
        )
        self.logger.info("Background listening started.")

    def _process_audio(self, audio: sr.AudioData):
        """
        Process a single AudioData snippet:
            - Recognize text (with retries)
            - If trigger phrase detected, cycle reply, speak, log CSV.
        """

        recognized_text = ""
        for attempt in range(1, self.config.MAX_RETRY + 1):
            try:
                recognized_text = self.recognizer.recognize_google(audio)
                recognized_text = recognized_text.lower()
                self.logger.debug(f"Recognized (attempt {attempt}): {recognized_text}")
                break
            except sr.UnknownValueError:
                self.logger.warning("Could not understand audio.")
                return
            except sr.RequestError as e:
                self.logger.error(f"API request error on attempt {attempt}: {e}")
                if attempt == self.config.MAX_RETRY:
                    return
                time.sleep(1)

       
        if self.config.PHRASE in recognized_text:
           
            reply = self._get_next_reply()
            log_msg = f"Trigger phrase detected. Replying with '{reply}'."
            self.logger.info(log_msg)
            print(f"🤖 → {reply}")

            
            try:
                self.tts.speak(reply)
            except Exception as e:
                self.logger.error(f"TTS speak failed: {e}")

            try:
                self.csv_logger.log_interaction(recognized_text, reply)
            except Exception as e:
                self.logger.error(f"CSV logging failed: {e}")

    def process_queue(self):
        """
        Continuously process queued audio snippets until stopped.
        """
        self.logger.debug("Starting processing loop.")
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            threading.Thread(target=self._process_audio,
                             args=(audio,),
                             daemon=True).start()

        self.logger.debug("Exiting processing loop.")

    def _get_next_reply(self) -> str:
        """
        Return the next name in the cycle.
        """
        idx = self.count % len(self.config.REPLIES)
        reply = self.config.REPLIES[idx]
        self.logger.debug(f"Cycle index: {self.count} → '{reply}'")
        self.count += 1
        return reply

    def run(self):
        """
        Start the cycler: set running flag, launch listening and processing.
        """
        self.running = True
        self.logger.info("VoiceCycler starting up.")
        print("Microphone ready. Say “who is he” to get a name. (Ctrl+C to exit)\n")

        self.listen_in_background()
        try:
            self.process_queue()
        finally:
            
            if self.stop_listening:
                self.stop_listening(wait_for_stop=False)
            self.logger.info("VoiceCycler shut down.")

def main():
    """
    Initialize configuration, logger, and VoiceCycler, then run.
    """
   
    try:
        _ = pyttsx3.init()
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}")
        sys.exit(1)

    config = Config()
    logger = setup_logging(config.LOG_FILE)

   
    cycler = VoiceCycler(config, logger)
    cycler.run()

if __name__ == "__main__":
    main()
