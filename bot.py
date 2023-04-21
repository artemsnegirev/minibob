import os
import logging

from typing import List
from aiogram import Bot, Dispatcher, executor, types

from minibob.inference import InferencePipe


TG_API_TOKEN = os.environ.get('TG_BOT_TOKEN')
assert TG_API_TOKEN is not None, 'Add telegram token to launch bot'

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_CACHE_DIR = os.environ.get('HF_CACHE_DIR')
HF_MODEL_NAME = os.environ.get('HF_MODEL_NAME')

logging.basicConfig(level=logging.INFO)

pipe = InferencePipe(HF_MODEL_NAME, HF_TOKEN, HF_CACHE_DIR)

bot = Bot(token=TG_API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer(f"Мини-Боб: привет, загадай слово и напиши мне его описание, а я попробую отгадать =)")

@dp.message_handler()
async def user_query(message: types.Message):
    answer_candidates: List[str] = pipe(message.text)

    await message.answer(f"Мини-Боб: мои предположения - {', '.join(answer_candidates)}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)