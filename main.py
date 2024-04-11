from dotenv import load_dotenv
import telebot
import os
from network import MyModel
from YandexGPT import get_response

load_dotenv()

if os.getenv('TOKEN') is not None:
    token = os.environ['TOKEN']
else:
    print("Пожалуйста, сохраните идентификатор телеграмм бота в соответствующую переменную среды TOKEN в файле .env.")
    exit()

bot = telebot.TeleBot(token)
model = MyModel()


@bot.message_handler(commands=['start', 'help'], content_types=['text'])
def start(message):
    bot.send_message(message.chat.id,
                     'Я определяю породу вашей собаки по фотографии.\nОтправьте фотографию вашего питомца.')


@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)

        photo = bot.download_file(file_info.file_path)
        save_path = './photo.jpg'
        with open(save_path, 'wb') as new_file:
            new_file.write(photo)
        bot.send_message(message.chat.id, "Идёт обработка...")

        model.get_predict(save_path)
        os.remove(save_path)
        print(model.predg_label)

        if model.predg_perc > 50:
            bot.reply_to(message,
                         f'''С вероятностью {model.predg_perc} %.\nПорода вашей собаки - {model.predg_label}.''')
            bot.send_message(message.chat.id, get_response(headers, folder_id, model.predg_label))
        else:
            bot.reply_to(message, "Похоже, на вашем изображении отсутствует собака.\nПопробуйте другое изображение.")
    except Exception as e:
        bot.reply_to(message, e)


if __name__ == "__main__":

    if os.getenv('IAM_TOKEN') is not None:
        iam_token = os.environ['IAM_TOKEN']
        headers = {
            "Content-Type": "application/json",
            'Authorization': f'Bearer {iam_token}',
        }
    elif os.getenv('API_KEY') is not None:
        api_key = os.environ['API_KEY']
        headers = {
            "Content-Type": "application/json",
            'Authorization': f'Api-Key {api_key}',
        }
    else:
        print('Пожалуйста, сохраните либо IAM-токен, либо ключ API в соответствующую переменную среды IAM_TOKEN или API_KEY в файле .env.')
        exit()

    if os.getenv('FOLDER_ID') is not None:
        folder_id = os.environ["FOLDER_ID"]
    else:
        print("Пожалуйста, сохраните идентификатор папки в соответствующую переменную среды FOLDER_ID в файле .env.")
        exit()

    bot.infinity_polling()
