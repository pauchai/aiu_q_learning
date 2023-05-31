'''
В этом задании, вам необходимо обучить агента играть в ViZDoom, в сценарий defend_the_center. Можно использовать лекционный ноутбук в качестве шаблонного кода, но необходимо создать свою нейронную сеть (т.е. немного поменять архитектуру нейронной сети из лекции).
Для выполнение этого задания, вам нужно будет скачать файлы сценария ViZDoom. Файлы можно скачать функцией, представленной в коде ниже.
Эти файлы также можно скачать c официальной репозитории ViZDoom, где они немного отличаются. Поэтому эти файлы рекоммендуется скачать по первой ссылке.
'''
#Сначала, скачиваем все нужные файлы для ViZDoom.
import gdown #Импортируем модуль для скачивания датасетов

#Скачиваем файлы для ViZDoom
#gdown.download('https://storage.yandexcloud.net/aiueducation/Content/advanced/l6/rl.zip', output = None, quiet = True)
#!unzip -qo rl.zip -d /content/Vizdoom/ #Разархивируем файлы
#%%bash - значит используем командную строку (не код питона)
# Устанавливаем нужные программы, которая требует от нас документация: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux

#apt-get update

#apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
#nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
#libopenal-dev timidity libwildmidi-dev unzip

# Устанавливаем библиотеку Boost
#apt-get install libboost-all-dev

# Устанавливаем зависимости Lua
#apt-get install liblua5.1-dev
#!pip install vizdoom==1.1.9
#!pip install imageio==2.4.1
#!pip install imageio==2.5

video_file = "epizode.avi"
#file_weights = "model.h5"
#vizdoom_scenario_file = "./Vizdoom/scenarios/defend_the_center.cfg"
vizdoom_stats_file = 'vizdoom_DQN_stats.txt'

from classes import QLearning, Vizdoom, GameRuntime
from vizdoom import  DoomGame         #Импортируем все функции из cреды ViZDoom
import tensorflow as tf       #Библиотека тензорфлоу
import numpy as np            #Библиотека numpy
import random                 #Импортируем модуль для генерации рандомных значений
import pickle                 #Модуль для сохранения результатов в файл
import cv2                    #Модуль для работы с картинками и видео (нужно для предобработки данных и записи результата)

#Импортируем тип данных deque из встроенного модуля питона

import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential #Импортируем функции для создания и загрузки модели из тензорфлоу
from tensorflow.keras.layers import *                             #Импортируем все слои из кераса
from tensorflow.keras.optimizers import RMSprop                   #Импортируем оптимизатор RMSprop
from tensorflow.keras import backend as K                         #Импортируем модуль для бэкэнда кераса
from tensorflow.keras.utils import to_categorical                 #Импортируем функцию для удобного onehot энкодинга
from moviepy.editor import *                                      #Импортируем полезные функции из библиотеки для возпроизведение видео с результатом
#from google.colab import output                                   #Импортируем функцию для управления вывода в колаб-ячейках
from funcs import show_scores, get_reward

# Функция создания модели для Deep Q-learning
def Make_DQN(
              input_shape,
              actions_size,
              learning_rate,
              
             ):
        model = Sequential()                                                                       
        model.add(Conv2D(32, 8, strides = (4,4), activation = 'relu', input_shape=(input_shape)))  
        model.add(Conv2D(64, 4, strides = (2,2), activation = 'relu'))                             
        #model.add(Conv2D(64, 4, activation = 'relu'))
        model.add(Flatten())                                                                       
        model.add(Dense(256, activation = 'relu'))                                                                            
        model.add(Dense(actions_size, activation = 'linear')) # Выходной слой должен иметь активационную функцию 'linear' — мы предсказываем награды на выходе НС.

        # Практика показывает, что RMSprop — хороший оптимизатор для обучения с подкреплением, однако можно использовать adam.
        optimizer = RMSprop(learning_rate = learning_rate) 
        
        # Компилируем модель с функцией ошибки mse и заданным оптимизатором.
        model.compile(loss = 'mse', optimizer = optimizer) 
        return model



def preprocess_frame(q_learning:QLearning, frame):
    ''' Функция преобразования изображений
        
        Args:
            frame - 

        Returns:
            Возвращаем предобработанное, нормализованное, решейпнутое изображение
    
    ''' 
    # Меняем оси:
    frame = np.rollaxis(frame, 0, 3) 
    
    # Меняем размерность картинки на (64×64):
    frame = cv2.resize(frame, (q_learning.image_width, q_learning.image_height), interpolation=cv2.INTER_CUBIC) 
    
    # Переводим в черно-белое:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    return frame.reshape(q_learning.image_width, q_learning.image_height, 1)/255 
  



def get_predict_action(q_learning:QLearning, state):
    
  ''' Функция предсказания награды за действие

      Args: state -
            epsilon - 
            action_size -

      Returns: выбранное действие и новое значение epsilon
  
  '''

  # Генерируем рандомное значение и сравниваем
  if random.random() <= q_learning.epsilon:                     
    action_index = np.random.randint(0, q_learning.runtime.game.get_action_size())  
  
  # Иначе (если рандомное число больше, чем эпсилон)
  else:                                              
    # Предсказываем все Q-значения при следующим действии (Q(s, a) для каждого действия a)
    Q_values = q_learning.main_model.predict(np.expand_dims(state, axis = 0)) 
    # Извлекаем индекс действия который приводит к максимальному значению Q(s, a)
    action_index = np.argmax(Q_values)                             

  # Снижаем значение эпсилон, если оно больше, чем final_epsilon, снижаем значение epsilon на epsilon_decay_factor.
  if q_learning.epsilon > q_learning.final_epsilon:                        
    q_learning.epsilon -= q_learning.epsilon_decay_factor

  return action_index

def add_to_memory(previous_state, 
                  action, 
                  reward, 
                  current_state, 
                  episode_done):
  
  ''' Функция записи информации в память
      
      Args:
        previous_state — массивы из состояния среды
        action — действие, которое было в нем принято
        reward — награда, которая была получена 
        current_state — следующее состояние, к которому действие привело 
        episode_done — булевое значение флагов окончания игры (кадр последний в эпизоде)

      Returns:
  '''
  
  # memory — глобальная переменная. Мы записываем в нее всю нужную информацию:
  q_learning.memory.append((previous_state, action, reward, current_state, episode_done))




observation_steps = 10000            # Количество шагов 
                                      #количество шагов, за которые алгоритм не обучается, 
                                      # но добавляет в память новую информацию (состояния, действия, награды).
                                      # Это необходимо, чтобы алгоритм не переобучался на маленьком объеме памяти.
target_model_update_frequency = 5000 # Частота обновления целевой  
#определяет количество шагов, после которого мы обновляем целевую модель (копируем все веса основной модели в целевую модель).
# В оригинальной статье по Q-learning использовалось 10 000 шагов, но в этом примере мы будем использовать 5 000 шагов.

# Другие гиперпараметры

timesteps_per_train = 100       # Обучаем модель раз в 100 шагов (не обязательно ждать до конца игры)
learning_rate = 1e-4            # Обычно в обучении с подкреплением ставят низкий шаг обучения, например 1e-4


# Инициализируем среду:

doomGame = Vizdoom(DoomGame())
game_runtime = GameRuntime(doomGame)
q_learning = QLearning(game_runtime)


# Создаем основную модель (будет обучаться):
q_learning.set_main_model(Make_DQN, learning_rate)   

# Создаем целевую сеть (не будет обучаться, периодически будет обновляться под основную модель):
q_learning.set_target_model(Make_DQN, learning_rate) 

# Устанавливаем параметры целевой модели (копируем в нее значения основной модели):
q_learning.update_target_model()                                           

# Устанавливаем количество кадров за каждое действие. Нам не нужен каждый кадр, 
# поэтому будем совершать действие и брать новое состояние лишь раз в 4 кадра:
frames_per_action = 4  

load_pretrained = False #Решаем, если мы обучаем модель с нуля или продолжаем предыдущую сессию обучения

#Если хотим продолжить текущее обучение, загружаем сохраненные веса для основной и целевой моделей:
if load_pretrained:
  q_learning.load_weights()
  #Также загружаем ранее сохраненные статистики из pickle файла:
  with open(vizdoom_stats_file, 'rb') as f:
    record_rewards, record_kills, record_ammos, episode_number, timestep, q_learning.epsilon = pickle.load(f)

#Иначе мы просто инициализируем списки, в которых будет храниться статистика о работе агента:
else:
  record_rewards = []          #Сюда будем записывать награды за 10 эпизодов (для анализа статистики)
  record_kills = []            #Сюда будем записывать количество убитых врагов (для анализа статистики)
  record_ammos = []            #Сюда будем записывать количество оставшихся патронов (для анализа статистики)

  episode_number = 1     #Инициализируем номер эпизода как 1
  timestep = 0           #Инициализируем номер шага как 0


#ИГРОВОЙ ЦИКЛ
  # Генерируем новый эпизод:
game_runtime.new_episode()
game_runtime.load_current_state()

# Извлекаем информацию об игре (количество убитых врагов, патроны, здоровье):
current_info = game_runtime.current_info

# Записываем информацию о текущем моменте как 'предыдущий' момент (чтобы потом мы могли сравнить разницу):
previous_info = current_info             

# Извлекаем кадр из игры (480×640):
frame = game_runtime.frame                                                       
# Предобрабатываем кадр в черно-белый (размер 64×64):
processed_frame = preprocess_frame(q_learning, frame)                                             
current_state = np.stack([processed_frame.reshape(64, 64)] * q_learning.num_frames, axis = 2)    

# В качестве первого состояния просто дублируем кадр 4 раза:
current_state = np.stack([processed_frame.reshape(64, 64)] * q_learning.num_frames, axis = 2)    
# Инициализируем предыдущий шаг как текущий шаг:
previous_state = current_state  

# Инициализируем награды:

interval_reward = 0  # за интервал (10 эпизодов) как 0
interval_kills = 0   # за количество убитых врагов (10 эпизодов) как 0
interval_ammos = 0   # за количество оставшихся патронов (10 эпизодов) как я

#ОБУЧЕНИЕ

while episode_number<1500:
  
  # Увеличиваем номер шага на 1:
  timestep += 1 
  # Извлекаем индекс награды и новое значение эпсилон:
  action_index = get_predict_action(q_learning, previous_state)  
  # Приводим награду в onehot массив:
  action_onehot = to_categorical(action_index)                              
  # Подаем действие в игровую среду в качестве списка: 
  game_runtime.game.set_action(action_onehot.tolist())                                   
  # Игра продвигается на 4 кадра (значение frames_per_action):
  game_runtime.game.advance_action(frames_per_action)                                    

  # Проверяем, если эпизод закончился:
  episode_done = game_runtime.game.is_episode_finished() 

  # Нам необходимо возобновить среду и записать нужные статистики когда заканчивается эпизод:
  if episode_done: 
    print(f"Закончился {episode_number}-й эпизод. Значение эпсилон: {round(q_learning.epsilon, 2)}, Количество убитых врагов: {current_info[0]}, количество оставшихся патронов: {current_info[1]}")
    
    episode_number += 1   # Увеличиваем номер эпизода на 1:
    interval_kills += current_info[0]
    interval_ammos += current_info[1]

    # Чтобы не собирать слишком много данных и чтобы их было удобно отображать на графике

    # Записываем результат раз в 10 эпизодов:
    if episode_number % 10 == 0 and episode_number > 0: 
      
      # Добавляем награду в список всех наград:
      record_rewards.append(interval_reward)            
      # Добавляем количество убитых врагов:
      record_kills.append(interval_kills)               
      # Добавляем количество неиспользованных патронов:
      record_ammos.append(interval_ammos)               
      # Записываем результаты в графики:
      show_scores(record_rewards, record_kills, record_ammos)  

      # Сохраняем веса модели:
      q_learning.save_main_model_weights()
      
      # Записываем статистику в файл через библиотеку pickle:
      with open(vizdoom_stats_file, 'wb') as f:  
        pickle.dump([record_rewards, record_kills, record_ammos, episode_number, timestep, q_learning.epsilon], f) 
      print("Статистика успешно сохранена.")

      # Заново инициализируем значения статистики для интервала в 10 эпизодов:
      interval_reward, interval_kills, interval_ammos = 0, 0, 0 

    # Начинаем новый эпизод игры:
    game_runtime.new_episode()                       
    # Извлекаем новое состояние игры:
 
    
  game_runtime.load_current_state()
  # Извлекаем информацию об игровой среде (количество убитых врагов, неиспользованных патронов, текущее здоровье):
  current_info = game_runtime.current_info   
  frame = game_runtime.frame
  # Предобрабатываем кадр (новая размерность будет 64×64×1):
  processed_frame = preprocess_frame(q_learning, frame)    
  # Обновляем состояние — удаляем последний кадр и добавляем новый:
  current_state = np.append(processed_frame, current_state[:, :, :q_learning.num_frames-1], axis = 2) 

  # Извлекаем награду за шаг из среды (логика, которую не можем менять):
  environment_reward = game_runtime.game.get_last_reward()                           
  # Извлекаем награду за шаг из самописной функции (самописная награда, значит, можем менять логику):
  custom_reward = get_reward(previous_info, current_info, episode_done) 
  # Общая награда — это сумма награды из среды и самописной награды:
  reward = environment_reward + custom_reward 

  # Добавляем награду в переменную для статистики:
  interval_reward += reward 

  # Добавляем предыдущее состояние, действие, награду и текущее состояние в память:
  add_to_memory(previous_state, action_index, reward, current_state, episode_done) 
  
  # Обучаем нашу модель раз в 100 шагов, но только если у нас достаточно данных в памяти:
  if timestep % timesteps_per_train == 0 and len(q_learning.memory) > observation_steps: 
    q_learning.train_network()

  # Обновляем целевую модель весами основной модели раз в заданное количество (5 000) шагов:
  if timestep % target_model_update_frequency == 0: 
    q_learning.update_target_model()

  # Запоминаем предыдущую информацию:
  previous_info = current_info    
  # Запоминаем предыдущее состояние:
  previous_state = current_state  


  # Кадры из игр будут записываться в этот массив:
video_frames = []  
# Устанавливаем эпсилон как 0 (после обучения):

# Снова инициализируем среду:
doomGame = Vizdoom(DoomGame())
game_runtime = GameRuntime(doomGame)
q_learning = QLearning(epsilon=0)

# Создаем основную модель (будет управлять агентом):
q_learning.set_main_model(Make_DQN, learning_rate)   
q_learning.load_main_model_weights()

# Генерируем новый эпизод:
game_runtime.new_episode()      
# Извлекаем первый кадр (это еще не полноценное состояние):                  
game_runtime.load_current_state()            

# Извлекаем кадр из игры (480×640):
frame = game_runtime.frame                                                       
# Предобрабатываем кадр в черно-белый (размер 64×64):
processed_frame = preprocess_frame(q_learning, frame)                                             
# В качестве первого состояния просто дублируем кадр 4 раза:
current_state = np.stack([processed_frame.reshape(64, 64)] * q_learning.num_frames, axis = 2)    
# Записываем текущее состояние в предыдущее состояние:
previous_state = current_state    

while True:
  # Извлекаем индекс награды и новое значение эпсилон:
  action_index = get_predict_action(q_learning, current_state)   
  # Приводим награду в onehot-массив:
  action_onehot = to_categorical(action_index)                              
  # Подаем действие в игровую среду в качестве списка:
  game_runtime.game.set_action(action_onehot.tolist())                                    
  # Игра продвигается на 4 кадра (значение frames_per_action):
  game_runtime.game.advance_action(frames_per_action)                                    

  # Предобрабатываем кадр в черно-белый (размер 64×64×1):
  game_runtime.load_current_state()                                    

  # Проверяем, если эпизод закончился:
  episode_done = q_learning.game.is_episode_finished() 

  # Нам необходимо возобновить среду и записать нужные статистики, когда заканчивается эпизод:
  if episode_done: 
    
    # Затем необходимо начать новый эпизод игры:
    game_runtime.new_episode()                      
                                       
    # Выходим из игрового цикла:
    break 
  game_runtime.load_current_state()  
  # Извлекаем новый кадр из игры:
  frame = game_runtime.frame           
  # Добавляем кадр в массив, меняем формат размерности (3, width, height) -> (width, height, 3):
  video_frames.append(np.rollaxis(frame, 0, 3)) 

  # Предобрабатываем кадр (новая размерность будет 64×64×1):
  processed_frame = preprocess_frame(q_learning, frame)    
  # Обновляем состояние — удаляем последний кадр и добавляем новый:
  current_state = np.append(processed_frame, current_state[:, :, :q_learning.num_frames-1], axis = 2) 

  # Запоминаем предыдущее состояние:
  previous_state = current_state  


  # Чем больше кадров в секунду, тем быстрее будет проигрываться видео
out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), 20, (640, 480)) 

# В цикле добавляем каждый кадр в видео (делаем предобработку кадра — меняем каналы с RGB в BGR, поскольку CV2 воспринимает каналы как BGR):
for i in range(len(video_frames)):   
  out.write(cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR))

# Закрываем объект для создания видео:
out.release()

# Указываем путь к видео:
path=video_file 

# Извлекаем видео из заданного пути (куда мы ранее записали видео через CV2):
clip=VideoFileClip(path) 

# Отображаем видео в Colab:
clip.ipython_display(width=640, maxduration = 40) 