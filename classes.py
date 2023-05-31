from vizdoom import DoomGame, ScreenResolution
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential #Импортируем функции для создания и загрузки модели из тензорфлоу
from tensorflow.keras.layers import *                             #Импортируем все слои из кераса
from tensorflow.keras.optimizers import RMSprop                   #Импортируем оптимизатор RMSprop
from tensorflow.keras import backend as K                         #Импортируем модуль для бэкэнда кераса
from tensorflow.keras.utils import to_categorical 
import numpy as np            #Библиотека numpy
import random
from collections import deque   #deque - это список где автоматический удаляются старые значения при добовлении новых, чтобы не было переполнение памяти.

from abc import ABC, abstractmethod

class GameAbstract(ABC):
    @abstractmethod
    def get_action_size(self):
        pass
    @abstractmethod
    def new_episode(self):
        pass
    @abstractmethod
    def get_state(self):
        pass
    @abstractmethod
    def set_action(self, value):
        pass

    @abstractmethod
    def is_episode_finished(self):
        pass

    @abstractmethod
    def advance_action(self, num_tics):
        pass

    @abstractmethod
    def get_last_reward(self):
        pass

class Vizdoom(GameAbstract):
    action_size:int = None
    def __init__(self, engine:DoomGame,
                 scenario_file:str = "./Vizdoom/scenarios/defend_the_center.cfg"
                 ):
        self.engine = engine
        self.engine.load_config(scenario_file)
        self.engine.set_screen_resolution(ScreenResolution.RES_640X480)
        self.engine.set_window_visible(False)
        self.engine.init()
  
    def get_action_size(self):
        return self.engine.get_available_buttons_size()

    def new_episode(self):
        self.engine.new_episode()

    def get_state(self):
        return self.engine.get_state()
    
    def set_action(self, value):
        return self.engine.set_action(value)
    
    def advance_action(self, num_tics):
        self.engine.advance_action(num_tics)

    def is_episode_finished(self):
        return self.engine.is_episode_finished()
    
    def get_last_reward(self):
        return self.engine.get_last_reward()

class GameRuntime:
    game:GameAbstract
    current_info:any
    frame:any


    def __init__(self, game):
        self.game = game
    
    def new_episode(self):
        self.game.new_episode()

    def load_current_state(self):
        game_data = self.game.get_state()
        self.current_info = game_data.game_variables
        self.frame = game_data.screen_buffer


class QLearning:
    load_pretrained = False
    #game:GameAbstract
    runtime:GameRuntime
    gamma:float  # Гамма   
                        # параметр для передачи наград между состояниями. 
                        # Значение от 0 до 1: 
                        # чем больше значения этого параметра, тем больше нейронная сеть будет приоритезировать награды, которая она может получить в будущем.
    
    batch_size:int      # Используем размер пакета в 32 
    timesteps_per_train:int  # Обучаем модель раз в 100 шагов (не обязательно ждать до конца игры)
    #Предобработка изображений
    image_width:int   # Ширина картинки (кадра)
    image_height:int  # Высота картинки (кадра)

    num_frames:int     # Количество последовательных кадров в одном состоянии (используется позже)

    state_shape:tuple # Размерность каждого состояния — размер картинки:
    main_model:Model
    target_model:Model

    # В памяти будет храниться не более 40 000 пар текущих и следующих состояний, действия которых нейронная сеть выбрала, а также их соответствующие награды
    maximum_memory_length:int          
    memory = None
    file_weights:str
    epsilon:float
    epsilon_decay_steps:int
    final_epsilon:int
    epsilon_decay_factor:float

    def __init__(self, 
                # game: GameAbstract, 
                 
                runtime:GameRuntime,
                 gamma:float = 0.95,
                 batch_size:int = 32,
                 timesteps_per_train = 100,
                 image_width:int = 64,
                 image_height:int = 64,
                 num_frames:int = 4,
                 maximum_memory_length:int = 40000,
                 file_weights:str = "model.h5",
                 epsilon:float = 1 ,
                 epsilon_decay_steps:int = 200000,
                 final_epsilon:int = 0.01
                ):
        self.runtime = runtime
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.timesteps_per_train = timesteps_per_train
        self.image_width = image_width
        self.image_height = image_height
        self.num_frames = num_frames
        self.state_shape =  (image_width, image_height, num_frames) 

        self.maximum_memory_length = maximum_memory_length
        self.memory =  deque([], maxlen = maximum_memory_length) 
        self.file_weights = file_weights

        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_factor = (self.epsilon - self.final_epsilon) / self.epsilon_decay_steps 

    def _set_model_from_creator(self, model_creator, learning_rate):
        return  model_creator(self.state_shape,  self.runtime.game.get_action_size(), learning_rate)
    
    def set_main_model(self, model_creator, learning_rate):
        self.main_model = self._set_model_from_creator(model_creator, learning_rate)
    
    def set_target_model(self, model_creator, learning_rate):
        self.target_model = self._set_model_from_creator(model_creator, learning_rate)
    
  

    def train_network(self): 

        ''' Функция обучения алгоритма
            
            Args: 

            Returns: обученная модель
        '''        

        # Извлекаем пакет данных из памяти:
        previous_states, actions, rewards, current_states, game_finished = self.sample_from_memory()

        # Предсказываем Q(s, a):   
        Q_values = self.main_model.predict(previous_states)                                            
        
        # Предсказываем Q(s', a'):  
        next_Q_values = self.target_model.predict(current_states)                                      

        # Модифицируем значения Q: 
        for i in range(len(current_states)): 

            # Если состояние последнее в эпизоде:  
            if game_finished[i]:               
                Q_values[i, actions[i]] = rewards[i]
            # Если состояние не последнее в эпизоде: 
            else: 
                Q_values[i, actions[i]] = rewards[i] + self.gamma * next_Q_values[i, actions[i]] 

        # Обучаем модель: 
        self.main_model.fit(previous_states, Q_values, batch_size = self.batch_size, verbose = 0) 


    def sample_from_memory(self):

        ''' Функция сэмплирования данных
            
            Args: 

            Returns: распакованные данные
        '''     
        
        # Определим размер памяти: 
        memory_batch_size = min(self.batch_size * self.timesteps_per_train, len(self.memory))      
        
        # Сэмплируем данные:
        mini_batch = random.sample(self.memory, memory_batch_size) 

        # Создаем массив из нулей с размерностью предыдущих состояний, массива действий, массива наград, текущих состояний, флагов окончания игры
        previous_states = np.zeros((memory_batch_size, self.image_width, self.image_height, self.num_frames))   
        actions = np.zeros(memory_batch_size)                                                     
        rewards = np.zeros(memory_batch_size)                                                    
        current_states = np.zeros((memory_batch_size, self.image_width, self.image_height, self.num_frames))    
        episode_done = np.zeros(memory_batch_size)                                               

        # Перебираем данные и копируем их значения в массивы нулей:
        for i in range(memory_batch_size):                  
            previous_states[i, :, :, :] = mini_batch[i][0]  
            actions[i] = mini_batch[i][1]                   
            rewards[i] = mini_batch[i][2]                   
            current_states[i, :, :, :] = mini_batch[i][3]   
            episode_done[i] = mini_batch[i][4]             

        return previous_states, actions.astype(np.uint8), rewards, current_states, episode_done

    def update_target_model(self): 

        ''' Функция обновления весов в целевой модели, т. е. той, что
            устанавливает веса целевой модели (которая не обучается) такими
            же, как веса основной модели (которая обучается)
        
        '''

        self.target_model.set_weights(self.main_model.get_weights())
        self.main_model.save_weights(self.file_weights)
    
    def _load_model_weights(self, model):
        model.load_weights(self.file_weights)
    
    def load_main_model_weights(self):
        self._load_model_weights(self.main_model)
    
    def load_target_model_weights(self):
        self._load_model_weights(self.target_model)
                
    def load_weights(self):
        self.load_main_model_weights()
        self.load_target_model_weights()

    def _save_model_weights(self, model:Model):
        model.save_weights(self.file_weights)

    def save_main_model_weights(self):
        self._save_model_weights(self.main_model)


class Reporter:
    q_learning:QLearning
    
    def __init__(self, q_learning):
        self.q_learning = q_learning

