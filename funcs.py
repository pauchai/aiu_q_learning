import matplotlib.pyplot as plt #Импортируем модуль pyplot из бибиотеки matplotlib
import numpy as np            #Библиотека numpy

def show_scores(scores, 
                killcount, 
                ammo):

  ''' Функция визуализации результата
      
      Args:
        scores -  
        killcount -  
        ammo -

      Returns: график
  '''     

  # Удаляем предыдущий вывод ячейки:
  #output.clear() 

  # Создаем два сабплота (в левом будут награды и средние награды, в правом будут количества убитых врагов и патронов): 
  fig, axes = plt.subplots(1, 2, figsize = (20, 8)) # Делаем размер графика большим

  # Устанавливаем большой размер полотна:
  axes[0].plot(scores, label = "Награда за эпизод")                          
  # Отрисовываем скользящие средние награды:
  axes[0].plot(moving_average(scores), label = "Скользящее среднее награды") 
  # Добавляем лейблы осей:
  axes[0].set_xlabel("Итерация", fontsize = 16)                              
  axes[0].set_ylabel("Награда", fontsize = 16)
  # Добавляем легенду к графику:
  axes[0].legend()                                                           

  # Отрисовываем количество убитых врагов:
  axes[1].plot(killcount, 'red', linestyle = '--', label = "Количество убитых врагов (сумма за 10 эпизодов)")              
  # Отрисовываем количество убитых врагов (скользящее среднее):
  axes[1].plot(moving_average(killcount), 'black', label = "Количество убитых врагов (скользящее среднее за 10 итераций)") 
  # Отрисовываем количество оставшихся патронов:
  axes[1].plot(ammo, 'green', linestyle = '--', label = "Осталось патронов (сумма за 10 эпизодов)")                        
  # Отрисовываем количество оставшихся патронов (скользящее среднее):
  axes[1].plot(moving_average(ammo), 'blue', label = "Осталось патронов (скользящее среднее за 10 итераций)")              
  # Добавляем лейблы осей:
  axes[1].set_xlabel("Итерация", fontsize = 16)  
  axes[1].set_ylabel("Значение", fontsize = 16)
  # Добавляем легенду к графику:
  axes[1].legend()                               

  # Отображаем график:
  plt.show() 

def moving_average(data, 
                   width = 10): 
  
  ''' Функция для подсчета скользящего среднего всех значений
      
      Args:
        data — входной массив, 
        width — длина, на которую считаем скользящее среднее
      
      Returns: результат свертки данных на фильтр из единиц — наше скользящее среднее 
  '''

  # Длина свертки:  
  width = min(width, len(data))    

  # Создадим паддинг для свертки:  
  data = np.concatenate([np.repeat(data[0], width), data])         
  
  # Возвращаем результат свертки: 
  return (np.convolve(data, np.ones(width), 'valid') / width)[1:]  


def get_reward(previous_info, 
               current_info, 
               episode_done):

    ''' Функция предобработки наград

        Args:
            previous_info — информация об игровой среде на предыдущем кадре (количество убитых врагов, патроны и здоровье)
            current_info — информация об игровой среде на текущем кадре (количество убитых врагов, патроны и здоровье)
            episode_done — булевое значение, которое говорит, если кадр последний в эпизоде.
            misc[0] — количество убитых врагов, misc[1] — патроны, misc[2] — здоровье
        
        Returns: подсчитанная награда
               
    '''   
    
    # Инициализируем награду как 0
    reward = 0     
    
    # Если кадр последний в игре, ставим награду как -0.1 и возвращаем ее (агент умер)
    if episode_done:                          
        reward = -0.1
        
        return reward
    
    # Если убили врага в кадре, увеличиваем награду на 1
    if current_info[0] > previous_info[0]:   
        reward += 1
    
    # Если потеряли здоровье, уменьшаем награду на 0.1
    if current_info[1] < previous_info[1]:   
        reward -= 0.1
    
    # Если использовали патрон, уменьшаем награду на 0.1
    if current_info[2] < previous_info[2]:   
        reward -= 0.1

    return reward 

