from scipy.signal import butter, filtfilt,morlet
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
#from scipy.stats.stats import pearsonr
import math
import pywt


def First_line_change(file):
    #Строчка,которая позволит нам производить анализ,по умолчанию не корректно,т.ч. надо заменить
    nfsline='COUNTER,INTERPOLATED,AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4,RAW_CQ,GYROX,GYROY,MARKER,MARKER_HARDWARE,SYNC,TIME_STAMP_s,TIME_STAMP_ms,TimeStamp,CQ_AF3,CQ_F7,CQ_F3,CQ_FC5,CQ_T7,CQ_P7,CQ_O1,CQ_O2,CQ_P8,CQ_T8,CQ_FC6,CQ_F4,CQ_F8,CQ_AF4,CQ_CMS,CQ_DRL'
    #Открываем файл на чтение
    with open(file, 'r') as f1:
        #Считываем все строчки
        lines = f1.readlines()
        #Открываем этот же файл на запись(при записи файл удаляет все записи,если там что-то было)
        with open(file, 'w') as f2:
            for num,line in enumerate(lines):
                #Не помню зачем это нужно,но,вроде,нужно,как узнаю-поменяю
                line = line.strip()
                #Если первая запись,то вставляем нашу строчку,иначе строчки,которые были в файле при чтении
                if num == 0:
                    f2.write(nfsline+'\n')
                else:
                    f2.write(line+'\n')

def Butterworth_Filter(signal):
    #Частота дискретизации
    frequency=128.0
    #Обрезаем от lowcut до highcut
    lowcut=0.16
    highcut=0.6
    #Частота Найквиста
    nyquist_f = 0.5 * frequency
    #Полоса пропускания(частоты среза выраженные,как доля частоты Найквиста)
    low = lowcut / nyquist_f
    high = highcut / nyquist_f
    #Тип полосы bandpass
    b, a = butter(3, [low, high],'bandpass')
    #Необходимо вернуть отфильрованный массив y
    after_butter_y = filtfilt(b, a, signal)
    return after_butter_y

def Fourier(array):
    #Новый список для модулей коэффициентов Фурье
    new_four_list=[]
    #встроенная функиция Фурье из numpy, на вход (массив,его размерность и какая-то хрень)
    four_list=np.fft.fft(array,len(array),axis=-1)
    for element in four_list:
        #Комплексное число из списка
        element_from_list=element
        #Преобразуем в строку для проверки на '-'
        string_element=str(element)
        #Проверяем,если есть -,то умножаем на -1
        if string_element[1]=='-':
            abs_element=round(abs(element_from_list)*-1,2)
        else:
            abs_element=round(abs(element_from_list),2)
        #Заполняем список из модулей коэфициентов Фурье
        new_four_list.append(abs_element)
    return new_four_list

def Wavelet_db(array):
    cA, cD = pywt.dwt(array, 'db1')
    coef=pywt.idwt(cA, cD, 'db1')
    return coef

def Wavelet_morl(array,width):
    #коэффициенты аппроксимации и детализации
    cA, cD = pywt.cwt(array,width,'morl')
    #Обратное дискретное вейвлет-преобразование
    coef=pywt.icwt(cA, cD, width,'morl')
    return coef

def Cepstral(signal):
    powerspectrum = np.abs(np.fft.fft(signal))**2
    cepstrum = np.fft.ifft(np.log(powerspectrum))
    return cepstrum

def Signal_analysis(file_name,electrode_mark,plot_output=None):
    global filt_y
    #Считываем данные из csv(разеделитель в документе - запятая),данные в виде таблицы со значениями всех каналов
    data_frame = pd.read_csv(file_name,sep=',')
    #Преобразовываем 1 нужный нам канал из таблицы в словарь(например,из всей таблицы только F7)
    dictionary_signal=data_frame[electrode_mark].to_dict()
    #Пролучаем x,y(x-индекс из словаря,y-значение по индексу из словаря),преобразовывая тип данных к list(списку)
    x=list(dictionary_signal.keys())
    y=list(dictionary_signal.values())
    #Нужны только данные до 2560,тк у нас 20 секунд,частота 128 -> 128*20=2560 записей
    x=x[:2560]
    y=y[:2560]
    #Отфильтруем сигнал для других функций
    filt_y=Butterworth_Filter(y)
    #Создадим csv с отфильтрованными y(для того,чтоб можно было проверить значения) \\Для отладки-не алгоритм
    os.chdir(r"C:\Users\Техобслуживание\Desktop\pythonCode")
    dff=pd.DataFrame({'Y-и':filt_y})
    dff.to_csv('Y-и после фильтрации.csv',sep=',',index=False)
    os.chdir(r"C:\Users\Техобслуживание\Desktop\pythonCode\igor")
    #Если хотим графики,то ставим метку "Да" при вызове функции,по умолчанию график не создается,переменную обязательно писать не нужу
    if plot_output=='Да':
        #subplot-функция для совмещения графиков в одном окне (кол-во строк,кол-во столбцов,номер ячейки в окне)
        #Строим график на x,y
        #grid-сеточка на координатной прямой
        plt.subplot (1, 2, 1)
        plt.plot (x,y)
        plt.grid(True)
        plt.title ("Сигнал")
        #Строим график на x,after_butter_y - отфильтрованный сигнал
        plt.subplot (1, 2, 2)
        filt_y=Butterworth_Filter(y)
        plt.plot (x,filt_y)
        plt.grid(True)
        plt.title("Отфильтрованный сигнал")
        #Выводим графики
        plt.show()
    #Применяем преобразование Фурье,получаем коэфициенты
    Fourier_coefficients=Fourier(filt_y)
    #Применяем преобразование Вейвлет,получаем коэфициенты(Добиши)
    Wavelet_coefficients=Wavelet_db(filt_y)
    #Применяем преобразование Вейвлет,получаем коэфициенты(Морле)
    Wavelet_coefficients=Wavelet_morl(filt_y,x)
    #Применяем кепстальное преобразование,получаем коэфициенты
    Cepstral_coefficients=Cepstral(filt_y)
    print(Cepstral_coefficients)
    return Fourier_coefficients

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    return diffprod / math.sqrt(xdiff2 * ydiff2)

def sorted_list(inputStr):
    return int(inputStr[ :inputStr.find("_")])

def check_correlation():
    #Переходим в директорию
    os.chdir(r"C:\Users\Техобслуживание\Desktop\pythonCode\igor")
    files=os.listdir(r"C:\Users\Техобслуживание\Desktop\pythonCode\igor")
    #Отсортированный список
    newfiles=sorted(files,key=sorted_list)
    #Цикл для перехода по каждому названию файла из всех в директории
    #Словарь для csv Фурье
    dict_for_Four={}
    #Словарь для csv корреляция Пиросна
    dict_for_Pears={}
    list_for_average=[]
    #Индексация файлов из папки
    for i in range(0,50):
        sum_correlation=0
        print(i)
        #Тот файл который анализируем
        number_1=Signal_analysis(newfiles[i],'AF3')
        #Добавляем словарь с данными по коэффициентам Фурье в словарь(для каждого файла)
        dict_for_Four.update({newfiles[i]:number_1})
        #Список для коэффициентов корреляции
        list_for_Pears=[]
        #Цикл для прохода по файлам(для корреляционного анализа)
        for file in newfiles:
            #Если сейчас файл,который мы анализируем,то пропустить
            if file==newfiles[i]:
                continue
            #Файл с которым анализируем
            next=Signal_analysis(file,'AF3')
            #Добавляем коэффициенты корреляции в list
            sum_correlation+=abs(pearson_def(number_1,next))
            list_for_Pears.append(pearson_def(number_1,next))
            #Записываем коэффициент корреляции в словарь(название файла,)
            dict_for_Pears.update({newfiles[i]:list_for_Pears})
        average_correlation=sum_correlation/49
        list_for_average.append(average_correlation)
    sum_average_correlation=0
    for element in list_for_average:
        sum_average_correlation+=element
    all_average_correlation=sum_average_correlation/50
    print(all_average_correlation)
    os.chdir(r"C:\Users\Техобслуживание\Desktop\pythonCode")
    #Создаем csv корреляция Пиросна
    all_dict_for_Pears=pd.DataFrame(data=dict_for_Pears)
    all_dict_for_Pears.to_csv('Коэффициенты корреляции.csv',sep=',',index=False)
    #Создаем csv Фурье
    all_dict_for_Four=pd.DataFrame(data=dict_for_Four)
    all_dict_for_Four.to_csv('Коэффициенты Фурье.csv',sep=',',index=False)
    os.chdir(r"C:\Users\Техобслуживание\Desktop\pythonCode\igor")



check_correlation()


#cutoff=40.0
#normal_cutoff = cutoff / nyq
#b,a=butter(2,normal_cutoff,'lowpass')
#ny=filtfilt(b, a, oldy)

#b, a = butter(3, [low, high],'bandpass')
#nny = filtfilt(b, a, ny)
#nny=nny[500:1500]
#plt.plot(x,nnyвкп)
#plt.grid(True)
#plt.show()
'''def Pearson_correlation(coef_F_1,coef_F_2):
    print(pearsonr(coef_F_1,coef_F_2))'''
