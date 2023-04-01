import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import operator
from tkinter import filedialog as fd
import tkinter.messagebox as mb
from skimage import io, transform,metrics

# метод градиента
def Gradient (file):
    # Размер окна и величина смещения
    ksize = 3
    dx = 1
    dy = 1

    # Вычисление градиента
    gradient_x = cv2.Sobel(file, cv2.CV_32F, dx, 0, ksize=ksize)
    gradient_y = cv2.Sobel(file, cv2.CV_32F, 0, dy, ksize=ksize)

    # Вычисление абсолютного значения градиента
    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)

    # Вычисление итогового градиента
    gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

    SumGradient = []
    for i in range(0, len(gradient), 1):
        SumGradient.append(round(sum(gradient[i]) / len(gradient[i]), 1))
    return SumGradient

# метод двумерного дискретного косинусного преобразования
def DCT(file):
    # Применение двумерного дискретного косинусного преобразования (DCT)
    dct = cv2.dct(np.float32(file))
    return dct

# метод дискретного преобразования Фурье
def DFT(file):

    # Применение двумерного дискретного преобразования Фурье (DFT)
    dft = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Сдвиг нулевых частот в центр
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

# метод гистограмм
def Histogram(file):
    histg = cv2.calcHist([file], [0], None, [256], [0, 256])
    return histg

# метод scale
def Scale(file):
    img = io.imread(file)

    img_res = transform.resize(img, (20, 20))
    return img_res

#функция отрисовки графиков
def plot_grafs(num_e):
    stat_dct = []
    stat_dft = []
    stat_scale = []
    stat_hist = []
    stat_grad = []
    delta_k_h = 200
    delta_k_g = 80
    t_img_a = []
    t_hist = []
    t_grad = []
    t_dft = []
    t_dct = []
    t_scale = []
    e_img_a = []
    e_hist = []
    e_grad = []
    e_dft = []
    e_dct = []
    e_scale = []
    for i in range(1,11,1):
        sum_h = 0
        sum_g = 0
        sum_sim_dft = 0
        sum_sim_dct=0
        sum_sim_scale=0
        for j in range(1,num_e+1,1):
            res_h = 0
            res_g = 0
            fln_e = f"ORL/s{i}/{j}.pgm"
            e_img = cv2.imread(fln_e, cv2.IMREAD_GRAYSCALE)
            e_img_a.append(e_img)
            e_hist.append(Histogram(e_img))
            e_grad.append(Gradient(e_img))
            e_dft.append(DFT(e_img))
            e_dct.append(DCT(e_img))
            e_scale.append(Scale(fln_e))

            for k in range(num_e+1, 11, 1):
                fln_t=f"ORL/s{i}/{k}.pgm"
                t_img = cv2.imread(fln_t, cv2.IMREAD_GRAYSCALE)
                t_img_a.append(t_img)
                t_hist.append(Histogram(t_img))
                t_grad.append(Gradient(t_img))
                t_dft.append(DFT(t_img))
                t_dct.append(DCT(t_img))
                t_scale.append(Scale(fln_t))
                in_e_h, e_m_h = max(enumerate(e_hist[j-1+num_e*(i-1)]), key=operator.itemgetter(1))
                in_e_g, e_m_g = max(enumerate(e_grad[j-1+num_e*(i-1)]), key=operator.itemgetter(1))
                t_max_h = t_hist[k - num_e-1+(10-num_e)*(i-1)][in_e_h]
                t_max_g = t_grad[k - num_e-1+(10-num_e)*(i-1)][in_e_g]
                delt_h = abs(e_m_h - t_max_h)
                delt_g = abs(e_m_g - t_max_g)
                if (delt_h < delta_k_h):
                    res_h +=1
                if (delt_g < delta_k_g):
                    res_g += 1
                mean_mag_e = np.mean(e_dft[j-1+num_e*(i-1)])
                mean_mag_t = np.mean(t_dft[k - num_e-1+(10-num_e)*(i-1)])
                similarity_percent_dft = mean_mag_t / mean_mag_e
                if(similarity_percent_dft>1):
                    similarity_percent_dft = 2 - similarity_percent_dft
                sum_sim_dft +=similarity_percent_dft
                linalg_norm_e = np.linalg.norm(e_dct[j-1+num_e*(i-1)])
                linalg_norm_t = np.linalg.norm(t_dct[k - num_e-1+(10-num_e)*(i-1)])
                similarity_percent_dct = linalg_norm_t/linalg_norm_e
                if (similarity_percent_dct > 1):
                    similarity_percent_dct = 2 - similarity_percent_dct
                sum_sim_dct += similarity_percent_dct
                ssim = metrics.structural_similarity(e_scale[j-1+num_e*(i-1)], t_scale[k - num_e-1+(10-num_e)*(i-1)],data_range= 255)
                if (ssim > 1):
                    ssim = 2 - ssim
                sum_sim_scale +=ssim

            sum_h+=res_h
            sum_g += res_g
        stat_hist.append(sum_h/((10-num_e)*num_e))
        stat_grad.append(sum_g / ((10 - num_e) * num_e))
        stat_dft.append(sum_sim_dft/((10 - num_e) * num_e))
        stat_dct.append(sum_sim_dct/((10 - num_e) * num_e))
        stat_scale.append(sum_sim_scale / ((10 - num_e) * num_e))

    plt.rcParams['figure.figsize'] = 18, 10
    fig1, ((ax_1, ax_2, ax_3, ax_4, ax_5, ax_6),(ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(2, 6)
    fig2, (axH, axG,axDFT,axDCT, axScale) = plt.subplots(1, 5)
    plt.ion()
    ax_1.set_title('Тестовая')
    i_a = ax_1.imshow(t_img_a[0])
    ax_2.set_title('Гистограмма')
    h_a, = ax_2.plot(t_hist[0], color="g")
    ax_3.set_title('DFT')
    df_a = ax_3.imshow(t_dft[0], cmap='gray', vmin=0, vmax=255)
    ax_4.set_title('DCT')
    dc_a = ax_4.imshow(np.abs(t_dct[0]), vmin=0, vmax=255)
    x = np.arange(len(t_grad[0]))
    ax_5.set_title('Градиент')
    g_a, = ax_5.plot(x, t_grad[0], color="g")
    ax_6.set_title('Scale')
    sc_a = ax_6.imshow(t_scale[0])

    ax1.set_title('Эталон')
    i_a_e = ax1.imshow(e_img_a[0])
    ax2.set_title('Гистограмма')
    h_a_e, = ax2.plot(e_hist[0], color="g")
    ax3.set_title('DFT')
    df_a_e = ax3.imshow(e_dft[0], cmap='gray', vmin=0, vmax=255)
    ax4.set_title('DCT')
    dc_a_e = ax4.imshow(np.abs(e_dct[0]), vmin=0, vmax=255)
    x_e = np.arange(len(e_grad[0]))
    ax5.set_title('Градиент')
    g_a_e, = ax5.plot(x_e, e_grad[0], color="g")
    ax6.set_title('Scale')
    sc_a_e = ax6.imshow(e_scale[0])
    x_r_g = np.arange(len(stat_grad))
    x_r_h = np.arange(len(stat_hist))
    x_r_dft = np.arange(len(stat_dft))
    x_r_dct = np.arange(len(stat_dct))
    x_r_scale = np.arange(len(stat_scale))
    axH.plot(x_r_h, stat_hist, color="g")
    axH.set_title('Гистограмма')
    axG.plot(x_r_g, stat_grad, color="g")
    axG.set_title('Градиент')
    axDFT.plot(x_r_dft, stat_dft, color="g")
    axDFT.set_title('DFT')
    axDCT.plot(x_r_dct, stat_dct, color="g")
    axDCT.set_title('DVT')
    axScale.plot(x_r_scale, stat_scale, color="g")
    axScale.set_title('Scale')
    fig1.show()
    fig2.show()

    for t in range(0, 10, 1):
        for p in range(0+num_e*t, num_e*t+num_e, 1):
            i_a_e.set_data(e_img_a[p])
            h_a_e.set_ydata(e_hist[p])
            df_a_e.set_data(e_dft[p])
            dc_a_e.set_data(e_dct[p])
            g_a_e.set_ydata(e_grad[p])
            sc_a_e.set_data(e_scale[p])
            for m in range((0+p*(10-num_e)), (10-num_e)*(p+1), 1):
                i_a.set_data(t_img_a[m])
                h_a.set_ydata(t_hist[m])
                df_a.set_data(t_dft[m])
                dc_a.set_data(t_dct[m])
                g_a.set_ydata(t_grad[m])
                sc_a.set_data(t_scale[m])
                fig1.canvas.draw()
                fig1.canvas.flush_events()



def get_num_etalons():
    num_etalons = num_etalons_entry.get()
    if num_etalons.isdigit() and int(num_etalons) > 0:
        plot_grafs(int(num_etalons))
    else:
        tk.showerror("Ошибка", "Введите целое положительное число")

def choose_test():
    filename1 = fd.askopenfilename()
    filename2 = fd.askopenfilename()
    plot_grafs_choosen(filename1, filename2)

def show_res(text):
    msg = text
    mb.showinfo("Результат", msg)

#функция отрисовки графиков для результатов (точность)
def plot_grafs_choosen(filename1, filename2):
    delta_k_h = 100
    delta_k_g = 80
    res_h = 0
    res_g = 0
    sum_sim_dft = 0
    sum_sim_dct = 0
    e_img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    e_hist = Histogram(e_img)
    e_grad = Gradient(e_img)
    e_dft = DFT(e_img)
    e_dct = DCT(e_img)
    e_scale = Scale(filename1)
    t_img = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    t_hist = Histogram(t_img)
    t_grad = Gradient(t_img)
    t_dft = DFT(t_img)
    t_dct = DCT(t_img)
    t_scale = Scale(filename2)
    t_or_img = plt.imread(filename2, cv2.IMREAD_GRAYSCALE)
    e_or_img = plt.imread(filename1, cv2.IMREAD_GRAYSCALE)
    in_e_m, e_m_h = max(enumerate(e_hist), key=operator.itemgetter(1))
    in_e_g, e_m_g = max(enumerate(e_grad), key=operator.itemgetter(1))
    t_max_h = t_hist[in_e_m]
    t_max_g = t_grad[in_e_g]
    delt_c = abs(e_m_h - t_max_h)
    delt_g = abs(e_m_g - t_max_g)
    if (delt_c < delta_k_h):
        res_h += 1
    if (delt_g < delta_k_g):
        res_g += 1
    mean_mag_e = np.mean(e_dft)
    mean_mag_t = np.mean(t_dft)
    similarity_percent_dft = mean_mag_t / mean_mag_e
    if (similarity_percent_dft > 1):
        similarity_percent_dft = 2 - similarity_percent_dft
    sum_sim_dft += similarity_percent_dft
    linalg_norm_e = np.linalg.norm(e_dct)
    linalg_norm_t = np.linalg.norm(t_dct)
    similarity_percent_dct = linalg_norm_t / linalg_norm_e
    if (similarity_percent_dct > 1):
        similarity_percent_dct = 2 - similarity_percent_dct
    sum_sim_dct += similarity_percent_dct
    plt.rcParams['figure.figsize'] = 18, 10
    plt.subplot(3, 6, 13)
    plt.imshow(e_or_img)
    plt.title("Эталон")
    plt.subplot(3, 6, 14)
    plt.plot(e_hist, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 15)
    plt.imshow(e_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(e_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 17)
    x = np.arange(len(e_grad))
    plt.plot(x, e_grad, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 18)
    plt.imshow(e_scale)
    plt.title("Scale")

    plt.subplot(3, 6, 1)
    plt.imshow(t_or_img)
    plt.title("Тестовая")
    plt.subplot(3, 6, 2)
    plt.plot(t_hist, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 3)
    plt.imshow(t_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(t_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 5)
    x = np.arange(len(t_grad))
    plt.plot(x, t_grad, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 6)
    plt.imshow(t_scale)
    plt.title("Scale")
    if (res_g != 0 and res_h != 0 and similarity_percent_dft >=0.5 and similarity_percent_dct >=0.5):
        show_res("Совпадает")
    else:
        show_res("Не совпадает")
    plt.show()

# Создание главного окна
root = tk.Tk()
root.geometry("240x120")

# Создание метки и поля для ввода количества эталонов
num_etalons_label = tk.Label(root, text="Количество эталонов:")
num_etalons_label.pack()
num_etalons_entry = tk.Entry(root)
num_etalons_entry.pack()

# Кнопка для подтверждения ввода
plot_button = tk.Button(root, text="Построить графики", command=get_num_etalons)
plot_button.pack()

# Кнопка для произвольной выборки
plot_button = tk.Button(root, text="Произвести произвольную выборку", command=choose_test)
plot_button.pack()

# Запуск цикла обработки событий
root.mainloop()