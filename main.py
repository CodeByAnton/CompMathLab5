import math
from math import factorial

import numpy as np
import matplotlib.pyplot as plt


def read_from_console():
    print("Введите в одной строке значения x, в другой y")
    line_x = input()
    x = [float(x) for x in line_x.split(" ")]
    line_y = input()
    y = [float(y) for y in line_y.split(" ")]
    return x, y


def read_from_file():
    file_name = input("Введите имя файла (в первой строке файла должны быть x, во второй y) \n")
    try:
        with open(file_name) as f:
            lines = f.readlines()
            x = [float(val) for val in lines[0].split()]
            y = [float(val) for val in lines[1].split()]


    except FileNotFoundError:
        print("Файла с таким именем нет")
    return x, y


def generate_data_from_function():
    functions_map = {
        "sin(x)": lambda x: np.sin(x),
        "cos(x)": lambda x: np.cos(x)
    }

    print("Выберите функцию ")
    i = 1
    for function_name in functions_map.keys():
        print(f"{i}. {function_name}")
        i += 1
    func_choice = int(input("Ваш выбор: "))
    n = int(input("Введите количество точек: "))
    a = float(input("Введите начало интервала: "))
    b = float(input("Введите конец интервала: "))
    x = np.linspace(a, b, n)
    function = list(functions_map.keys())
    y = functions_map[function[func_choice - 1]](x)
    return x, y


def input_data():
    print("Выберите способ ввода данных:")
    print("1. Ввести данные вручную")
    print("2. Загрузить данные из файла")
    print("3. Сгенерировать данные на основе функции")
    choice = int(input("Ваш выбор: "))

    if choice == 1:
        x, y = read_from_console()


    elif choice == 2:
        x, y = read_from_file()

    elif choice == 3:
        x, y = generate_data_from_function()
    else:
        print("Некорректный выбор")
        return None, None

    return x, y


def create_shared_differences_table(x, y):
    ##Таблица разделенных разностей
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def create_finite_differences_table(x, y):
    ##Таблица конечных разностей
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1])

    return table


def create_central_differences_table(y):
    n = len(y)
    table = [y[:]]

    for k in range(1, n):
        last = table[-1][:]
        table.append(
            [last[i + 1] - last[i] for i in range(n - k)])

    return table




def print_finite_differences_table(x, y):
    table = create_finite_differences_table(x, y)
    print("Таблица конечных разностей:")

    n = len(table)
    for i in range(n):
        print(f'{i}\t{x[i]:.4f}', end='\t')

        for j in range(n - i):
            print(f'\t{table[i, j]:.4f}\t', end='')
        print()
    print()


def lagrange_interpolation(x, y, value):
    name = "Метод Лагранжа"
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (value - x[j]) / (x[i] - x[j])
        result += term
    return result


def newton_interpolation_with_shared_difference(x, y, value):
    name = "Метод Ньютона с разделенными разностями"
    n = len(x)
    result = y[0]
    temp = 1.0
    shared_differences_table = create_shared_differences_table(x, y)
    for i in range(1, n):
        temp *= (value - x[i - 1])
        result += temp * (shared_differences_table[0, i])
    return result



def generate_array_offset(n):
    dts = [0]
    for i in range(1, n + 1):
        dts.append(-i)
        dts.append(i)
    return dts


def check_finite_differences(x):
    step = x[1] - x[0]
    tolerance = 0.01  # допустимая погрешность

    for i in range(1, len(x) - 1):
        current_step = x[i + 1] - x[i]
        if not math.isclose(current_step, step, abs_tol=tolerance):
            return False
    return True


def first_interpolation_gauss_form(n, xs, ys, alpha_index, dts, h, fin_diffs, value):
    result = ys[alpha_index]
    t = (value - xs[alpha_index]) / h
    for k in range(1, n):
        term = 1
        for j in range(k):
            term *= t + dts[j]
        result += term * fin_diffs[k][len(fin_diffs[k]) // 2] / factorial(k)

    return result


def second_interpolation_gauss_form(n, xs, ys, alpha_index, dts, h, fin_diffs, value):
    result = ys[alpha_index]
    t = (value - xs[alpha_index]) / h
    for k in range(1, n):
        product = 1
        for j in range(k):
            product *= t - dts[j]
        result += product * fin_diffs[k][len(fin_diffs[k]) // 2 - (1 - len(fin_diffs[k]) % 2)] / factorial(k)
    return result


def gauss_interpolation(x, y, value):
    name = "Метод Гаусса"
    n = len(x)
    central_differences_table = create_central_differences_table(y)
    alpha_index = n // 2
    h = x[1] - x[0]
    dts = generate_array_offset(n // 2)
    f1 = first_interpolation_gauss_form(n, x, y, alpha_index, dts, h, central_differences_table, value)
    f2 = second_interpolation_gauss_form(n, x, y, alpha_index, dts, h, central_differences_table, value)
    if value > x[alpha_index]:
        return f1
    else:
        return f2


def stirling_interpolation(x, y, value):
    name = "Метод Стирлинга"
    n = len(x)
    central_differences_table = create_central_differences_table(y)
    alpha_index = n // 2
    h = x[1] - x[0]
    dts = generate_array_offset(n // 2)
    f1 = first_interpolation_gauss_form(n, x, y, alpha_index, dts, h, central_differences_table, value)
    f2 = second_interpolation_gauss_form(n, x, y, alpha_index, dts, h, central_differences_table, value)
    return (f1 + f2) / 2


def bessel_interpolation(x, y, value):
    n = len(x)
    central_differences_table = create_central_differences_table(y)
    alpha_index = n // 2
    h = x[1] - x[0]
    dts = generate_array_offset(n // 2)
    t = (value - x[alpha_index - 1]) / h
    # print(f"t={t}")
    result = (y[alpha_index - 1] + y[alpha_index]) / 2
    k = 0
    l = 0
    for i in range(1, n):
        if i % 2 == 1:
            current = (t - 0.5)
            for j in range(i - 1):
                current = current * (t + dts[j])

            current /= (factorial(i))
            current *= central_differences_table[i][alpha_index - 1 - k]
            result += current
            k += 1
        else:
            current = 1
            for j in range(i):
                current *= (t + dts[j])
            current /= (factorial(i))
            current *= (central_differences_table[i][alpha_index - l - 2] + central_differences_table[i][
                alpha_index - 1 - l]) / 2
            l += 1
            result += current

    return result


def generate_data_for_graphik(x, y, interpolation_func):
    xs, ys = [], []
    dx = 0.1
    a = x[0] - dx
    b = x[-1] + dx
    i = a
    while i <= b:
        xs.append(i)
        ys.append(interpolation_func(x, y, i))
        i += dx
    return xs, ys


def plot_interpolation(xs, ys, interpolation_func, interpolation_label, value, x, y):
    plt.plot(xs, ys)
    plt.title(interpolation_label)
    plt.scatter(value, interpolation_func(x, y, value), color='blue')
    plt.scatter(x, y, color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_all_interpolations(x, y, x_value, results, y_coord):
    plt.scatter(x, y, color='red', label='Исходные точки')

    labels = {
        'lagrange': "Метод Лагранжа",
        'newton': "Метод Ньютона с разделенными разностями",
        'gauss': "Метод Гаусса",
        'stirling': "Метод Стирлинга",
        'bessel': "Метод Бесселя"
    }

    for method, (xs, ys) in results.items():
        plt.plot(xs, ys, label=labels[method])

    plt.scatter(x_value, y_coord, color='blue', label='Значение функции для заданного аргумента')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='best', fontsize='small')
    plt.title("График все интерполяционных многочленов")
    plt.show()


def main():
    global gauss_result, stirling_result, bessel_result, newton_result
    gauss_result = None
    stirling_result = None
    bessel_result = None
    newton_result = None
    x, y = input_data()
    if x is None or y is None:
        return
    if len(x) != len(y):
        print("Количество X должно быть равно кличеству Y")
        return

    value = float(input("Введите значение аргумента для интерполяции:"))
    if not (x[0] < value < x[-1]):
        print("Аргуменет интерполяции должен быть между x_min и x_max")
        return
    print()

    print_finite_differences_table(x, y)
    print(f"Приближенное значение функции для x = {value}:\n")

    lagrange_result = lagrange_interpolation(x, y, value)
    results = {
        'lagrange': generate_data_for_graphik(x, y, lagrange_interpolation)}

    if not check_finite_differences(x):
        newton_result = newton_interpolation_with_shared_difference(x, y, value)
        results['newton'] = generate_data_for_graphik(x, y, newton_interpolation_with_shared_difference)

    if len(x) % 2 == 1 and check_finite_differences(x):
        gauss_result = gauss_interpolation(x, y, value)
        stirling_result = stirling_interpolation(x, y, value)
        results['gauss'] = generate_data_for_graphik(x, y, gauss_interpolation)
        results['stirling'] = generate_data_for_graphik(x, y, stirling_interpolation)

    if len(x) % 2 == 0 and check_finite_differences(x):
        bessel_result = bessel_interpolation(x, y, value)
        results['bessel'] = generate_data_for_graphik(x, y, bessel_interpolation)

    print(f"Метод Лагранжа: {lagrange_result}\n")
    xs, ys = generate_data_for_graphik(x, y, lagrange_interpolation)
    plot_interpolation(xs, ys, lagrange_interpolation, "Метод Лагранжа", value, x, y)

    if newton_result is not None:
        print(f"Метод Ньютона с разделенными разностями: {newton_result}\n")
        xs, ys = generate_data_for_graphik(x, y, newton_interpolation_with_shared_difference)
        plot_interpolation(xs, ys, newton_interpolation_with_shared_difference,
                           "Метод Ньютона с разделенными разностями",
                           value, x, y)

    else:
        print(
            "Невозможно построить интерполяционный многочлен Ньютона с разделенными разностями из-за некорректных входных данных\n"
            "Проверьте значения x, они не должны быть равноотстоящими\n")

    if gauss_result is not None:
        print(f"Метод Гаусса: {gauss_result}\n")
        xs, ys = generate_data_for_graphik(x, y, gauss_interpolation)
        plot_interpolation(xs, ys, gauss_interpolation, "Метод Гаусса", value, x, y)
    else:
        print("Невозможно построить интерполяционный многочлен Гаусса из-за некорректных входных данных\n"
              "Проверьте значения x, они должны быть равноотстоящими и их количество должно быть нечетным\n")

    if stirling_result is not None:
        print(f"Метод Стирлинга: {stirling_result}\n")
        xs, ys = generate_data_for_graphik(x, y, gauss_interpolation)
        plot_interpolation(xs, ys, stirling_interpolation, "Метод Стирлинга", value, x, y)
    else:
        print("Невозможно построить интерполяционный многочлен Стирлинга из-за некорректных входных данных\n"
              "Проверьте значения x, они должны быть равноотстоящими и их количество должно быть нечетным\n")

    if bessel_result is not None:
        print(f"Метод Бесселя: {bessel_result}\n")
        xs, ys = generate_data_for_graphik(x, y, bessel_interpolation)
        plot_interpolation(xs, ys, bessel_interpolation, "Метод Бесселя", value, x, y)
    else:
        print("Невозможно построить интерполяционный многочлен Бесселя из-за некорректных входных данных\n"
              "Проверьте значения x, они должны быть равноотстоящими и их количество должно быть четным\n")
    plot_all_interpolations(x, y, value, results, y_coord=lagrange_result)


if __name__ == "__main__":
    main()
