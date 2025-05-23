from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

def print_data(input_data, num_data):
    for i in range(num_data):
        input_as_list = input_data[i].tolist()
        input_as_list = [round(num, 4) for num in input_as_list]
        data_str = "".join([str(num).ljust(10) for num in input_as_list])
        print(data_str)

def print_metrics(gen, bare):
    r2s = []
    rmses = []
    rpds = []

    for i in range(gen.shape[1]):
        g = gen[:, i]
        b = bare[:, i]
        r2s.append(r2_score(b, g))  # Correct order for R-squared
        rmse = root_mean_squared_error(b, g)
        rmses.append(rmse)

        # Conventional RPD calculation
        std_dev = np.std(b)
        rpd = std_dev / rmse if rmse != 0 else float('inf')
        rpds.append(rpd)
