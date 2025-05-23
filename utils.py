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

    # for i in range(gen.shape[1]):
    #     print(f"Band {i+1}: R^2 {r2s[i]:.4f}; RMSE {rmses[i]:.4f}; RPD {rpds[i]:.4f}")

# Usage example
# Replace 'gen' and 'bare' with your actual data arrays
# print_metrics(gen, bare)
# print_data(input_data, num_data)
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from generator import Generator
# from discriminator import Discriminator
# import utils
# import pandas as pd
# from sklearn import model_selection
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
#
# # Load Sentinel-2 paired dataset
# dataset_full = pd.read_excel("data/S2_Veg_to_bare_paired_dataset.xlsx")
#
# # Normalize input data between 0 and 1
# scaler = MinMaxScaler()
# dataset_full.iloc[:, 1:] = scaler.fit_transform(dataset_full.iloc[:, 1:])
#
# # Train-test split
# train, test = model_selection.train_test_split(dataset_full, test_size=0.2, random_state=42)
# train.to_csv("results/train.csv", index=False)
# test.to_csv("results/test.csv", index=False)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# dataset_train = train.to_numpy()
# dataset_train = torch.tensor(dataset_train, dtype=torch.float32).to(device)
#
# dataset_test = test.to_numpy()
# dataset_test = torch.tensor(dataset_test, dtype=torch.float32).to(device)
#
# loss_comparison = nn.BCEWithLogitsLoss()
# L1_loss = nn.L1Loss()
#
# def discriminator_training(inputs, targets, discriminator_opt):
#     discriminator_opt.zero_grad()
#     output = discriminator(inputs, targets)
#     label = torch.ones(size=output.shape, dtype=torch.float, device=device)
#     real_loss = loss_comparison(output, label)
#     gen_image = generator(inputs).detach()
#     fake_output = discriminator(inputs, gen_image)
#     fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=device)
#     fake_loss = loss_comparison(fake_output, fake_label)
#     Total_loss = (real_loss + fake_loss) / 2
#     Total_loss.backward()
#     discriminator_opt.step()
#     return Total_loss
#
# def generator_training(inputs, targets, generator_opt, L1_lambda):
#     generator_opt.zero_grad()
#     generated_image = generator(inputs)
#     disc_output = discriminator(inputs, generated_image)
#     desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=device)
#     generator_loss = loss_comparison(disc_output, desired_output) + L1_lambda * L1_loss(generated_image, targets)
#     generator_loss.backward()
#     generator_opt.step()
#     return generator_loss, generated_image
#
# L1_lambda = 100
# NUM_EPOCHS = 5000
# lr = 0.01
#
# discriminator = Discriminator()
# generator = Generator()
#
# discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr)
# generator_opt = optim.Adam(generator.parameters(), lr=lr)
#
# discriminator = discriminator.to(device)
# generator = generator.to(device)
#
# for epoch in range(NUM_EPOCHS):
#     inputs = dataset_train[:, 1:11]  # 10 Sentinel-2 vegetation bands
#     targets = dataset_train[:, 11:21]  # 10 corresponding bare soil bands
#
#     Disc_Loss = discriminator_training(inputs, targets, discriminator_opt)
#     for i in range(2):
#         Gen_Loss, generator_image = generator_training(inputs, targets, generator_opt, L1_lambda)
#
#     if (epoch % 10) == 0:
#         print(f"After epoch {epoch + 1}, results for first 5 data:")
#         print("Vegetation")
#         utils.print_data(inputs, 5)
#         print("Generated Bare")
#         utils.print_data(generator_image, 5)
#         print("Actual Bare")
#         utils.print_data(targets, 5)
#
# # Save final generated dataset
# dataset_full = dataset_full.to_numpy()
# dataset_full = torch.tensor(dataset_full, dtype=torch.float32).to(device)
# inputs = dataset_full[:, 1:11]
# gen = generator(inputs)
# dataset_full = torch.cat((dataset_full, gen), dim=1)
#
# dataset_full = dataset_full.detach().cpu().numpy()
# dataset_full[:, 11:21] = scaler.inverse_transform(dataset_full[:, 11:21])
# columns = ['OC'] + [f'B{i}_veg' for i in range(1, 11)] + [f'B{i}_bare' for i in range(1, 11)] + [f'B{i}_bare_gen' for i in range(1, 11)]
# df = pd.DataFrame(dataset_full, columns=columns)
# df.to_csv('results/generated_test_S2.csv', index=False)
#
# # Display test results
# inputs = dataset_test[:, 1:11]
# targets = dataset_test[:, 11:21]
# gen = generator(inputs)
# veg = inputs.detach().cpu()
# gen = gen.detach().cpu()
# bare = targets.detach().cpu()
# print("Training Done. Showing Test Results (for first 5 data).")
# print("Vegetation")
# utils.print_data(veg, 5)
# print("Generated Bare")
# utils.print_data(gen, 5)
# print("Actual Bare")
# utils.print_data(bare, 5)
# utils.print_metrics(gen.numpy(), bare.numpy())
