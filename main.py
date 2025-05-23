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
#
#
# dataset_full = pd.read_csv("data/paired_dataset_veg2bare.csv")
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
#     #print(f"Loss {real_loss.item()} {fake_loss.item()} {Total_loss.item()}")
#     Total_loss.backward()
#     discriminator_opt.step()
#     return Total_loss
#
#
# def generator_training(inputs, targets, generator_opt, L1_lambda):
#     generator_opt.zero_grad()
#     generated_image = generator(inputs)
#     disc_output = discriminator(inputs, generated_image)
#     desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=device)
#     generator_loss = loss_comparison(disc_output, desired_output) + L1_lambda * torch.abs(
#         generated_image - targets).sum()
#     generator_loss.backward()
#     generator_opt.step()
#
#     return generator_loss, generated_image
#
# L1_lambda = 100
# NUM_EPOCHS= 5000
# lr=0.01
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
#     inputs = dataset_train[:,1:8]
#     targets = dataset_train[:,9:]
#
#     Disc_Loss = discriminator_training(inputs,targets,discriminator_opt)
#     for i in range(2):
#         Gen_Loss, generator_image = generator_training(inputs,targets, generator_opt, L1_lambda)
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
# dataset_full = dataset_full.to_numpy()
# dataset_full = torch.tensor(dataset_full, dtype=torch.float32).to(device)
# inputs = dataset_full[:, 1:8]
# gen = generator(inputs)
# dataset_full=torch.cat((dataset_full,gen),dim=1)
#
# columns = ['OC', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'B1_bare', 'B2_bare', 'B3_bare', 'B4_bare', 'B5_bare', 'B6_bare', 'B7_bare','B1_bare_gen', 'B2_bare_gen', 'B3_bare_gen', 'B4_bare_gen', 'B5_bare_gen', 'B6_bare_gen', 'B7_bare_gen']
# df = pd.DataFrame(dataset_full.detach().cpu().numpy(), columns=columns)
# df.to_csv('results/generated_test1.csv', index=False)
#
# inputs = dataset_test[:, 1:8]
# targets = dataset_test[:, 9:]
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
#     generator_loss = loss_comparison(disc_output, desired_output) + L1_lambda * torch.abs(
#         generated_image - targets).sum()
#     generator_loss.backward()
#     generator_opt.step()
#     return generator_loss, generated_image
#
# L1_lambda = 100
# NUM_EPOCHS = 500
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
# columns = ['OC'] + [f'B{i}_veg' for i in range(1, 11)] + [f'B{i}_bare' for i in range(1, 11)] + [f'B{i}_bare_gen' for i in range(1, 11)]
# df = pd.DataFrame(dataset_full.detach().cpu().numpy(), columns=columns)
# df.to_csv('results/generated_test_S3.csv', index=False)
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
# NUM_EPOCHS = 500
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
# # Select only the originally scaled columns (Vegetation + Bare Soil bands)
#
# # Restore original scale for Vegetation + Bare Soil bands
# scaled_columns = dataset_full[:, 1:21]  # Only the original 10 Veg + 10 Bare bands
# scaled_columns = scaler.inverse_transform(scaled_columns)  # Inverse transform
# dataset_full[:, 1:21] = scaled_columns  # Assign back
#
# # ✅ Apply inverse transformation to generated bare soil bands (columns 21-31)
# generated_bands = dataset_full[:, 21:31]  # Select generated bare soil bands
# generated_bands = scaler.inverse_transform(generated_bands)  # Restore scale
# dataset_full[:, 21:31] = generated_bands  # Assign back
# # scaled_columns = dataset_full[:, 1:21]  # Only the 10 Veg + 10 Bare bands
# # scaled_columns = scaler.inverse_transform(scaled_columns)  # Inverse transform only these columns
# #
# # # Assign back the inverse-transformed values
# # dataset_full[:, 1:21] = scaled_columns  # Restore only the original bands
#   # Apply to all scaled columns
# columns = ['OC'] + [f'B{i}_veg' for i in range(1, 11)] + [f'B{i}_bare' for i in range(1, 11)] + [f'B{i}_bare_gen' for i in range(1, 11)]
# df = pd.DataFrame(dataset_full, columns=columns)
# df.to_csv('results/generated_test1000_S2.csv', index=False)
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


import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
import utils
import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Sentinel-2 paired dataset
dataset_full = pd.read_excel("data/S2_Veg_to_bare_paired_dataset.xlsx")

# Normalize input data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(dataset_full.iloc[:, 1:])  # Fit on all bands (vegetation + bare soil)
dataset_full.iloc[:, 1:] = scaler.transform(dataset_full.iloc[:, 1:])

# Create a separate scaler for bare soil bands
bare_soil_scaler = MinMaxScaler()
bare_soil_scaler.fit(dataset_full.iloc[:, 11:21])  # Fit only on original bare soil bands

# Train-test split
train, test = model_selection.train_test_split(dataset_full, test_size=0.2, random_state=42)
train.to_csv("results/train.csv", index=False)
test.to_csv("results/test.csv", index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = train.to_numpy()
dataset_train = torch.tensor(dataset_train, dtype=torch.float32).to(device)

dataset_test = test.to_numpy()
dataset_test = torch.tensor(dataset_test, dtype=torch.float32).to(device)

loss_comparison = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

def discriminator_training(inputs, targets, discriminator_opt):
    discriminator_opt.zero_grad()
    output = discriminator(inputs, targets)
    label = torch.ones(size=output.shape, dtype=torch.float, device=device)
    real_loss = loss_comparison(output, label)
    gen_image = generator(inputs).detach()
    fake_output = discriminator(inputs, gen_image)
    fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=device)
    fake_loss = loss_comparison(fake_output, fake_label)
    Total_loss = (real_loss + fake_loss) / 2
    Total_loss.backward()
    discriminator_opt.step()
    return Total_loss

def generator_training(inputs, targets, generator_opt, L1_lambda):
    generator_opt.zero_grad()
    generated_image = generator(inputs)
    disc_output = discriminator(inputs, generated_image)
    desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=device)
    generator_loss = loss_comparison(disc_output, desired_output) + L1_lambda * L1_loss(generated_image, targets)
    generator_loss.backward()
    generator_opt.step()
    return generator_loss, generated_image

L1_lambda = 100
NUM_EPOCHS = 500
lr = 0.01

discriminator = Discriminator()
generator = Generator()

discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr)
generator_opt = optim.Adam(generator.parameters(), lr=lr)

discriminator = discriminator.to(device)
generator = generator.to(device)

for epoch in range(NUM_EPOCHS):
    inputs = dataset_train[:, 1:11]  # 10 Sentinel-2 vegetation bands
    targets = dataset_train[:, 11:21]  # 10 corresponding bare soil bands

    Disc_Loss = discriminator_training(inputs, targets, discriminator_opt)
    for i in range(2):
        Gen_Loss, generator_image = generator_training(inputs, targets, generator_opt, L1_lambda)

    if (epoch % 10) == 0:
        print(f"After epoch {epoch + 1}, results for first 5 data:")
        print("Vegetation")
        utils.print_data(inputs, 5)
        print("Generated Bare")
        utils.print_data(generator_image, 5)
        print("Actual Bare")
        utils.print_data(targets, 5)

# Save final generated dataset
dataset_full = dataset_full.to_numpy()
dataset_full = torch.tensor(dataset_full, dtype=torch.float32).to(device)
inputs = dataset_full[:, 1:11]
gen = generator(inputs)
dataset_full = torch.cat((dataset_full, gen), dim=1)

dataset_full = dataset_full.detach().cpu().numpy()

# Restore original scale for Vegetation + Bare Soil bands
scaled_columns = dataset_full[:, 1:21]  # Only the original 10 Veg + 10 Bare bands
scaled_columns = scaler.inverse_transform(scaled_columns)  # Inverse transform
dataset_full[:, 1:21] = scaled_columns  # Assign back

# ✅ Apply inverse transformation to generated bare soil bands using bare soil scaler
generated_bands = dataset_full[:, 21:31]  # Select generated bare soil bands
generated_bands = bare_soil_scaler.inverse_transform(generated_bands)  # Restore scale
dataset_full[:, 21:31] = generated_bands  # Assign back

# Save final dataset
df_columns = ['OC'] + [f'B{i}_veg' for i in range(1, 11)] + [f'B{i}_bare' for i in range(1, 11)] + [f'B{i}_bare_gen' for i in range(1, 11)]
df = pd.DataFrame(dataset_full, columns=df_columns)
df.to_csv('results/generated_test_S2.csv', index=False)

# Display test results
inputs = dataset_test[:, 1:11]
targets = dataset_test[:, 11:21]
gen = generator(inputs)
veg = inputs.detach().cpu()
gen = gen.detach().cpu()
bare = targets.detach().cpu()
print("Training Done. Showing Test Results (for first 5 data).")
print("Vegetation")
utils.print_data(veg, 5)
print("Generated Bare")
utils.print_data(gen, 5)
print("Actual Bare")
utils.print_data(bare, 5)
utils.print_metrics(gen.numpy(), bare.numpy())
