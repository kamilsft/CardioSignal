# Adam optimizer is used for the training of the model from https://github.com/cankocagil/DCGAN

from torch.utils.data import DataLoader, Dataset # might be needed for DataLoader
from dataset import dataset
from Models.cycle_gan.Generators.Generator import Generator
from Models.cycle_gan.discriminator.Discriminator import Discriminator
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import random_split # this is used to split the data into training and validation sets
import argparse

# compute canada 
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lambda_L1', type=float, default=5.0)
parser.add_argument('--lambda_identity', type=float, default=0.5)
parser.add_argument('--lambda_cycle', type=float, default=10.0)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
Beta1 = args.beta1
lambda_L1 = args.lambda_L1
lambda_identity = args.lambda_identity
lambda_cycle = args.lambda_cycle
save_dir = args.save_dir


healthy_data_folder = os.getenv('HEALTHY_DATA_FOLDER', r'YourPATh')
unhealthy_data_folder = os.getenv('UNHEALTHY_DATA_FOLDER', r'YourPATH')

# Training configs
# healthy_data_folder = r''
# unhealthy_data_folder = r''

# batch_size = 4 # number of traing samples in each duration
# num_epochs = 10
# learning_rate = 0.0002
# Beta1 = 0.5  # optimizer, helps with the stability of the training "memory" helps the model to remember past guess and stay in the right direction, fast adaptation

# lambda_L1 = 5.0 # lambda for the identity loss
# lambda_identity = 0.5  #lambda for the cycle consistency loss
# lambda_cycle = 10.0 #lambda 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataset([healthy_data_folder, unhealthy_data_folder])
training_size = int(0.95 * len(dataset))  # 80% for training
validation_size = len(dataset) - training_size  # 20% for validation
training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# initializing the models
generatorECGPCG = Generator(input_channels=1, sequence_length=48000).to(device)
generatorPCGECG = Generator(input_channels=1, sequence_length=48000).to(device)
discriminatorPCG = Discriminator(input_channels=1).to(device)
discriminatorECG = Discriminator(input_channels=1).to(device)
# initializing the optimizers and loss functions

generator_optimizer = torch.optim.Adam(list(generatorPCGECG.parameters()) + list(generatorECGPCG.parameters()), lr=learning_rate, betas=(Beta1, 0.999))
discriminatorPCG_optimizer = torch.optim.Adam(discriminatorPCG.parameters(), lr=learning_rate, betas=(Beta1, 0.999))
discriminatorECG_optimizer = torch.optim.Adam(discriminatorECG.parameters(), lr=learning_rate, betas=(Beta1, 0.999))
l1_loss = nn.L1Loss()
loss_GAN = nn.BCELoss()

# overfitting_check
previous_validation_loss = float('inf')
previous_training_loss = float('inf') 

# training loop
for epoch in range(num_epochs):
    for pcg, ecg in training_dataloader:
        ecg = ecg.to(device)
        pcg = pcg.to(device)

        # Train Discriminator PCG
        fake_pcg = generatorECGPCG(ecg).detach() 
        with torch.no_grad():
            real_out_pcg = discriminatorPCG(pcg) # speeding up the process by not calculating gradients for real PCG 
        fake_out_pcg = discriminatorPCG(fake_pcg)
        real_label = torch.ones(real_out_pcg.shape, device=device)
        fake_label = torch.zeros(fake_out_pcg.shape, device=device)
        # losses
        real_loss = loss_GAN(real_out_pcg, real_label) * 0.47 
        fake_loss = loss_GAN(fake_out_pcg, fake_label) * 0.53
        discriminatorPCG_loss = real_loss + fake_loss # effective weight that needs to be achievned is 1

        discriminatorPCG_optimizer.zero_grad() 
        #debugging
        if epoch % 3 == 0: 
            print(f"Discriminator PCG Real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f}, Total: {discriminatorPCG_loss.item():.4f}")
        discriminatorPCG_loss.backward()
        #torch.nn.utils.clip_grad_norm_(discriminatorPCG.parameters(), max_norm=1.0)  # gradient clipping
        discriminatorPCG_optimizer.step()

        # Train Discriminator ECG
        fake_ecg = generatorPCGECG(pcg).detach()
        with torch.no_grad():
            real_out_ecg = discriminatorECG(ecg) # speeding up the process by not calculating gradients for real ECG 
        fake_out_ecg = discriminatorECG(fake_ecg)
        real_label = torch.ones(real_out_ecg.shape, device=device)  
        fake_label = torch.zeros(fake_out_ecg.shape, device=device)
        # losses
        real_loss = loss_GAN(real_out_ecg, real_label) * 0.47
        fake_loss = loss_GAN(fake_out_ecg, fake_label) * 0.53
        discriminatorECG_loss = real_loss + fake_loss

        discriminatorECG_optimizer.zero_grad()
        #debugging
        if epoch % 3 == 0:
            print(f"Discriminator ECG Real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f}, Total: {discriminatorECG_loss.item():.4f}")
        discriminatorECG_loss.backward()
        discriminatorECG_optimizer.step()
        
        # Train Generators
        generator_optimizer.zero_grad()

        # GAN loss or adversarial loss 
        # adversarial loss for generatorPCGECG
        # generatorPCGECG tries to make fake ECG look real to discriminatorECG
        generated_ecg = generatorPCGECG(pcg)
        output_from_GeneratorPCG = discriminatorECG(generated_ecg)
        adversarial_loss_ecg = loss_GAN(output_from_GeneratorPCG, real_label)

        # adversarial loss for generatorECGPCG
        # generatorECGPCG tries to make fake PCG look real to discriminatorPCG
        generated_pcg = generatorECGPCG(ecg)
        output_from_GeneratorECG = discriminatorPCG(generated_pcg)
        adversarial_loss_pcg = loss_GAN(output_from_GeneratorECG, real_label)

        # Total adversarial loss // to preserving 
        adversarial_loss = (adversarial_loss_ecg + adversarial_loss_pcg)

        #cycle consistency loss 
        #first we generate fake ECG from PCG and then generate fake PCG from the generated ECG
        cycle_pcg = generatorECGPCG(generated_ecg)
        cycle_loss_pcg = l1_loss(cycle_pcg, pcg) * lambda_cycle
        # then we generate fake PCG from ECG and then generate fake ECG from the generated PCG
        cycle_ecg = generatorPCGECG(generated_pcg) # l1 loss
        cycle_loss_ecg = l1_loss(cycle_ecg, ecg) * lambda_cycle
        # Total cycle consistency loss
        cycle_loss = (cycle_loss_pcg + cycle_loss_ecg)

        # identity loss or l1 loss
        # generatorECGPCG tries to generate PCG from PCG and generatorPCGECG tries to generate ECG from ECG
        # this is to ensure that the generators are not changing the input data
        identity_loss_pcg = l1_loss(generatorECGPCG(pcg), pcg) * lambda_identity * lambda_L1
        identity_loss_ecg = l1_loss(generatorPCGECG(ecg), ecg) * lambda_identity * lambda_L1
        # Total identity loss
        identity_loss = (identity_loss_pcg + identity_loss_ecg)

        # Total generator loss
        generator_loss = adversarial_loss + cycle_loss + identity_loss
        generator_loss.backward()
        generator_optimizer.step()
        # Print losses
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Discriminator_PCGLoss: {discriminatorPCG_loss.item():.4f}, "
                  f"Discriminator_ECGLoss: {discriminatorECG_loss.item():.4f}, "
                  f"Generator_Loss: {generator_loss.item():.4f}, "
                  f"CycleLoss: {cycle_loss.item():.4f}, "
                  f"IdentityLoss: {identity_loss.item():.4f}")
            
            os.makedirs(save_dir, exist_ok=True)

            # we have to save our models checkpoints so the data wont get lost
            torch.save({
                'epoch': epoch,
                'model_state_dict': generatorECGPCG.state_dict(),
                'optimizer_state_dict': generator_optimizer.state_dict(),
                'loss': generator_loss,
            }, os.path.join(save_dir, f'checkpoint_GeneratorECGPCG_epoch{epoch+1}.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': generatorPCGECG.state_dict(),
                'optimizer_state_dict': generator_optimizer.state_dict(),
                'loss': generator_loss,
            }, os.path.join(save_dir, f'checkpoint_GeneratorPCGECG_epoch{epoch+1}.pth'))
            torch.save({
                'epoch': epoch, 
                'model_state_dict': discriminatorPCG.state_dict(),
                'optimizer_state_dict': discriminatorPCG_optimizer.state_dict(),
                'loss': discriminatorPCG_loss,
            }, os.path.join(save_dir, f'checkpoint_DiscriminatorPCG_epoch{epoch+1}.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminatorECG.state_dict(),
                'optimizer_state_dict': discriminatorECG_optimizer.state_dict(),
                'loss': discriminatorECG_loss,
            }, os.path.join(save_dir, f'checkpoint_DiscriminatorECG_epoch{epoch+1}.pth'))

         
            if epoch >= 1:
                # setting generators back to training mode
                generatorECGPCG.train()
                generatorPCGECG.train()

                # The overfitting check needs to be done here using the validation loss and compare it

                # overfitting_check if the validation loss is increasing while the training loss is decreasing, it indicates overfitting
                # This can be done by keeping track of the validation loss and comparing it with the training loss
                # if the validation loss is decreasing while the training loss is decreasing, it indicates that the model is learning
                generatorECGPCG.eval()
                generatorPCGECG.eval()
                validation_generator_loss = 0.0
                validation_cycle_loss = 0.0
                validation_identity_loss = 0.0
                validation_L1_loss = 0.0
                validation_adversarial_loss = 0.0
                total_batches = 0

                with torch.no_grad():
                    for validation_ecg, validation_pcg in validation_dataloader:
                        validation_ecg = validation_ecg.to(device)
                        validation_pcg = validation_pcg.to(device)
                        fake_validation_pcg = generatorECGPCG(validation_ecg)
                        fake_validation_ecg = generatorPCGECG(validation_pcg)
                        # calulating adversarial losses for the validation set
                        out_validation_pcg = discriminatorPCG(fake_validation_pcg)
                        out_validation_ecg = discriminatorECG(fake_validation_ecg)
                        validation_adversarial_loss = (loss_GAN(out_validation_pcg, torch.ones_like(out_validation_pcg, device=device)) +
                                                        loss_GAN(out_validation_ecg, torch.ones_like(out_validation_ecg, device=device))) / 2
                        # calulating cycle consistency losses for the validation set
                        validation_cycle_pcg = generatorECGPCG(fake_validation_ecg)
                        validation_cycle_ecg = generatorPCGECG(fake_validation_pcg)
                        cycle_loss_validation = (l1_loss(validation_cycle_pcg, validation_pcg) + l1_loss(validation_cycle_ecg, validation_ecg)) * lambda_cycle
                        validation_cycle_loss += cycle_loss_validation.item()

                        # calulating identity losses for the validation set
                        identity_loss_validation_pcg = l1_loss(generatorECGPCG(validation_pcg), validation_pcg) * lambda_identity
                        identity_loss_validation_ecg = l1_loss(generatorPCGECG(validation_ecg), validation_ecg) * lambda_identity
                        validation_identity_loss += (identity_loss_validation_pcg + identity_loss_validation_ecg).item()

                        # total generator validation loss for the validation set
                        validation_generator_loss += (validation_adversarial_loss + cycle_loss_validation + validation_identity_loss).item()
                        total_batches += 1

                        # average losses for the validation set
                        average_generator_loss = validation_generator_loss / total_batches
                        average_cycle_loss = validation_cycle_loss / total_batches
                        average_identity_loss = validation_identity_loss / total_batches
                        average_adversarial_loss = validation_adversarial_loss / total_batches
                        # Print validation losses
                        print(f"Validation Losses at Epoch {epoch + 1}: "
                            f"Generator Loss: {validation_generator_loss:.4f}, "
                            f"Cycle Loss: {validation_cycle_loss:.4f}, "
                            f"Identity Loss: {validation_identity_loss:.4f}, "
                            f"Adversarial Loss: {validation_adversarial_loss:.4f}")
                        
                        # restore the training mode for the generators
                        generatorECGPCG.train()
                        generatorPCGECG.train()

                        # checking for overfitting
                        if epoch > 0:
                            if validation_generator_loss > previous_validation_loss and generator_loss < previous_training_loss:
                                print("Overfitting being detected.  Validation loss is increasing while training loss is decreasing.")
                            else:
                                print("No overfitting being detected. Continuing training.")