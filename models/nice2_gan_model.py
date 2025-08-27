import time, itertools
from data.image_folder import ImageFolder
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from . import networks
from util.Nice_utils import *
from glob import glob
from thop import profile
from thop import clever_format
import cv2
import os
import numpy as np
from tqdm import tqdm

class NICE2(object):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        return parser
        
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'NICE2_light'
        else:
            self.model_name = 'NICE2'

        self.result_dir = args.result_dir
        self.dataset = args.dataroot
        self.name = args.name
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_epoch_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.load_size
        self.img_ch = args.input_nc
        if len(args.gpu_ids) > 0:
            self.device = args.gpu_ids[0]
        else:
            self.device = 'cpu'
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        
        # Cache for intermediate results to avoid recomputation
        self.feature_cache = {}
        
        # Performance monitoring
        self.log_freq = 50  # Print every 50 steps instead of every step
        self.recent_losses = {'d_loss': [], 'g_loss': []}
        
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# Model: NICE2 (Optimized)")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)

        self.build_model()

    ##################################################################################
    # Model
    ##################################################################################

    def get_infinite_dataloader(self, loader):
        """Create infinite dataloader to avoid try/except pattern"""
        while True:
            yield from loader

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join(self.dataset, 'trainA'), train_transform, extend_paths=True, return_paths=True)
        self.trainB = ImageFolder(os.path.join(self.dataset, 'trainB'), train_transform, extend_paths=True, return_paths=True)
        self.testA = ImageFolder(os.path.join(self.dataset, 'testA'), test_transform, extend_paths=True, return_paths=True)
        self.testB = ImageFolder(os.path.join(self.dataset, 'testB'), test_transform, extend_paths=True, return_paths=True)
        
        # Optimized data loading with higher num_workers and prefetch_factor
        num_workers = min(4, os.cpu_count())
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=2)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=2)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

        # Create infinite iterators to avoid try/except pattern
        self.trainA_iter = self.get_infinite_dataloader(self.trainA_loader)
        self.trainB_iter = self.get_infinite_dataloader(self.trainB_loader)
        self.testA_iter = self.get_infinite_dataloader(self.testA_loader)
        self.testB_iter = self.get_infinite_dataloader(self.testB_loader)

        """ Define Generator, Discriminator """
        self.gen2B = networks.NiceResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = networks.NiceResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = networks.NiceDiscriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = networks.NiceDiscriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)

        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input,))
        macs, params = clever_format([macs * 2, params * 2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        _, _, _, _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae,))
        macs, params = clever_format([macs * 2, params * 2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

    def lightweight_monitoring(self, step, real_A, real_B):
        """Lightweight monitoring without full evaluation"""
        with torch.no_grad():
            _, _, _, _, real_A_z = self.disA(real_A)
            _, _, _, _, real_B_z = self.disB(real_B)
            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)
            
            # Simple quality metrics - store in progress bar postfix instead of printing
            l1_loss_A2B = self.L1_loss(fake_A2B, real_B).item()
            l1_loss_B2A = self.L1_loss(fake_B2A, real_A).item()
            
            # Update progress bar with quality metrics (non-blocking)
            if hasattr(self, '_current_pbar'):
                self._current_pbar.set_postfix({
                    'D_loss': f"{self.recent_losses['d_loss'][-1] if self.recent_losses['d_loss'] else 0:.4f}",
                    'G_loss': f"{self.recent_losses['g_loss'][-1] if self.recent_losses['g_loss'] else 0:.4f}",
                    'L1_A2B': f"{l1_loss_A2B:.4f}",
                    'L1_B2A': f"{l1_loss_B2A:.4f}"
                })

    def train(self):
        loss_A = []
        loss_B = []

        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

        self.start_iter = 1
        if self.resume:
            params = torch.load(os.path.join(self.result_dir, self.name + '_params_latest.pt'))
            self.gen2B.load_state_dict(params['gen2B'])
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])
            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])
            self.start_iter = params['start_iter'] + 1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
            print("Model loaded successfully")

        print("self.start_iter", self.start_iter)
        print('NICE2 training start with optimizations!')
        start_time = time.time()
        last_log_time = start_time

        # Create progress bar
        pbar = tqdm(range(self.start_iter, self.iteration + 1), 
                   desc="Training", 
                   unit="step",
                   ncols=120)
        
        # Store reference for lightweight monitoring
        self._current_pbar = pbar

        for step in pbar:
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # Optimized data loading - no try/except needed
            real_A, _ = next(self.trainA_iter)
            real_B, _ = next(self.trainB_iter)
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Cache discriminator features to avoid recomputation
            real_LA_logit, real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(real_A)
            real_LB_logit, real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(real_B)

            # Cache for later use
            self.feature_cache['real_A_z'] = real_A_z.detach()
            self.feature_cache['real_B_z'] = real_B_z.detach()

            # Update D
            self.D_optim.zero_grad()

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
            D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            # Use cached features to avoid recomputation
            real_A_z = self.feature_cache['real_A_z']
            real_B_z = self.feature_cache['real_B_z']

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(fake_A2B)

            fake_B2A2B = self.gen2B(fake_A_z)
            fake_A2B2A = self.gen2A(fake_B_z)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
            G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

            G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_z)
            fake_B2B = self.gen2B(real_B_z)

            G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B)

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # Accumulate losses for efficient logging
            self.recent_losses['d_loss'].append(Discriminator_loss.item())
            self.recent_losses['g_loss'].append(Generator_loss.item())

            # Update progress bar with current losses (non-blocking)
            pbar.set_postfix({
                'D_loss': f"{Discriminator_loss.item():.4f}",
                'G_loss': f"{Generator_loss.item():.4f}"
            })

            # Periodic detailed logging (much less frequent)
            if step % self.log_freq == 0:
                current_time = time.time()
                avg_d_loss = sum(self.recent_losses['d_loss']) / len(self.recent_losses['d_loss'])
                avg_g_loss = sum(self.recent_losses['g_loss']) / len(self.recent_losses['g_loss'])
                time_per_step = (current_time - last_log_time) / self.log_freq
                
                print(f"\n[{step:5d}/{self.iteration:5d}] "
                      f"avg_d_loss: {avg_d_loss:.6f}, avg_g_loss: {avg_g_loss:.6f}, "
                      f"time/step: {time_per_step:.3f}s")
                
                # Clear accumulated losses
                self.recent_losses['d_loss'].clear()
                self.recent_losses['g_loss'].clear()
                last_log_time = current_time

            if step % 1 == 0:
                loss_A.append([D_loss_A, G_loss_A, (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA), G_cycle_loss_A, G_recon_loss_A])
                loss_B.append([D_loss_B, G_loss_B, (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB), G_cycle_loss_B, G_recon_loss_B])

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.name, 'model'), step)

            # Reduced visualization frequency - evaluate every print_freq * 5 steps instead of every print_freq
            if step % self.print_freq == 0:
                # Save learning rates to progress bar instead of printing
                pbar.set_description(f"Training - D_lr: {self.D_optim.param_groups[0]['lr']:.6f}, G_lr: {self.G_optim.param_groups[0]['lr']:.6f}")
                self.save_path("_params_latest.pt", step)
                
                # Lightweight monitoring instead of full evaluation
                self.lightweight_monitoring(step, real_A, real_B)

            # Full visualization less frequently
            if step % (self.print_freq * 5) == 0:
                self.full_evaluation(step)

            # Memory cleanup every 100 steps
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close progress bar
        pbar.close()
        
        # Final training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"Average time per step: {total_time/(self.iteration-self.start_iter+1):.3f}s")

        np.savetxt('loss_log_A', loss_A)
        np.savetxt('loss_log_B', loss_B)

    def full_evaluation(self, step):
        """Full evaluation with image generation - called less frequently"""
        testnum = 2  # Reduced from 4 to 2 for efficiency
        train_sample_num = testnum
        test_sample_num = testnum
        
        A2B = np.zeros((self.img_size * 5, 0, 3))
        B2A = np.zeros((self.img_size * 5, 0, 3))

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()
        
        with torch.no_grad():
            # Train samples evaluation - stay on GPU
            for _ in range(train_sample_num):
                real_A, _ = next(self.trainA_iter)
                real_B, _ = next(self.trainB_iter)
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                _, _, _, A_heatmap, real_A_z = self.disA(real_A)
                _, _, _, B_heatmap, real_B_z = self.disB(real_B)
                fake_A2B = self.gen2B(real_A_z)
                fake_B2A = self.gen2A(real_B_z)
                _, _, _, _, fake_A_z = self.disA(fake_B2A)
                _, _, _, _, fake_B_z = self.disB(fake_A2B)
                fake_B2A2B = self.gen2B(fake_A_z)
                fake_A2B2A = self.gen2A(fake_B_z)
                fake_A2A = self.gen2A(real_A_z)
                fake_B2B = self.gen2B(real_B_z)

                # Move to CPU only for visualization
                A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0].cpu()))),
                                                           cam(tensor2numpy(A_heatmap[0].cpu()), self.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2A[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2B[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0].cpu())))), 0)), 1)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0].cpu()))),
                                                           cam(tensor2numpy(B_heatmap[0].cpu()), self.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2B[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2A[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0].cpu())))), 0)), 1)

            # Test samples evaluation
            for _ in range(test_sample_num):
                real_A, _ = next(self.testA_iter)
                real_B, _ = next(self.testB_iter)
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                _, _, _, A_heatmap, real_A_z = self.disA(real_A)
                _, _, _, B_heatmap, real_B_z = self.disB(real_B)
                fake_A2B = self.gen2B(real_A_z)
                fake_B2A = self.gen2A(real_B_z)
                _, _, _, _, fake_A_z = self.disA(fake_B2A)
                _, _, _, _, fake_B_z = self.disB(fake_A2B)
                fake_B2A2B = self.gen2B(fake_A_z)
                fake_A2B2A = self.gen2A(fake_B_z)
                fake_A2A = self.gen2A(real_A_z)
                fake_B2B = self.gen2B(real_B_z)

                A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0].cpu()))),
                                                           cam(tensor2numpy(A_heatmap[0].cpu()), self.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2A[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2B[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0].cpu())))), 0)), 1)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0].cpu()))),
                                                           cam(tensor2numpy(B_heatmap[0].cpu()), self.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2B[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2A[0].cpu()))),
                                                           RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0].cpu())))), 0)), 1)

        # Save visualization images
        os.makedirs(os.path.join(self.result_dir, self.name, 'images'), exist_ok=True)
        cv2.imwrite(os.path.join(self.result_dir, self.name, 'images', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.name, 'images', 'B2A_%07d.png' % step), B2A * 255.0)

        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

    def save(self, dir, step):
        os.makedirs(dir, exist_ok=True)
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, self.name + '_params_%07d.pt' % step))

    def save_path(self, path_g, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.name + path_g))

    def load(self):
        params = torch.load(os.path.join(self.result_dir, self.name + '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print("Starting test with iter:", self.start_iter)

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()
        
        # Create output directories
        os.makedirs(os.path.join(self.result_dir, self.name, 'fakeB'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, self.name, 'fakeA'), exist_ok=True)
        
        with torch.no_grad():
            for n, (real_A, real_A_path) in enumerate(self.testA_loader):
                real_A = real_A.to(self.device)
                
                _, _, _, _, real_A_z = self.disA(real_A)
                fake_A2B = self.gen2B(real_A_z)

                A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0].cpu())))
                print(real_A_path[0])
                cv2.imwrite(os.path.join(self.result_dir, self.name, 'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

            for n, (real_B, real_B_path) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)
                
                _, _, _, _, real_B_z = self.disB(real_B)
                fake_B2A = self.gen2A(real_B_z)

                B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0].cpu())))
                print(real_B_path[0])
                cv2.imwrite(os.path.join(self.result_dir, self.name, 'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)