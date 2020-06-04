"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from . import networks
from util.UNIT_utils import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import os,sys
import tensorboardX
from data.image_folder import ImageFolder

class UNIT(nn.Module):
    @staticmethod
    def modify_commandline_options(parser,isTrain):
        return parser
    def __init__(self, args):
        super(UNIT, self).__init__()

        lr = args.lr
        args.vgg_w = 0
        # Initiate the networks
        self.args = args
        if len(args.gpu_ids)>0:
            self.device = args.gpu_ids[0]
        else:
            self.device = 'cpu'
        args.device = self.device
        self.gen_a = networks.VAEGen(args.input_nc, args)  # auto-encoder for domain a
        self.gen_b = networks.VAEGen(args.input_nc, args)  # auto-encoder for domain b
        self.dis_a = networks.MsImageDis(args.input_nc,args)  # discriminator for domain a
        self.dis_b = networks.MsImageDis(args.input_nc,args)  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = args.beta1
        beta2 = args.beta2
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=args.weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=args.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_opt, args)
        self.gen_scheduler = get_scheduler(self.gen_opt, args)

        # Network weight initialization
        self.apply(weights_init(args.init))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        #self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        #self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, args):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if args.recon_x_cyc_w > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if args.recon_x_cyc_w > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if args.vgg_w > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if args.vgg_w > 0 else 0
        # total loss
        self.loss_gen_total = args.gan_w * self.loss_gen_adv_a + \
                              args.gan_w * self.loss_gen_adv_b + \
                              args.recon_x_w * self.loss_gen_recon_x_a + \
                              args.recon_kl_w * self.loss_gen_recon_kl_a + \
                              args.recon_x_w * self.loss_gen_recon_x_b + \
                              args.recon_kl_w * self.loss_gen_recon_kl_b + \
                              args.recon_x_cyc_w * self.loss_gen_cyc_x_a + \
                              args.recon_kl_cyc_w * self.loss_gen_recon_kl_cyc_aba + \
                              args.recon_x_cyc_w * self.loss_gen_cyc_x_b + \
                              args.recon_kl_cyc_w * self.loss_gen_recon_kl_cyc_bab + \
                              args.vgg_w * self.loss_gen_vgg_a + \
                              args.vgg_w * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()


    def sample(self, x_a, x_b):
        #self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        #self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, args):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = args.gan_w * self.loss_dis_a + args.gan_w * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, args):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, args, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, args, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def train(self):
        args = self.args
        display_size = args.display_ncols
        max_iter = args.iteration
        self.to(self.device)
        train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(args)
        train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).to(self.device)
        train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).to(self.device)
        test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).to(self.device)
        test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).to(self.device)

        # Setup logger and output folders
        model_name = args.name
        train_writer = tensorboardX.SummaryWriter(os.path.join(args.result_dir,model_name))
        output_directory = os.path.join(args.result_dir,model_name)
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

        # Start training
        iterations = self.resume(checkpoint_directory, args=args) if args.resume else 0
        while True:
            for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):

                images_a, images_b = images_a.to(self.device).detach(), images_b.to(self.device).detach()

                with Timer("Elapsed time in update: %f"):
                    # Main training code
                    self.dis_update(images_a, images_b, args)
                    self.gen_update(images_a, images_b, args)
                    self.update_learning_rate()
                    if len(args.gpu_ids)>0: self.torch.cuda.synchronize()

                # Dump training stats in log file
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, self, train_writer)

                # Write images
                if (iterations + 1) % args.display_freq == 0 or (iterations+1) >= max_iter:
                    with torch.no_grad():
                        test_image_outputs = self.sample(test_display_images_a, test_display_images_b)
                        train_image_outputs = self.sample(train_display_images_a, train_display_images_b)
                    write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                    write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                    # HTML
                    write_html(output_directory + "/index.html", iterations + 1, args.display_freq, 'images')

                if (iterations + 1) % args.print_freq == 0 or (iterations+1) >= max_iter:
                    with torch.no_grad():
                        image_outputs = self.sample(train_display_images_a, train_display_images_b)
                    write_2images(image_outputs, display_size, image_directory, 'train_current')

                # Save network weights
                if (iterations + 1) % args.save_epoch_freq == 0 or (iterations+1) >= max_iter:
                    self.save(checkpoint_directory, iterations)

                iterations += 1
                if (iterations) >= max_iter:
                    sys.exit('Finish training')
    def test(self):
        args = self.args
        input_dim = args.input_nc
        if args.direction == 'AtoB':
            a2b=1
            test_type = 'testA'
            output_type = 'fakeB'
        else:
            a2b =0
            test_type = 'testB'
            output_type = 'fakeA'
        # Setup model and data loader
        image_path = os.path.join(self.args.dataroot,test_type)
        image_names = ImageFolder(image_path, transform=None, return_paths=True)
        data_loader = get_data_loader_folder(image_path, 1, False, new_size=args.load_size, crop=False)
        checkpoint = os.path.join(args.result_dir,args.name,'model','gen_%08d.pt' % (args.iteration))
        try:
            state_dict = torch.load(checkpoint)
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
        except:
            state_dict = pytorch03_to_pytorch04(torch.load(checkpoint))
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])

        self.to(self.device)
        #self.eval()
        encode = self.gen_a.encode if a2b else self.gen_b.encode # encode function
        decode = self.gen_b.decode if a2b else self.gen_a.decode # decode function
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])
            images = Variable(images.to(self.device), volatile=True)
            content, _ = encode(images)

            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(args.result_dir,args.name,output_type,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
