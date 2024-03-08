"""
Semantic segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""
import os
import time
from colorama import Fore, Style
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from losses.semantic_loss import CrossEntropyLoss2d, FocalLoss
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import TrainUtils, semantic_color_encoding, IoU
import torch.nn.functional as F
from torch import nn
import mask_gen
from custom_collate import SegCollate

'''
For CutMix.
'''
# cutmix_mask_prop_range = (0.4, 0.75)
# cutmix_boxmask_n_boxes = 3
cutmix_boxmask_fixed_aspect_ratio = False
cutmix_boxmask_by_size = False
cutmix_boxmask_outside_bounds = False
cutmix_boxmask_no_invert = False



def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class SemanticInit(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        semantic_class_weights = dict(
            woodscape_enet=([3.25, 2.33, 20.42, 30.59, 38.4, 45.73, 10.76, 34.16, 44.3, 49.19]),
            woodscape_mfb=(0.04, 0.03, 0.43, 0.99, 2.02, 4.97, 0.17, 1.01, 3.32, 20.35))

        print(f"=> Setting Class weights based on: {args.semantic_class_weighting} \n"
              f"=> {semantic_class_weights[args.semantic_class_weighting]}")

        semantic_class_weights = torch.tensor(semantic_class_weights[args.semantic_class_weighting]).to(args.device)

        # Setup Metrics
        self.metric = IoU(args.semantic_num_classes, args.dataset, ignore_index=None)

        if args.semantic_loss == "cross_entropy":
            self.semantic_criterion = CrossEntropyLoss2d(weight=semantic_class_weights)
        elif args.semantic_loss == "focal_loss":
            self.semantic_criterion = FocalLoss(weight=semantic_class_weights, gamma=2, size_average=True)

        self.best_semantic_iou = 0.0
        self.alpha = 0.5  # to blend semantic predictions with color image
        self.color_encoding = semantic_color_encoding(args)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}", str(self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.args.input_height
                to_save['width'] = self.args.input_width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "adam.pth")
        if self.epoch > 50:  # Optimizer file is quite large! Sometimes, life is a compromise.
            torch.save(self.optimizer_l.state_dict(), save_path)


class SemanticSemiSupModelCPSCutMixFixMatch(SemanticInit):
    def __init__(self, args):
        super().__init__(args)
        mask_generator = mask_gen.BoxMaskGenerator(prop_range=(args.cutmix_mask_prop_range_low, args.cutmix_mask_prop_range_high), n_boxes=args.cutmix_boxmask_n_boxes,
                                                   random_aspect_ratio=not cutmix_boxmask_fixed_aspect_ratio,
                                                   prop_by_area=not cutmix_boxmask_by_size, within_bounds=not cutmix_boxmask_outside_bounds,
                                                   invert=not cutmix_boxmask_no_invert)

        add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
            mask_generator
        )
        collate_fn = SegCollate()
        mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

        # --- Init model ---
        self.models["encoder_l"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["semantic_l"] = SemanticDecoder(self.models["encoder_l"].num_ch_enc,
                                                  n_classes=args.semantic_num_classes).to(self.device)

        self.parameters_to_train_l = []
        self.parameters_to_train_l += list(self.models["encoder_l"].parameters())
        self.parameters_to_train_l += list(self.models["semantic_l"].parameters())

        if args.use_multiple_gpu:
            self.models["encoder_l"] = torch.nn.DataParallel(self.models["encoder_l"])
            self.models["semantic_l"] = torch.nn.DataParallel(self.models["semantic_l"])

        self.models["encoder_r"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["semantic_r"] = SemanticDecoder(self.models["encoder_r"].num_ch_enc,
                                                  n_classes=args.semantic_num_classes).to(self.device)
        self.parameters_to_train_r = []
        self.parameters_to_train_r += list(self.models["encoder_r"].parameters())
        self.parameters_to_train_r += list(self.models["semantic_r"].parameters())

        if args.use_multiple_gpu:
            self.models["encoder_r"] = torch.nn.DataParallel(self.models["encoder_r"])
            self.models["semantic_r"] = torch.nn.DataParallel(self.models["semantic_r"])

        print(f"=> Training on the {self.args.dataset.upper()} dataset \n"
              f"=> Training model named: {self.args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {self.args.output_directory} \n"
              f"=> Training is using the cuda device id: {self.args.cuda_visible_devices} \n"
              f"=> Loading {self.args.dataset} training and validation dataset")

        # --- Load Data ---

        # train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
        #                                                unsupervised=False, collate_fn=collate_fn)
        # unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, \
        #                                                                              train_source=config.unsup_source,
        #                                                                              unsupervised=True,
        #                                                                              collate_fn=mask_collate_fn)
        # unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, \
        #                                                                              train_source=config.unsup_source,
        #                                                                              unsupervised=True,
        #                                                                              collate_fn=collate_fn)

        if 'hard_aug' in args:
            hard_aug = args.hard_aug
            print(f"hard_aug: {hard_aug}")
        else:
            hard_aug = False

        train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                            path_file=args.train_file,
                                            is_train=True,
                                            config=args, hard_aug=hard_aug)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=False)

        ulb_train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                            path_file=args.unlabelled_file,
                                            is_train=True,
                                            config=args, hard_aug=hard_aug)

        self.ulb_train_loader_0 = DataLoader(ulb_train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=False,
                                       collate_fn=mask_collate_fn)

        self.ulb_train_loader_1 = DataLoader(ulb_train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn)

        val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        print(f"=> Total number of training examples: {len(train_dataset)} \n"
              f"=> Total number of unlabelled examples: {len(ulb_train_dataset)}\n"
              f"=> Total number of validation examples: {len(val_dataset)}")

        self.num_total_steps = len(train_dataset) // args.batch_size * args.epochs

        self.optimizer_l = torch.optim.Adam(self.parameters_to_train_l, self.args.learning_rate)
        self.lr_scheduler_l = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_l, self.args.scheduler_step_size)

        self.optimizer_r = torch.optim.Adam(self.parameters_to_train_r, self.args.learning_rate)
        self.lr_scheduler_r = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_r, self.args.scheduler_step_size)

        if args.pretrained_weights:
            self.load_model()

        self.save_args()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def log_time(self, batch_idx, duration, losses, data_time, gpu_time):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.args.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print(f"{Fore.GREEN}epoch {self.epoch:>3}{Style.RESET_ALL} "
              f"| batch {batch_idx:>6} "
              f"| current lr {self.optimizer_l.param_groups[0]['lr']:.7f} "
              f"| examples/s: {samples_per_sec:5.1f} "
              f"| {Fore.RED}loss_sup_l: {losses['semantic_loss_l'].cpu().data:.5f}{Style.RESET_ALL} "
              f"| {Fore.RED}loss_sup_r: {losses['semantic_loss_r'].cpu().data:.5f}{Style.RESET_ALL} "
              f"| {Fore.RED}loss_cps: {losses['CPS_loss'].cpu().data:.5f}{Style.RESET_ALL} "
              f"| {Fore.BLUE}time elapsed: {self.sec_to_hm_str(time_sofar)}{Style.RESET_ALL} "
              f"| {Fore.CYAN}time left: {self.sec_to_hm_str(training_time_left)}{Style.RESET_ALL} "
              f"| CPU/GPU time: {data_time:0.1f}s/{gpu_time:0.1f}s")

    def semantic_train(self):
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            dataloader = iter(self.train_loader)
            unsupervised_dataloader_0 = iter(self.ulb_train_loader_0)
            unsupervised_dataloader_1 = iter(self.ulb_train_loader_1)

            for batch_idx, minibatch in enumerate(dataloader):
            # for batch_idx in range(self.args.niters_per_epoch):
                # inputs_unlabelled = next(iter(self.ulb_train_loader))
                current_time = time.time()
                data_loading_time += (current_time - before_op_time)
                before_op_time = current_time

                # minibatch = dataloader.next()
                unsup_minibatch_0 = next(unsupervised_dataloader_0)
                unsup_minibatch_1 = next(unsupervised_dataloader_1)

                self.inputs_to_device(minibatch)
                self.inputs_to_device(unsup_minibatch_0)
                self.inputs_to_device(unsup_minibatch_1)

                imgs = minibatch["color_aug", 0, 0]
                gts = minibatch["semantic_labels", 0, 0]
                unsup_imgs_0 = unsup_minibatch_0["color_aug", 0, 0]
                unsup_imgs_1 = unsup_minibatch_1["color_aug", 0, 0]
                mask_params = unsup_minibatch_0['mask_params']

                # imgs = inputs_labelled["color_aug", 0, 0]
                # unsup_imgs = inputs_unlabelled["color_aug", 0, 0]
                # gts = inputs_labelled["semantic_labels", 0, 0]

                batch_mix_masks = mask_params
                unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
                with torch.no_grad():
                    # Estimate the pseudo-label with branch#1 & supervise branch#2
                    features_sup_l = self.models["encoder_l"](unsup_imgs_0)
                    logits_u0_tea_1 = self.models["semantic_l"](features_sup_l)["semantic", 0]

                    # _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)

                    features_sup_l = self.models["encoder_l"](unsup_imgs_1)
                    logits_u1_tea_1 = self.models["semantic_l"](features_sup_l)["semantic", 0]

                    # _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
                    logits_u0_tea_1 = logits_u0_tea_1.detach()
                    logits_u1_tea_1 = logits_u1_tea_1.detach()

                    # Estimate the pseudo-label with branch#2 & supervise branch#1
                    features_sup_r = self.models["encoder_r"](unsup_imgs_0)
                    logits_u0_tea_2 = self.models["semantic_r"](features_sup_r)["semantic", 0]

                    # _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)

                    features_sup_r = self.models["encoder_r"](unsup_imgs_1)
                    logits_u1_tea_2 = self.models["semantic_r"](features_sup_r)["semantic", 0]
                    # _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)

                    logits_u0_tea_2 = logits_u0_tea_2.detach()
                    logits_u1_tea_2 = logits_u1_tea_2.detach()

                logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
                # _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
                # ps_label_1 = ps_label_1.long()
                logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
                # _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
                # ps_label_2 = ps_label_2.long()

                features_sup_l = self.models["encoder_l"](unsup_imgs_mixed)
                logits_cons_stu_1 = self.models["semantic_l"](features_sup_l)["semantic", 0]

                features_unsup_r = self.models["encoder_r"](unsup_imgs_mixed)
                logits_cons_stu_2 = self.models["semantic_r"](features_unsup_r)["semantic", 0]

                # apply fixmatch loss
                pseudo_label_l = torch.softmax(logits_cons_tea_1, dim=1)
                max_probs_l, max_l = torch.max(pseudo_label_l, dim=1)
                mask_l = max_probs_l.ge(self.args.p_cutoff).float()

                pseudo_label_r = torch.softmax(logits_cons_tea_2, dim=1)
                max_probs_r, max_r = torch.max(pseudo_label_r, dim=1)
                mask_r = max_probs_r.ge(self.args.p_cutoff).float()

                loss_l = ce_loss(logits_cons_stu_1, max_r, reduction='none') * mask_r
                loss_r = ce_loss(logits_cons_stu_2, max_l, reduction='none') * mask_l

                # update historical loss 
                for i in range(features_sup_l.size(0)):
                    dataloader.dataset.update_loss(minibatch['idx'][i], loss_l[i].item())

                cps_loss = loss_l.mean() + loss_r.mean()

                # cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
                cps_loss = cps_loss * self.args.cps_weight

                features_sup_l = self.models["encoder_l"](imgs)
                sup_pred_l = self.models["semantic_l"](features_sup_l)["semantic", 0]

                features_unsup_r = self.models["encoder_r"](imgs)
                sup_pred_r = self.models["semantic_r"](features_unsup_r)["semantic", 0]

                loss_sup = criterion(sup_pred_l, gts)
                loss_sup_r = criterion(sup_pred_r, gts)

                losses = dict()
                losses["semantic_loss_l"] = loss_sup
                losses["semantic_loss_r"] = loss_sup_r
                losses["CPS_loss"] = cps_loss

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer_l.zero_grad()
                self.optimizer_r.zero_grad()

                loss = loss_sup + loss_sup_r + cps_loss
                loss.backward()

                self.optimizer_l.step()
                self.optimizer_r.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses, data_loading_time, gpu_time)
                    self.semantic_statistics("train", minibatch, sup_pred_l, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            val_metrics = self.semantic_val()
            print(self.epoch, "IoU:", val_metrics["mean_iou"])
            if val_metrics["mean_iou"] >= self.best_semantic_iou:
                print(f"=> Saving model weights with mean_iou of {val_metrics['mean_iou']:.3f} "
                      f"at step {self.step} on {self.epoch} epoch.")
                self.best_semantic_iou = val_metrics["mean_iou"]
                self.save_model()
            print(self.best_semantic_iou)

            self.lr_scheduler_l.step()
            self.lr_scheduler_r.step()

            # update mask 
            dataloader.dataset.update_mask()
            

        print("Training complete!")

    @torch.no_grad()
    def semantic_val(self):
        """Validate the semantic model"""
        self.set_eval()
        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            features = self.models["encoder_l"](inputs["color", 0, 0])
            outputs = self.models["semantic_l"](features)
            losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0], inputs["semantic_labels", 0, 0])
            _, predictions = torch.max(outputs["semantic", 0].data, 1)
            self.metric.add(predictions, inputs["semantic_labels", 0, 0])
        outputs["class_iou"], outputs["mean_iou"] = self.metric.value()

        # Compute stats for the tensorboard
        self.semantic_statistics("val", inputs, outputs, losses)
        self.metric.reset()
        del inputs, losses
        self.set_train()

        return outputs

    def semantic_statistics(self, mode, inputs, outputs, losses) -> None:
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)

        if mode == "val":
            writer.add_scalar(f"mean_iou", outputs["mean_iou"], self.step)
            for k, v in outputs["class_iou"].items():
                writer.add_scalar(f"class_iou/{k}", v, self.step)

        writer.add_scalar("learning_rate", self.optimizer_l.param_groups[0]['lr'], self.step)

        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            if self.args.train.startswith("semantic"):
                writer.add_image(f"color/{j}", inputs[("color", 0, 0)][j], self.step)

            # Predictions is one-hot encoded with "num_classes" channels.
            # Convert it to a single int using the indices where the maximum (1) occurs
            try:
                _, predictions = torch.max(outputs["semantic", 0][j].data, 0)
            except:
                _, predictions = torch.max(outputs[j].data, 0)
            predictions_gray = predictions.byte().squeeze().cpu().detach().numpy()
            color_semantic = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = predictions_gray != 0
            color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - self.alpha) +
                                                   self.color_encoding[predictions_gray[not_background]] * self.alpha)
            writer.add_image(f"semantic_pred_0/{j}", color_semantic.transpose(2, 0, 1), self.step)

            labels = inputs["semantic_labels", 0, 0][j].data
            labels_gray = labels.byte().squeeze().cpu().detach().numpy()
            labels_rgb = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = labels_gray != 0
            labels_rgb[not_background, ...] = (labels_rgb[not_background, ...] * (1 - self.alpha) +
                                               self.color_encoding[labels_gray[not_background]] * self.alpha)
            writer.add_image(f"semantic_labels_0/{j}", labels_rgb.transpose(2, 0, 1), self.step)
