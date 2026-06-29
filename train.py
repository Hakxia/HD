import math
import os
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from model.ctamp import ContinuousTAMP, masked_completion_energy
from model.masking import build_balanced_mask_bank, sample_hidden_mask
from utils import TrainReport


def build_cosine_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive.")
    num_warmup_steps = max(0, int(num_warmup_steps))

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_training_feature_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(dataset, "features"):
        raise ValueError("Training dataset must expose a read-only features property.")
    features = torch.as_tensor(dataset.features, dtype=torch.float32)
    if features.ndim == 3 and features.shape[1] == 1:
        features = features[:, 0, :]
    elif features.ndim != 2:
        raise ValueError(f"Expected training features [N, 1, D] or [N, D], got {tuple(features.shape)}.")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training feature statistics contain NaN or Inf.")
    return mean, std


def build_hard_negative_pool(
    motion_embeddings: torch.Tensor,
    seen_labels: np.ndarray | torch.Tensor,
    pool_size: int,
) -> dict[int, torch.Tensor]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    seen = torch.as_tensor(seen_labels, dtype=torch.long, device="cpu")
    if seen.numel() < 2:
        raise ValueError("At least two seen labels are required for negative sampling.")
    motion = F.normalize(motion_embeddings.float().cpu(), dim=-1)
    seen_motion = motion.index_select(0, seen)
    similarity = seen_motion @ seen_motion.T
    similarity.fill_diagonal_(-float("inf"))
    topk = min(pool_size, seen.numel() - 1)

    pool: dict[int, torch.Tensor] = {}
    for row, label in enumerate(seen.tolist()):
        indices = similarity[row].topk(topk).indices
        pool[int(label)] = seen.index_select(0, indices).clone()
    return pool


def sample_candidate_labels(
    labels: torch.Tensor,
    hard_negative_pool: dict[int, torch.Tensor],
    seen_labels: torch.Tensor,
    num_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_negatives < 1:
        raise ValueError("num_negatives must be at least 1.")
    labels = labels.long()
    device = labels.device
    seen_labels = seen_labels.to(device=device, dtype=torch.long)
    if seen_labels.numel() <= num_negatives:
        raise ValueError("Not enough seen labels to sample the requested number of negatives.")

    candidates = torch.empty(labels.shape[0], 1 + num_negatives, device=device, dtype=torch.long)
    candidates[:, 0] = labels

    for row, label_tensor in enumerate(labels):
        label = int(label_tensor.item())
        if label not in hard_negative_pool:
            raise ValueError(f"Label {label} is not present in the hard negative pool.")

        selected: list[int] = []
        pool = hard_negative_pool[label].to(device=device, dtype=torch.long)
        pool = pool[pool != label]
        if pool.numel() > 0:
            perm = torch.randperm(pool.numel(), device=device, generator=generator)
            for value in pool.index_select(0, perm).tolist():
                if value != label and value not in selected:
                    selected.append(int(value))
                if len(selected) == num_negatives:
                    break

        if len(selected) < num_negatives:
            fallback = seen_labels[seen_labels != label]
            if selected:
                selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
                fallback = fallback[~torch.isin(fallback, selected_tensor)]
            perm = torch.randperm(fallback.numel(), device=device, generator=generator)
            for value in fallback.index_select(0, perm).tolist():
                if value != label and value not in selected:
                    selected.append(int(value))
                if len(selected) == num_negatives:
                    break

        if len(selected) != num_negatives:
            raise RuntimeError(f"Could not sample {num_negatives} negatives for label {label}.")
        candidates[row, 1:] = torch.tensor(selected, device=device, dtype=torch.long)

    return candidates


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.train_data_loader = data_loader.get("train")
        self.test_data_loader = data_loader["test"]

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.unseen_labels = np.load(args.unseen_label_path).astype(np.int64)
        if len(self.unseen_labels) != args.unseen_label:
            raise ValueError("unseen_label does not match unseen_label_path length.")
        if np.any(self.unseen_labels < 0) or np.any(self.unseen_labels >= args.num_classes):
            raise ValueError("Unseen labels must be in [0, num_classes).")
        all_labels = np.arange(args.num_classes, dtype=np.int64)
        self.seen_labels = all_labels[~np.isin(all_labels, self.unseen_labels)]
        self.seen_labels_tensor_cpu = torch.as_tensor(self.seen_labels, dtype=torch.long)
        self.unseen_labels_tensor = torch.as_tensor(self.unseen_labels, dtype=torch.long, device=self.accelerator.device)
        lookup = torch.full((args.num_classes,), -1, dtype=torch.long, device=self.accelerator.device)
        lookup[self.unseen_labels_tensor] = torch.arange(len(self.unseen_labels), device=self.accelerator.device)
        self.unseen_local_lookup = lookup

        if self.train_data_loader is None:
            raise ValueError("A training dataloader is required to compute seen-class feature statistics.")
        feature_mean, feature_std = compute_training_feature_stats(self.train_data_loader.dataset)

        self.tokenizer = CLIPTokenizer.from_pretrained(args.text_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.text_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.class_names, self.class_descriptions = self.load_class_texts()
        name_embeddings, motion_embeddings = self.encode_all_class_texts()
        self.text_dim = int(name_embeddings.shape[-1])
        self.name_embeddings = name_embeddings.to(self.accelerator.device)
        self.motion_embeddings = motion_embeddings.to(self.accelerator.device)
        self.hard_negative_pool = build_hard_negative_pool(
            motion_embeddings=motion_embeddings,
            seen_labels=self.seen_labels_tensor_cpu,
            pool_size=args.hard_negative_pool_size,
        )

        del self.text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = ContinuousTAMP(
            feature_dim=args.feature_dim,
            num_groups=args.num_groups,
            text_dim=self.text_dim,
            model_dim=args.model_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            permutation_seed=args.permutation_seed,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        if args.optimizer != "AdamW":
            raise ValueError("C-TAMP currently supports optimizer: AdamW.")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.lr_scheduler != "cosine":
            raise ValueError("C-TAMP currently supports lr_scheduler: cosine.")
        self.lr_scheduler = build_cosine_scheduler(self.optimizer, args.num_warmup, args.num_iter)

        prepare_items = [self.model, self.optimizer, self.lr_scheduler, self.test_data_loader]
        if self.train_data_loader is not None:
            prepare_items.append(self.train_data_loader)
        prepared = self.accelerator.prepare(*prepare_items)
        self.model, self.optimizer, self.lr_scheduler, self.test_data_loader = prepared[:4]
        if self.train_data_loader is not None:
            self.train_data_loader = prepared[4]

        self.test_mask_bank = build_balanced_mask_bank(
            num_masks=args.num_test_masks,
            num_groups=args.num_groups,
            mask_ratio=args.test_mask_ratio,
            seed=args.test_mask_seed,
        ).to(self.accelerator.device)

        self.mask_ratios = [float(v) for v in args.mask_ratios]
        self.mask_generator = torch.Generator(device=self.accelerator.device)
        self.mask_generator.manual_seed(int(args.seed) + 1001)
        self.candidate_generator = torch.Generator(device=self.accelerator.device)
        self.candidate_generator.manual_seed(int(args.seed) + 2002)
        self.last_mask_ratio = self.mask_ratios[0]

    @property
    def module(self) -> ContinuousTAMP:
        return self.accelerator.unwrap_model(self.model)

    def load_class_texts(self) -> tuple[list[str], list[str]]:
        df = pd.read_csv(self.args.class_name_path)
        if "label" not in df.columns:
            raise ValueError(f"{self.args.class_name_path} must contain a label column.")
        csv_names = [str(value) for value in df["label"].tolist()]
        with open(self.args.class_description_path, "r", encoding="utf-8") as file:
            descriptions = [line.strip() for line in file.readlines()]
        if len(csv_names) != self.args.num_classes:
            raise ValueError("class_name_path row count must equal num_classes.")
        if len(descriptions) != self.args.num_classes:
            raise ValueError("class_description_path line count must equal num_classes.")
        names = self.build_name_prompts(csv_names, descriptions)
        descriptions = self.build_description_prompts(csv_names, descriptions)
        return names, descriptions

    def build_name_prompts(self, csv_names: list[str], descriptions: list[str]) -> list[str]:
        mode = getattr(self.args, "class_name_mode", "csv_label")
        if mode == "csv_label":
            return csv_names

        quoted_names = []
        for description in descriptions:
            match = re.match(r'^[\"“](.*?)[\"”]', description)
            if match:
                quoted_names.append(match.group(1))
            else:
                quoted_names.append(description.split(" is ", 1)[0].strip().strip('"“”'))

        if mode == "quoted_description_name":
            return quoted_names
        if mode == "quoted_name_plus_csv_label":
            return [f"{quoted}: {csv}" for quoted, csv in zip(quoted_names, csv_names)]
        raise ValueError(f"Unsupported class_name_mode: {mode}")

    def build_description_prompts(self, csv_names: list[str], descriptions: list[str]) -> list[str]:
        mode = getattr(self.args, "class_description_mode", "raw")
        if mode == "raw":
            return descriptions
        if mode == "append_csv_label":
            return [f"{description} Skeleton motion: {csv}" for description, csv in zip(descriptions, csv_names)]
        if mode == "prepend_csv_label":
            return [f"Skeleton motion: {csv}. {description}" for description, csv in zip(descriptions, csv_names)]
        raise ValueError(f"Unsupported class_description_mode: {mode}")

    @torch.no_grad()
    def encode_all_class_texts(self) -> tuple[torch.Tensor, torch.Tensor]:
        def encode(prompts: list[str]) -> torch.Tensor:
            outputs = []
            for batch in _batched(prompts, self.args.text_encode_batch_size):
                text_inputs = self.tokenizer(
                    batch,
                    padding="max_length",
                    max_length=self.args.text_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.accelerator.device)
                encoded = self.text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                )
                pooled = getattr(encoded, "pooler_output", None)
                if pooled is None:
                    eos_positions = text_inputs.input_ids.argmax(dim=-1)
                    pooled = encoded.last_hidden_state[torch.arange(len(batch), device=self.accelerator.device), eos_positions]
                outputs.append(pooled.detach().float().cpu())
            return torch.cat(outputs, dim=0)

        return encode(self.class_names), encode(self.class_descriptions)

    def _mean_metric(self, value: torch.Tensor) -> float:
        gathered = self.accelerator.gather_for_metrics(value.detach().float().reshape(1))
        return float(gathered.mean().item())

    def save_best_model(self):
        save_path = os.path.join(self.args.work_dir, "best")
        self.accelerator.save_state(save_path)
        if self.accelerator.is_main_process:
            torch.save(
                {
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "args": vars(self.args),
                },
                os.path.join(save_path, "model_state.pt"),
            )

    def save_checkpoint(self, global_step: int):
        save_path = os.path.join(self.args.work_dir, f"checkpoint-{global_step}")
        self.accelerator.save_state(save_path)
        if self.accelerator.is_main_process:
            torch.save(
                {
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "args": vars(self.args),
                },
                os.path.join(save_path, "model_state.pt"),
            )

    def load_state(self, checkpoint_path: str):
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for test phase.")
        self.accelerator.load_state(checkpoint_path)

    def train(self, train_log, global_step: int) -> int:
        self.model.train()
        report = TrainReport()
        start = time.time()

        progress = tqdm(enumerate(self.train_data_loader), disable=not self.accelerator.is_main_process)
        for _idx, (features, labels) in progress:
            if global_step >= self.args.num_iter:
                break
            with self.accelerator.accumulate(self.model):
                features = features.to(self.accelerator.device, dtype=torch.float32)
                labels = labels.to(self.accelerator.device, dtype=torch.long)
                batch_size = features.shape[0]

                ratio_idx = torch.randint(
                    len(self.mask_ratios),
                    (1,),
                    device=self.accelerator.device,
                    generator=self.mask_generator,
                ).item()
                mask_ratio = self.mask_ratios[ratio_idx]
                self.last_mask_ratio = mask_ratio
                hidden_mask = sample_hidden_mask(
                    batch_size=batch_size,
                    num_groups=self.args.num_groups,
                    mask_ratio=mask_ratio,
                    device=self.accelerator.device,
                    generator=self.mask_generator,
                )
                hidden_counts = hidden_mask.sum(dim=1)
                if not torch.all(hidden_counts == hidden_counts[0]):
                    raise RuntimeError("Every training sample must hide the same number of groups.")
                if hidden_counts[0] <= 0 or hidden_counts[0] >= self.args.num_groups:
                    raise RuntimeError("Training mask must keep at least one hidden and one visible group.")

                candidate_labels = sample_candidate_labels(
                    labels=labels,
                    hard_negative_pool=self.hard_negative_pool,
                    seen_labels=self.seen_labels_tensor_cpu.to(self.accelerator.device),
                    num_negatives=self.args.num_hard_negatives,
                    generator=self.candidate_generator,
                )
                num_candidates = candidate_labels.shape[1]

                base_groups = self.module.encode_feature_groups(features)
                groups_expanded = (
                    base_groups.unsqueeze(1)
                    .expand(batch_size, num_candidates, self.args.num_groups, self.module.group_dim)
                    .reshape(batch_size * num_candidates, self.args.num_groups, self.module.group_dim)
                )
                mask_expanded = (
                    hidden_mask.unsqueeze(1)
                    .expand(batch_size, num_candidates, self.args.num_groups)
                    .reshape(batch_size * num_candidates, self.args.num_groups)
                )
                flat_labels = candidate_labels.reshape(-1)
                name_text = self.name_embeddings.index_select(0, flat_labels)
                motion_text = self.motion_embeddings.index_select(0, flat_labels)

                prediction = self.module.predict_groups(groups_expanded, mask_expanded, name_text, motion_text)
                energy = masked_completion_energy(
                    prediction=prediction,
                    target=groups_expanded,
                    hidden_mask=mask_expanded,
                    beta=self.args.huber_beta,
                ).reshape(batch_size, num_candidates)

                completion_loss = energy[:, 0].mean()
                logits = -energy / self.args.energy_temperature
                targets = torch.zeros(batch_size, dtype=torch.long, device=self.accelerator.device)
                energy_loss = F.cross_entropy(logits, targets)
                total_loss = self.args.completion_weight * completion_loss + self.args.energy_weight * energy_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Training loss is NaN or Inf.")

                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                positive_energy = energy[:, 0].mean()
                negative_energy = energy[:, 1:].mean()
                energy_gap = negative_energy - positive_energy
                candidate_accuracy = (energy.argmin(dim=1) == 0).float().mean()

                metric_total_loss = self._mean_metric(total_loss)
                metric_completion_loss = self._mean_metric(completion_loss)
                metric_energy_loss = self._mean_metric(energy_loss)
                metric_positive_energy = self._mean_metric(positive_energy)
                metric_negative_energy = self._mean_metric(negative_energy)
                metric_energy_gap = self._mean_metric(energy_gap)
                metric_candidate_accuracy = self._mean_metric(candidate_accuracy)

                if self.accelerator.is_main_process:
                    report.update(
                        batch_size=batch_size,
                        total_loss=metric_total_loss,
                        completion_loss=metric_completion_loss,
                        energy_loss=metric_energy_loss,
                        positive_energy=metric_positive_energy,
                        negative_energy=metric_negative_energy,
                        energy_gap=metric_energy_gap,
                        candidate_accuracy=metric_candidate_accuracy,
                    )

            global_step += 1
            if global_step % self.args.log_iter == 0 or global_step >= self.args.num_iter:
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                period_time = time.time() - start
                train_log.write(
                    f"Iter[{global_step}/{self.args.num_iter}]\t"
                    + report.result_str(lr, period_time, self.last_mask_ratio)
                )
                start = time.time()
                report = TrainReport()

            if global_step % self.args.save_iter == 0:
                self.save_checkpoint(global_step)

        return global_step

    @torch.no_grad()
    def test(self) -> float:
        self.model.eval()
        correct = torch.zeros(1, device=self.accelerator.device, dtype=torch.long)
        total = torch.zeros(1, device=self.accelerator.device, dtype=torch.long)
        unseen_labels = self.unseen_labels_tensor

        for features, labels in tqdm(self.test_data_loader, disable=not self.accelerator.is_main_process):
            features = features.to(self.accelerator.device, dtype=torch.float32)
            labels = labels.to(self.accelerator.device, dtype=torch.long)
            batch_size = features.shape[0]
            base_groups = self.module.encode_feature_groups(features)
            energy_sum = torch.zeros(batch_size, len(unseen_labels), device=self.accelerator.device, dtype=torch.float32)

            for mask in self.test_mask_bank:
                hidden_mask = mask.unsqueeze(0).expand(batch_size, self.args.num_groups)
                for start in range(0, len(unseen_labels), self.args.class_chunk_size):
                    end = min(start + self.args.class_chunk_size, len(unseen_labels))
                    class_chunk = unseen_labels[start:end]
                    chunk_size = class_chunk.numel()
                    groups_expanded = (
                        base_groups.unsqueeze(1)
                        .expand(batch_size, chunk_size, self.args.num_groups, self.module.group_dim)
                        .reshape(batch_size * chunk_size, self.args.num_groups, self.module.group_dim)
                    )
                    mask_expanded = (
                        hidden_mask.unsqueeze(1)
                        .expand(batch_size, chunk_size, self.args.num_groups)
                        .reshape(batch_size * chunk_size, self.args.num_groups)
                    )
                    flat_labels = class_chunk.unsqueeze(0).expand(batch_size, chunk_size).reshape(-1)
                    name_text = self.name_embeddings.index_select(0, flat_labels)
                    motion_text = self.motion_embeddings.index_select(0, flat_labels)
                    prediction = self.module.predict_groups(groups_expanded, mask_expanded, name_text, motion_text)
                    energy = masked_completion_energy(
                        prediction=prediction,
                        target=groups_expanded,
                        hidden_mask=mask_expanded,
                        beta=self.args.huber_beta,
                    ).reshape(batch_size, chunk_size)
                    energy_sum[:, start:end] += energy.float()

            energy_mean = energy_sum / float(self.args.num_test_masks)
            pred_local = energy_mean.argmin(dim=1)
            target_local = self.unseen_local_lookup.index_select(0, labels)
            if (target_local < 0).any():
                raise ValueError("Test labels must belong to unseen labels.")
            correct += (pred_local == target_local).sum()
            total += batch_size

        correct_all = self.accelerator.gather_for_metrics(correct).sum()
        total_all = self.accelerator.gather_for_metrics(total).sum().clamp_min(1)
        return float((correct_all.float() / total_all.float()).item())

    @torch.no_grad()
    def diagnose_text_usage(self, max_batches: int = 1) -> dict[str, float]:
        self.model.eval()
        positive_values = []
        shuffled_values = []
        for batch_idx, (features, labels) in enumerate(self.test_data_loader):
            if batch_idx >= max_batches:
                break
            features = features.to(self.accelerator.device, dtype=torch.float32)
            labels = labels.to(self.accelerator.device, dtype=torch.long)
            batch_size = features.shape[0]
            groups = self.module.encode_feature_groups(features)
            hidden_mask = self.test_mask_bank[0].unsqueeze(0).expand(batch_size, self.args.num_groups)
            shuffled = labels[torch.randperm(batch_size, device=self.accelerator.device)]
            pred_pos = self.module.predict_groups(
                groups,
                hidden_mask,
                self.name_embeddings.index_select(0, labels),
                self.motion_embeddings.index_select(0, labels),
            )
            pred_shuf = self.module.predict_groups(
                groups,
                hidden_mask,
                self.name_embeddings.index_select(0, shuffled),
                self.motion_embeddings.index_select(0, shuffled),
            )
            positive_values.append(masked_completion_energy(pred_pos, groups, hidden_mask, self.args.huber_beta))
            shuffled_values.append(masked_completion_energy(pred_shuf, groups, hidden_mask, self.args.huber_beta))

        positive = torch.cat(positive_values).mean()
        shuffled = torch.cat(shuffled_values).mean()
        return {
            "positive_energy": self._mean_metric(positive),
            "shuffled_energy": self._mean_metric(shuffled),
            "gap": self._mean_metric(shuffled - positive),
        }
