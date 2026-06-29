import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTAMP(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        num_groups: int = 16,
        text_dim: int = 1024,
        model_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        permutation_seed: int = 2025,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        if feature_dim % num_groups != 0:
            raise ValueError("feature_dim must be divisible by num_groups.")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_dim = feature_dim // num_groups
        self.model_dim = model_dim

        mean = torch.zeros(feature_dim, dtype=torch.float32) if feature_mean is None else feature_mean.float()
        std = torch.ones(feature_dim, dtype=torch.float32) if feature_std is None else feature_std.float()
        if mean.shape != (feature_dim,):
            raise ValueError(f"feature_mean must have shape [{feature_dim}].")
        if std.shape != (feature_dim,):
            raise ValueError(f"feature_std must have shape [{feature_dim}].")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone().clamp_min(1e-6))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(permutation_seed))
        permutation = torch.randperm(feature_dim, generator=generator)
        self.register_buffer("permutation", permutation, persistent=True)

        self.group_projection = nn.Linear(self.group_dim, model_dim)
        self.group_position_embedding = nn.Parameter(torch.zeros(1, num_groups, model_dim))
        self.mask_token = nn.Parameter(torch.zeros(model_dim))

        self.name_projection = nn.Linear(text_dim, model_dim)
        self.motion_projection = nn.Linear(text_dim, model_dim)
        self.name_type_embedding = nn.Parameter(torch.zeros(model_dim))
        self.motion_type_embedding = nn.Parameter(torch.zeros(model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_head = nn.Linear(model_dim, self.group_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.group_position_embedding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.name_type_embedding, std=0.02)
        nn.init.normal_(self.motion_type_embedding, std=0.02)

    def canonicalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            if features.shape[1] != self.feature_dim:
                raise ValueError(f"Expected features with shape [B, {self.feature_dim}], got {tuple(features.shape)}.")
            return features
        if features.ndim == 3:
            if features.shape[1] != 1 or features.shape[2] != self.feature_dim:
                raise ValueError(
                    f"Expected features with shape [B, 1, {self.feature_dim}], got {tuple(features.shape)}."
                )
            return features[:, 0, :]
        raise ValueError(f"Expected features with 2 or 3 dimensions, got {features.ndim}.")

    def encode_feature_groups(self, features: torch.Tensor) -> torch.Tensor:
        features = self.canonicalize_features(features).float()
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        permuted = normalized.index_select(dim=-1, index=self.permutation)
        return permuted.reshape(features.shape[0], self.num_groups, self.group_dim)

    def build_group_tokens(
        self,
        groups: torch.Tensor,
        hidden_mask: torch.Tensor,
    ) -> torch.Tensor:
        if groups.shape[-2:] != (self.num_groups, self.group_dim):
            raise ValueError(
                f"groups must have shape [N, {self.num_groups}, {self.group_dim}], got {tuple(groups.shape)}."
            )
        if hidden_mask.shape != groups.shape[:2]:
            raise ValueError(f"hidden_mask shape {tuple(hidden_mask.shape)} does not match groups {tuple(groups.shape)}.")
        if hidden_mask.dtype != torch.bool:
            raise ValueError("hidden_mask must be a bool tensor.")

        visible_tokens = self.group_projection(groups)
        mask_tokens = self.mask_token.view(1, 1, self.model_dim).expand_as(visible_tokens)
        tokens = torch.where(hidden_mask.unsqueeze(-1), mask_tokens, visible_tokens)
        return tokens + self.group_position_embedding

    def predict_groups(
        self,
        groups: torch.Tensor,
        hidden_mask: torch.Tensor,
        name_text_embedding: torch.Tensor,
        motion_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if name_text_embedding.ndim != 2 or motion_text_embedding.ndim != 2:
            raise ValueError("Text embeddings must have shape [N, text_dim].")
        if name_text_embedding.shape[0] != groups.shape[0] or motion_text_embedding.shape[0] != groups.shape[0]:
            raise ValueError("Text embedding batch size must match groups.")

        group_tokens = self.build_group_tokens(groups, hidden_mask)
        name_token = self.name_projection(name_text_embedding.float()) + self.name_type_embedding
        motion_token = self.motion_projection(motion_text_embedding.float()) + self.motion_type_embedding
        text_tokens = torch.stack((name_token, motion_token), dim=1)
        tokens = torch.cat((group_tokens, text_tokens), dim=1)

        encoded = self.transformer_encoder(tokens)
        group_output = self.final_norm(encoded[:, : self.num_groups, :])
        return self.output_head(group_output)

    def forward(
        self,
        features: torch.Tensor,
        hidden_mask: torch.Tensor,
        name_text_embedding: torch.Tensor,
        motion_text_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_groups = self.encode_feature_groups(features)
        prediction_groups = self.predict_groups(
            target_groups,
            hidden_mask,
            name_text_embedding,
            motion_text_embedding,
        )
        return prediction_groups, target_groups


def masked_completion_energy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hidden_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must equal target shape {tuple(target.shape)}.")
    if prediction.ndim != 3:
        raise ValueError("prediction and target must have shape [N, G, group_dim].")
    if hidden_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"hidden_mask shape {tuple(hidden_mask.shape)} must match prediction prefix {tuple(prediction.shape[:2])}."
        )
    if hidden_mask.dtype != torch.bool:
        raise ValueError("hidden_mask must be a bool tensor.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    loss = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none", beta=beta)
    mask = hidden_mask.unsqueeze(-1).to(dtype=loss.dtype)
    denom = (hidden_mask.sum(dim=1).to(dtype=loss.dtype) * prediction.shape[-1]).clamp_min(1.0)
    return (loss * mask).sum(dim=(1, 2)) / denom
