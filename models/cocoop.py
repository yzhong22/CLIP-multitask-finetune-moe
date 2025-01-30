import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.clip import CLIPModel
from collections import OrderedDict
from einops import rearrange

from open_clip.hf_model import ClsPooler


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LNDsss
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, ctx_init, ctx_dim, vis_dim):
        super().__init__()
        n_ctx = n_ctx
        ctx_init = ctx_init
        ctx_dim = ctx_dim
        vis_dim = vis_dim

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        self.n_ctx = n_ctx

    def init_prompts(self, texts_multitasks, tokenizer, word_embeddings, context_length=256):

        self.n_tasks = len(texts_multitasks)

        n_classes = []
        for i in range(self.n_tasks):
            texts_multitasks[i] = [self.prompt_prefix + " " + text for text in texts_multitasks[i]]
            n_classes.append(len(texts_multitasks[i]))
        self.n_classes = n_classes
        self.texts_multitasks = texts_multitasks

        device = next(self.parameters()).device

        print(f"Text contexxt: '{texts_multitasks}'")

        with torch.no_grad():
            texts_tokenized = [tokenizer(texts, context_length).to(device) for texts in texts_multitasks]
            self.texts_tokenized = texts_tokenized
            embeddings = [word_embeddings(texts) for texts in texts_tokenized]

            for i in range(self.n_tasks):
                self.register_buffer(f"token_prefix_{i}", embeddings[i][:, :1, :])  # SOS
                self.register_buffer(f"token_suffix_{i}", embeddings[i][:, 1 + self.n_ctx :, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts_multitask = []
        for i in range(self.n_tasks):
            prefix = getattr(self, f"token_prefix_{i}")
            suffix = getattr(self, f"token_suffix_{i}")
            prompts = []
            for ctx_shifted_i in ctx_shifted:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_classes[i], -1, -1)
                pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
                prompts.append(pts_i)
            prompts_multitask.append(torch.stack(prompts))

        return prompts_multitask


class CoCoOpModel(CLIPModel):
    def __init__(self, backbone: Backbone, args) -> None:
        super().__init__(
            backbone,
        )

        self.prompt_learner = PromptLearner(
            n_ctx=args.cocoop_n_ctx, ctx_init=args.cocoop_ctx_init, ctx_dim=768, vis_dim=512
        )

    def init_text_features(self, texts_multitasks):
        training = self.training
        self.eval()

        self.prompt_learner.init_prompts(
            texts_multitasks,
            tokenizer=self.backbone.tokenizer,
            word_embeddings=self.backbone.model.text.transformer.get_input_embeddings(),
            context_length=self.backbone.context_length,
        )

        self.train(training)

    def encode_text(self, x, x_tokenized):
        # x shape: n, n ctx, ctx dim
        attn_mask = (x_tokenized != self.backbone.model.text.config.pad_token_id).long()
        out = self.backbone.model.text.transformer(inputs_embeds=x, attention_mask=attn_mask)
        pooled_out = self.backbone.model.text.pooler(out, attn_mask)
        projected = self.backbone.model.text.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.backbone.model.text.pooler.cls_token_position, :]
            if type(self.backbone.model.text.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if self.backbone.model.text.output_tokens:
            return projected, tokens
        return projected

    def forward(self, image, text_features=None, is_multi_label=False):
        bs = image.shape[0]
        with torch.no_grad():
            output = self.encode_image(image)

        if is_multi_label:
            prompts = self.prompt_learner(F.normalize(output["image_feature"], dim=-1))

            n_tasks = len(prompts)
            n_classes = 2

            logits = []
            for i_bs in range(bs):
                text_features = []

                texts_tokenized = torch.stack(self.prompt_learner.texts_tokenized, dim=0)  # n_task, 2, C
                prompts_ibs = torch.stack([x[i_bs] for x in prompts], dim=0)  # n_task, 2, n_ctx, ctx_dim

                texts_tokenized = rearrange(texts_tokenized, "n d c -> (n d) c", d=n_classes)
                prompts_ibs = rearrange(prompts_ibs, "n d nc c -> (n d) nc c", d=n_classes)

                text_features = self.encode_text(prompts_ibs, texts_tokenized)
                text_features = rearrange(text_features, "(n d) c -> n d c", n=n_tasks, d=n_classes)

                # for i in range(n_tasks):
                #     texts_tokenized = self.prompt_learner.texts_tokenized[i]
                #     text_features.append(self.encode_text(prompts[i][i_bs], texts_tokenized))
                # text_features = torch.stack(text_features, dim=0)

                logits.append(
                    self.backbone.logit_scale
                    * F.normalize(output["image_feature"][i_bs], dim=-1)
                    @ F.normalize(text_features, dim=-1).transpose(1, 2)
                )
            logits = torch.stack(logits, dim=0)
            output["logits"] = logits

        else:
            raise NotImplementedError

        return output

    def load_state_dict(self, state_dict, strict=True):
        """
        Override the load_state_dict method to skip keys containing 'token_prefix',
        even when strict=True.

        Args:
            state_dict (dict): The state dictionary to load.
            strict (bool): Whether to strictly enforce key matching.

        Returns:
            NamedTuple with missing_keys and unexpected_keys.
        """
        # Filter out keys containing 'token_prefix' from the state_dict
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if ("token_prefix" not in k) and ("token_suffix" not in k)
        }

        # Identify the skipped keys
        skipped_keys = [k for k in state_dict.keys() if ("token_prefix" in k) or ("token_suffix" in k)]
        if skipped_keys:
            print(f"Skipping keys: {skipped_keys}")

        # Call the parent class's load_state_dict with the filtered dictionary
        result = super().load_state_dict(filtered_state_dict, strict=False)  # Use strict=False here

        if strict:
            # Handle strict mode: Check for missing and unexpected keys
            missing_keys = [k for k in result.missing_keys if ("token_prefix" not in k) and ("token_suffix" not in k)]
            unexpected_keys = result.unexpected_keys

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Error(s) in loading state_dict: Missing keys: {missing_keys}, "
                    f"Unexpected keys: {unexpected_keys}"
                )

        return result
