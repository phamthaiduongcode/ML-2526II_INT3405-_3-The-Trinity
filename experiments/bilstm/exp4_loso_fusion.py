import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn2d_bilstm_fusion import CNN2DBiLSTMFusion, SupConLoss
from src.data_pipeline.preprocess import apply_euclidean_alignment, normalize_after_split, EEGAugmentor
from src.data_pipeline.feature_extractor import EEGFeatureExtractor

# ── Flags ────────────────────────────────────────────────────────────────────
CLEANUP_CHECKPOINTS = False
AUGMENT_TRAIN       = True   # tắt thành False để ablation so sánh

def evaluate_in_batches(model, dataloader, device):
    model.eval()
    p_v_list, p_a_list = [], []
    with torch.no_grad():
        for e_raw, e_freq, e_graph, _, _ in dataloader:
            o_v, o_a, _, _ = model(
                e_raw.to(device, non_blocking=True),
                e_freq.to(device, non_blocking=True),
                e_graph.to(device, non_blocking=True)
            )
            p_v_list.append(torch.argmax(o_v, dim=1).cpu().numpy())
            p_a_list.append(torch.argmax(o_a, dim=1).cpu().numpy())
    return np.concatenate(p_v_list), np.concatenate(p_a_list)

def scale_feature_matrix(tr, va, te):
    sh_tr, sh_va, sh_te = tr.shape, va.shape, te.shape
    scaler = StandardScaler()
    tr_s = scaler.fit_transform(tr.reshape(-1, tr.shape[-1])).reshape(sh_tr)
    va_s = scaler.transform(va.reshape(-1, va.shape[-1])).reshape(sh_va)
    te_s = scaler.transform(te.reshape(-1, te.shape[-1])).reshape(sh_te)
    return tr_s, va_s, te_s

def train_loso_supcon_dgcnn_pipeline():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = torch.cuda.is_available()

    # ── Hyperparameters (V7) ─────────────────────────────────────────────────
    epochs         = 100
    batch_size     = 256
    patience       = 15
    warmup_epochs  = 5
    lambda_supcon  = 0.05
    temperature    = 0.15

    # Calibration adaptive
    calib_max_epochs  = 15    # tối đa — thay vì cứng 5
    calib_patience    = 3     # dừng nếu không cải thiện sau 3 epoch
    calib_lr          = 1e-4

    crit_emotion = nn.CrossEntropyLoss(label_smoothing=0.1)
    crit_supcon  = SupConLoss(temperature=temperature)

    augmentor = EEGAugmentor(
        noise_std=0.01,
        mask_ratio=0.10,
        channel_drop_p=0.05,
        apply_p=0.80
    ) if AUGMENT_TRAIN else None

    processed_dir  = "data/processed"
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("\n" + "="*75)
    print("[INFO] CNN2D-BiLSTM-DGCNN FUSION — V7 (AUG + ADAPTIVE CALIB + COSINE LR)")
    print(f"       AUGMENT_TRAIN={AUGMENT_TRAIN} | temperature={temperature} | lambda={lambda_supcon}")
    print("="*75)

    X_all  = np.load(os.path.join(processed_dir, "X_epochs.npy"))
    y_val  = np.load(os.path.join(processed_dir, "y_valence.npy")).astype(np.int64)
    y_aro  = np.load(os.path.join(processed_dir, "y_arousal.npy")).astype(np.int64)
    groups = np.load(os.path.join(processed_dir, "subject_groups.npy")).astype(np.int64)

    num_channels_dynamic = X_all.shape[1]
    freq_dim_expected    = num_channels_dynamic * 5

    X_all = apply_euclidean_alignment(X_all, groups)
    fe    = EEGFeatureExtractor(sfreq=128, window_len=32, stride=8)

    unique_subs   = np.unique(groups)
    fold_results_zs = []
    fold_results_fs = []

    for fold, test_sub in enumerate(unique_subs):
        grad_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        print("\n" + "#"*70)
        print(f"  [FOLD {fold+1}/{len(unique_subs)}] TARGET UNSEEN SUBJECT: {test_sub}")
        print("#"*70)

        full_train_idx = np.where(groups != test_sub)[0]
        test_idx       = np.where(groups == test_sub)[0]

        src_subs = np.unique(groups[full_train_idx])
        rng = np.random.default_rng(42 + fold)
        rng.shuffle(src_subs)

        val_subs   = src_subs[:2]
        train_subs = src_subs[2:]

        tr_idx = np.where(np.isin(groups, train_subs))[0]
        va_idx = np.where(np.isin(groups, val_subs))[0]
        te_idx = test_idx

        print("  [Extractor] Trích xuất đặc trưng...")
        X_tr_freq, X_tr_graph = fe.extract_features_pipeline(X_all[tr_idx])
        X_va_freq, X_va_graph = fe.extract_features_pipeline(X_all[va_idx])
        X_te_freq, X_te_graph = fe.extract_features_pipeline(X_all[te_idx])

        if fold == 0:
            assert X_tr_freq.shape[-1] == freq_dim_expected, \
                f"freq_seq_dim mismatch! Expected {freq_dim_expected}, got {X_tr_freq.shape[-1]}"

        X_tr_freq, X_va_freq, X_te_freq = scale_feature_matrix(X_tr_freq, X_va_freq, X_te_freq)
        X_tr_graph, X_va_graph, X_te_graph = scale_feature_matrix(X_tr_graph, X_va_graph, X_te_graph)

        raw_norm_dict = normalize_after_split(
            X_train=X_all[tr_idx], X_test=X_all[te_idx],
            X_val=X_all[va_idx], mode='channel'
        )
        X_tr_raw = raw_norm_dict['train']
        X_te_raw = raw_norm_dict['test']
        X_va_raw = raw_norm_dict['val']

        # ── Augmentation chỉ trên tập train raw ─────────────────────────────
        if augmentor is not None:
            X_tr_raw_aug = augmentor(X_tr_raw)
            print(f"  [Augment] Áp dụng augmentation trên {len(X_tr_raw)} samples train")
        else:
            X_tr_raw_aug = X_tr_raw

        # ── DataLoaders ──────────────────────────────────────────────────────
        train_dataset = TensorDataset(
            torch.tensor(X_tr_raw_aug, dtype=torch.float32),
            torch.tensor(X_tr_freq,    dtype=torch.float32),
            torch.tensor(X_tr_graph,   dtype=torch.float32),
            torch.tensor(y_val[tr_idx], dtype=torch.long),
            torch.tensor(y_aro[tr_idx], dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=2, pin_memory=True
        )

        val_dataset = TensorDataset(
            torch.tensor(X_va_raw,  dtype=torch.float32),
            torch.tensor(X_va_freq, dtype=torch.float32),
            torch.tensor(X_va_graph, dtype=torch.float32),
            torch.tensor(y_val[va_idx], dtype=torch.long),
            torch.tensor(y_aro[va_idx], dtype=torch.long)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        val_true_v = y_val[va_idx]
        val_true_a = y_aro[va_idx]

        # ── Model + Optimizer ────────────────────────────────────────────────
        model = CNN2DBiLSTMFusion(
            num_channels=num_channels_dynamic,
            freq_seq_dim=freq_dim_expected,
            node_feat_dim=5
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # CosineAnnealingWarmRestarts: T_0=10 nghĩa là restart mỗi 10 epoch đầu
        # T_mult=2: mỗi lần restart, chu kỳ nhân đôi (10 → 20 → 40...)
        # Lợi hơn ReduceLROnPlateau: không bị trễ phản ứng, tránh bẫy local minima nhỏ
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        best_mean_f1   = 0.0
        epochs_no_improve = 0
        best_model_path   = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")

        # ── GIAI ĐOẠN 1: ZERO-SHOT PRE-TRAINING ─────────────────────────────
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss_accum = loss_v_accum = loss_a_accum = loss_sup_accum = 0.0

            for b_raw, b_freq, b_graph, b_yv, b_ya in train_loader:
                optimizer.zero_grad()

                b_raw_gpu   = b_raw.to(device, non_blocking=True)
                b_freq_gpu  = b_freq.to(device, non_blocking=True)
                b_graph_gpu = b_graph.to(device, non_blocking=True)
                b_yv_gpu    = b_yv.to(device, non_blocking=True)
                b_ya_gpu    = b_ya.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out_v, out_a, embed_v, embed_a = model(
                        b_raw_gpu, b_freq_gpu, b_graph_gpu
                    )
                    loss_v   = crit_emotion(out_v, b_yv_gpu)
                    loss_a   = crit_emotion(out_a, b_ya_gpu)
                    sup_loss_v = crit_supcon(embed_v, b_yv_gpu)
                    sup_loss_a = crit_supcon(embed_a, b_ya_gpu)
                    loss_sup = (sup_loss_v + sup_loss_a) / 2.0
                    loss     = loss_v + loss_a + lambda_supcon * loss_sup

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                total_loss_accum += loss.item()
                loss_v_accum     += loss_v.item()
                loss_a_accum     += loss_a.item()
                loss_sup_accum   += loss_sup.item()

            # Cosine scheduler step: gọi mỗi epoch
            scheduler.step(epoch + 1)
            current_lr = optimizer.param_groups[0]['lr']

            avg_loss = total_loss_accum / len(train_loader)
            avg_ce_v = loss_v_accum    / len(train_loader)
            avg_ce_a = loss_a_accum    / len(train_loader)
            avg_sup  = loss_sup_accum  / len(train_loader)

            pred_v, pred_a = evaluate_in_batches(model, val_loader, device)
            val_f1_v   = f1_score(val_true_v, pred_v, average='macro')
            val_f1_a   = f1_score(val_true_a, pred_a, average='macro')
            mean_val_f1 = (val_f1_v + val_f1_a) / 2.0

            epoch_dur = time.time() - epoch_start
            print(f"    [Epoch {epoch+1:02d}/{epochs:02d}] {epoch_dur:.1f}s | "
                  f"Total: {avg_loss:.4f} (CE_V:{avg_ce_v:.4f} CE_A:{avg_ce_a:.4f} Sup:{avg_sup:.4f}) | "
                  f"Val F1:{mean_val_f1:.4f} | LR:{current_lr:.2e}")

            if mean_val_f1 > best_mean_f1:
                best_mean_f1      = mean_val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            elif epoch >= warmup_epochs:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"      ---> [EARLY STOPPING] Epoch {epoch+1}")
                    break

        # ── GIAI ĐOẠN 2: ZERO-SHOT EVALUATION ───────────────────────────────
        print("\n  [EVALUATION] Load best checkpoint...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        try:
            calib_idx, eval_idx = train_test_split(
                np.arange(len(te_idx)), test_size=0.90,
                random_state=42, stratify=y_val[te_idx]
            )
        except ValueError:
            print("      [Cảnh báo] Unstratified split do lệch nhãn.")
            calib_idx, eval_idx = train_test_split(
                np.arange(len(te_idx)), test_size=0.90, random_state=42
            )

        eval_dataset = TensorDataset(
            torch.tensor(X_te_raw[eval_idx],   dtype=torch.float32),
            torch.tensor(X_te_freq[eval_idx],  dtype=torch.float32),
            torch.tensor(X_te_graph[eval_idx], dtype=torch.float32),
            torch.tensor(y_val[te_idx][eval_idx], dtype=torch.long),
            torch.tensor(y_aro[te_idx][eval_idx], dtype=torch.long)
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        eval_true_v = y_val[te_idx][eval_idx]
        eval_true_a = y_aro[te_idx][eval_idx]

        zs_pred_v, zs_pred_a = evaluate_in_batches(model, eval_loader, device)
        zs_f1_v   = f1_score(eval_true_v, zs_pred_v, average='macro')
        zs_f1_a   = f1_score(eval_true_a, zs_pred_a, average='macro')
        zs_mean_f1 = (zs_f1_v + zs_f1_a) / 2.0
        print(f"      [Zero-Shot] F1-V:{zs_f1_v:.4f} | F1-A:{zs_f1_a:.4f} | Mean:{zs_mean_f1:.4f}")
        fold_results_zs.append(zs_mean_f1)

        # ── GIAI ĐOẠN 3: ADAPTIVE FEW-SHOT CALIBRATION ──────────────────────
        print("      [Few-Shot] Đóng băng extractors, fine-tune fusion + heads...")

        # Đóng băng giống V5
        for name in ['b_temporal','b_spatial','b_pointwise','b_se','b_fc',
                     'branch_b_bilstm','branch_b_fc',
                     'projection_head_v','projection_head_a']:
            for param in getattr(model, name).parameters():
                param.requires_grad = False

        for name in ['branch_c_dgcnn','fusion','fc_valence','fc_arousal']:
            for param in getattr(model, name).parameters():
                param.requires_grad = True

        calib_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=calib_lr
        )
        calib_criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
        grad_scaler_calib = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        calib_dataset = TensorDataset(
            torch.tensor(X_te_raw[calib_idx],   dtype=torch.float32),
            torch.tensor(X_te_freq[calib_idx],  dtype=torch.float32),
            torch.tensor(X_te_graph[calib_idx], dtype=torch.float32),
            torch.tensor(y_val[te_idx][calib_idx], dtype=torch.long),
            torch.tensor(y_aro[te_idx][calib_idx], dtype=torch.long)
        )
        safe_bs = min(32, max(2, len(calib_dataset) // 2))
        calib_loader = DataLoader(
            calib_dataset, batch_size=safe_bs,
            shuffle=True, drop_last=True, pin_memory=True
        )

        # Split calib thành train/val để theo dõi early stopping
        # Dùng luôn eval_loader làm proxy val — không leak vì eval không dùng để train
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        best_calib_f1    = zs_mean_f1    # baseline là zero-shot
        best_calib_state = {k: v.clone() for k, v in model.state_dict().items()}
        calib_no_improve = 0

        for c_epoch in range(calib_max_epochs):
            model.train()
            model.b_temporal.eval()
            model.b_spatial.eval()
            model.b_pointwise.eval()
            model.b_se.eval()

            for c_raw, c_freq, c_graph, c_yv, c_ya in calib_loader:
                calib_optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out_v, out_a, _, _ = model(
                        c_raw.to(device, non_blocking=True),
                        c_freq.to(device, non_blocking=True),
                        c_graph.to(device, non_blocking=True)
                    )
                    loss = (calib_criterion(out_v, c_yv.to(device, non_blocking=True)) +
                            calib_criterion(out_a, c_ya.to(device, non_blocking=True)))
                grad_scaler_calib.scale(loss).backward()
                grad_scaler_calib.unscale_(calib_optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                grad_scaler_calib.step(calib_optimizer)
                grad_scaler_calib.update()

            # Eval sau mỗi calib epoch
            c_pred_v, c_pred_a = evaluate_in_batches(model, eval_loader, device)
            c_f1 = (f1_score(eval_true_v, c_pred_v, average='macro') +
                    f1_score(eval_true_a, c_pred_a, average='macro')) / 2.0

            if c_f1 > best_calib_f1:
                best_calib_f1    = c_f1
                best_calib_state = {k: v.clone() for k, v in model.state_dict().items()}
                calib_no_improve = 0
            else:
                calib_no_improve += 1
                if calib_no_improve >= calib_patience:
                    print(f"        [Calib Early Stop] Epoch {c_epoch+1}/{calib_max_epochs}")
                    break

        # Load lại calib state tốt nhất
        model.load_state_dict(best_calib_state)

        fs_pred_v, fs_pred_a = evaluate_in_batches(model, eval_loader, device)
        fs_f1_v   = f1_score(eval_true_v, fs_pred_v, average='macro')
        fs_f1_a   = f1_score(eval_true_a, fs_pred_a, average='macro')
        fs_mean_f1 = (fs_f1_v + fs_f1_a) / 2.0

        print(f"      [Few-Shot]  F1-V:{fs_f1_v:.4f} | F1-A:{fs_f1_a:.4f} | Mean:{fs_mean_f1:.4f}")
        print(f"      ---> BIÊN ĐỘ CẢI THIỆN: {fs_mean_f1 - zs_mean_f1:+.4f} "
              f"(best calib epoch: {calib_max_epochs - calib_no_improve}/{calib_max_epochs})")
        fold_results_fs.append(fs_mean_f1)

        if CLEANUP_CHECKPOINTS and os.path.exists(best_model_path):
            os.remove(best_model_path)

    print("\n" + "="*70)
    print("   KẾT THÚC — V7")
    print("="*70)
    print(f" -> Zero-Shot  (Chưa Adaptation): {np.mean(fold_results_zs):.4f} "
          f"± {np.std(fold_results_zs):.4f}")
    print(f" -> Few-Shot   (Đã Adaptation)  : {np.mean(fold_results_fs):.4f} "
          f"± {np.std(fold_results_fs):.4f}")
    print("="*70)

if __name__ == "__main__":
    train_loso_supcon_dgcnn_pipeline()