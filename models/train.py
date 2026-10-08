import torch
import torch.nn as nn
import torch.optim as optim

def greedy_decode(log_probs, idx2char, blank_idx=0):
    """
    Decodifica la salida log_softmax del modelo (T, N, C) a cadenas de texto.
    """
    preds = torch.argmax(log_probs, dim=2).permute(1, 0)
    
    decoded_texts = []
    for seq in preds:
        char_list = []
        prev_idx = None
        for idx in seq.tolist():
            if idx != blank_idx and idx != prev_idx:
                char_list.append(idx2char[idx])
            prev_idx = idx
        decoded_texts.append("".join(char_list))
        
    return decoded_texts


def calculate_cer(predictions, targets):
    """
    Calcula el Character Error Rate (CER) basado en la distancia de Levenshtein simple.
    """
    total_dist = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        dp = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(target) + 1):
            dp[0][j] = j
            
        for i in range(1, len(pred) + 1):
            for j in range(1, len(target) + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                    
        total_dist += dp[len(pred)][len(target)]
        total_chars += max(len(target), 1)
        
    return total_dist / total_chars

def validate_epoch(model, val_loader, criterion, device, idx2char, blank_idx=0):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths, labels_text in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            log_probs = model(images)
            
            T, batch_size, _ = log_probs.shape
            input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long, device=device)
            
            # Loss CTC
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()
            
            preds_text = greedy_decode(log_probs, idx2char, blank_idx=blank_idx)
            all_preds.extend(preds_text)
            all_targets.extend(labels_text)
            
    avg_loss = val_loss / len(val_loader)
    cer = calculate_cer(all_preds, all_targets)
    
    return avg_loss, cer, all_preds, all_targets


def train_model(model, train_loader, val_loader, idx2char, epochs=20, lr=1e-3, device='cuda'):
    model = model.to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            log_probs = model(images)
            
            T, batch_size, _ = log_probs.shape
            input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validación
        avg_val_loss, val_cer, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, idx2char, blank_idx=0
        )
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_crnn_model.pth")
            saved_str = " -> ¡Modelo Guardado!"
        else:
            saved_str = ""

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER: {val_cer:.4f}{saved_str}")
        
        print(f"  Real : {val_targets[:2]}")
        print(f"  Pred : {val_preds[:2]}")
        print("-" * 60)