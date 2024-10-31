from inference_inat import test_inat
import numpy as np

# interpolate between two models (open_clip)
def patch(finetuned_model, zero_shot_model, tokenizer, img_dir, json_path, interp_interval=10, batch_size=512, num_workers=8, device='cuda'):
    alpha = np.linspace(0, 1, interp_interval)
    theta_finetuned = {k: v.clone() for k, v in finetuned_model.state_dict().items()}
    theta_zero_shot = {k: v.clone() for k, v in zero_shot_model.state_dict().items()}

    acc = []

    for i in range(interp_interval):
        theta = {k: alpha[i]*theta_finetuned[k] + (1-alpha[i])*theta_zero_shot[k] for k in theta_finetuned.keys()}
        finetuned_model.load_state_dict(theta)
        with torch.no_grad():
            acc.append(test_inat(finetuned_model, tokenizer, img_dir, json_path, batch_size=batch_size, num_workers=num_workers, device=device))
    
    return alpha[np.argmax(acc)], acc

def patch(finetuned_model, zero_shot_model, tokenizer, img_dir, json_path, interp_interval=10, batch_size=512, num_workers=8, device='cuda'):
    alpha = np.linspace(0, 1, interp_interval)
    theta_finetuned = {k: v.clone() for k, v in finetuned_model.state_dict().items()}
    theta_zero_shot = {k: v.clone() for k, v in zero_shot_model.state_dict().items()}

    acc = []

    for i in range(interp_interval):
        theta = {k: alpha[i]*theta_finetuned[k] + (1-alpha[i])*theta_zero_shot[k] for k in theta_finetuned.keys()}
        finetuned_model.load_state_dict(theta)
        with torch.no_grad():
            acc.append(test_inat(finetuned_model, tokenizer, img_dir, json_path, batch_size=batch_size, num_workers=num_workers, device=device))
    
    return alpha[np.argmax(acc)], acc
