import torch
def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_r2(preds, labels, null_val):
    """Coefficient of determination (R^2) with the same masking convention as the
    other metrics. Returns NaN when the masked total sum of squares is ~0."""
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    n = torch.sum(mask)
    if n < 1:
        return torch.tensor(float('nan'))
    label_mean = torch.sum(labels * mask) / n
    ss_res = torch.sum(((preds - labels) ** 2) * mask)
    ss_tot = torch.sum(((labels - label_mean) ** 2) * mask)
    if ss_tot <= 1e-12:
        return torch.tensor(float('nan'))
    return 1.0 - ss_res / ss_tot


def masked_ioa(preds, labels, null_val):
    """Willmott index of agreement (IOA) with the same masking convention as the
    other metrics. Returns NaN when the masked denominator is ~0."""
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    n = torch.sum(mask)
    if n < 1:
        return torch.tensor(float('nan'))
    label_mean = torch.sum(labels * mask) / n
    ss_res = torch.sum(((preds - labels) ** 2) * mask)
    den = torch.sum(((torch.abs(preds - label_mean) + torch.abs(labels - label_mean)) ** 2) * mask)
    if den <= 1e-12:
        return torch.tensor(float('nan'))
    return 1.0 - ss_res / den


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    r2 = masked_r2(preds, labels, null_val).item()
    ioa = masked_ioa(preds, labels, null_val).item()
    return mae, mape, rmse, r2, ioa


def masked_fre_mae(preds, labels, null_val=None):
    # print(preds.shape, labels.shape)
    preds = torch.fft.rfft(preds, dim=1)
    labels = torch.fft.rfft(labels, dim=1)
    # print(preds.shape, labels.shape)
    null_val = labels.abs().min() if labels.abs().min() < 1 else torch.tensor(0)
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)