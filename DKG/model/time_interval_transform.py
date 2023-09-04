import torch
import torch.nn as nn


class TimeIntervalTransform(nn.Module):
    EPSILON = 1e-10

    def __init__(self, log_transform=True, normalize=False, time_intervals=None):
        super().__init__()

        self.enc_dim = 1
        self.log_transform = log_transform
        self.normalize = normalize

        if self.log_transform and time_intervals is not None:
            self.time_intervals = torch.log(time_intervals + self.EPSILON)
        else:
            self.time_intervals = time_intervals

        if self.normalize:
            self.mean_time_interval = self.time_intervals.mean()
            self.std_time_interval = self.time_intervals.std()

    def forward(self, time_intervals):
        return self.transform(time_intervals)

    def transform(self, time_intervals):
        if self.log_transform:
            time_intervals = torch.log(time_intervals + self.EPSILON)

        if self.normalize:
            return (time_intervals - self.mean_time_interval) / self.std_time_interval
        else:
            return time_intervals

    def reverse_transform(self, time_intervals):
        if self.log_transform:
            time_intervals = torch.exp(time_intervals)

        if self.normalize:
            return time_intervals * self.std_time_interval + self.mean_time_interval
        else:
            return time_intervals

    def __repr__(self):
        field_desc = [f"log_transform={self.log_transform}", f"normalize={self.normalize}"]
        return f"{self.__class__.__name__}({', '.join(field_desc)})"
