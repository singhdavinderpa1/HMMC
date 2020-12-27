from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

print(sampler
      )