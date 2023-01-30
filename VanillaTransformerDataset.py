
class VanillaTransformerDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.input = df['input'].values
        self.target = df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.input[index]
        target = self.target[index]

        return {
            'input': text,
            'target': target
        }