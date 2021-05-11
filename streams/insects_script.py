class RecurringInsectsDataset(Dataset):
    def __init__(self, repetitions):
        self.drift_indices = []
        self.x, self.y = self.load_insects(repetitions)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]

    def get_drift_indices(self):
        return self.drift_indices

    def load_insects(self, repetitions):
        data = arff.loadarff('./data_loaders/datasets/INSECTS-incremental-reoccurring-balanced-norm.arff')
        df = pd.DataFrame(data[0])
        le = preprocessing.LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        data_list = df.values.tolist()
        x = [row[:33] for row in data_list]
        y = [row[-1] for row in data_list]

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        return x, y
