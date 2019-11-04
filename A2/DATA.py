import pandas as pd


class UnlimitedDataWorks:

    def __init__(self, deg):
        self.exp = []
        for i in range(deg+1):
            for j in range(deg+1):
                if i+j <= deg:
                    self.exp.append((i, j))

    def train_test_split(self, dataframe):
        self.data = pd.DataFrame([])
        self.count = -1
        for (a, b) in self.exp:
            self.count += 1
            res = (dataframe["lat"] ** a) * (dataframe["lon"] ** b)
            self.data.insert(self.count, "col" + str(a) + str(b), res, True)

        # min-max normalize the data:
        self.data = (self.data-self.data.min()) / (self.data.max()-self.data.min())
        dataframe = (dataframe-dataframe.min()) / (dataframe.max()-dataframe.min())
        self.data["col00"] = [1.0]*len(self.data)

        # generate a 70-20-10 split on the data:
        X = self.data[:304113]
        Y = dataframe["alt"][:304113]
        xval = self.data[304113:391088]
        yval = dataframe["alt"][304113:391088]
        x = self.data[391088:]
        y = dataframe["alt"][391088:]
        return (X, Y, xval, yval, x, y)
